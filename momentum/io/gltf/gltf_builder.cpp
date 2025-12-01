/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/gltf/gltf_builder.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character.h"
#include "momentum/character/character_state.h"
#include "momentum/character/character_utility.h"
#include "momentum/character/collision_geometry_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/filesystem.h"
#include "momentum/common/log.h"
#include "momentum/io/gltf/gltf_io.h"
#include "momentum/io/gltf/utils/accessor_utils.h"
#include "momentum/io/gltf/utils/coordinate_utils.h"
#include "momentum/io/gltf/utils/json_utils.h"
#include "momentum/io/skeleton/locator_io.h"
#include "momentum/io/skeleton/parameter_transform_io.h"
#include "momentum/io/skeleton/parameters_io.h"
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/math/mppca.h"
#include "momentum/math/utility.h"

#include <fx/gltf.h>

#include <variant>

namespace {

using namespace momentum;

nlohmann::json& addMomentumExtension(nlohmann::json& extensionsAndExtras) {
  if (extensionsAndExtras.count("extensions") == 0 ||
      extensionsAndExtras["extensions"].count("FB_momentum") == 0) {
    extensionsAndExtras["extensions"]["FB_momentum"] = nlohmann::json::object();
  }
  return extensionsAndExtras["extensions"]["FB_momentum"];
}

void initDefaultScene(fx::gltf::Document& model) {
  model.scene = 0;
  model.scenes.resize(1);
}

[[nodiscard]] fx::gltf::Scene& getDefaultScene(fx::gltf::Document& model) {
  MT_CHECK(model.scenes.size() == 1, "{}", model.scenes.size());
  return model.scenes.at(0);
}

fx::gltf::Animation& addMomentumAnimationToModel(
    fx::gltf::Document& model,
    const std::string& nameIn = "default") {
  auto animationIter = std::find_if(
      model.animations.begin(), model.animations.end(), [nameIn](const fx::gltf::Animation& anim) {
        return nameIn == anim.name;
      });
  if (animationIter == model.animations.end()) {
    model.animations.emplace_back();
    auto& animation = model.animations.at(model.animations.size() - 1);
    animation.name = nameIn;
    return animation;
  }
  return *animationIter;
}

void setBufferView(
    fx::gltf::Document& model,
    const uint32_t accessorId,
    fx::gltf::BufferView::TargetType targetType) {
  MT_CHECK(accessorId < model.accessors.size());
  const auto accessor = model.accessors[accessorId];
  MT_CHECK(accessor.bufferView < model.bufferViews.size());
  // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
  auto& view = model.bufferViews[accessor.bufferView];
  view.target = targetType;
};

// Add a mesh node with only one primitive (i.e. combined mesh)
// Use a flag to decide if we should save the vertex color. If a mesh only has default color, we may
// not want to save it.
size_t addMeshToModel(fx::gltf::Document& model, const Mesh& mesh, const bool addColor = false) {
  if (mesh.vertices.empty()) {
    return std::numeric_limits<size_t>::max();
  }

  model.meshes.emplace_back();
  auto& m = model.meshes.back();
  m.primitives.emplace_back();
  auto& prim = m.primitives.back();

  // Add faces to mesh
  if ((!mesh.faces.empty()) && (mesh.faces[0].size() > 0)) {
    prim.mode = fx::gltf::Primitive::Mode::Triangles;
    const std::span<const int32_t> faces(&mesh.faces[0][0], mesh.faces.size() * 3);
    prim.indices = createAccessorBuffer<const int32_t>(model, faces);
    setBufferView(model, prim.indices, fx::gltf::BufferView::TargetType::ElementArrayBuffer);
  } else {
    prim.mode = fx::gltf::Primitive::Mode::Points;
  }

  // Add vertices to mesh
  std::vector<Vector3f> verts = mesh.vertices;
  fromMomentumVec3f(verts);
  MT_CHECK((!verts.empty()) && (verts[0].size() > 0));
  prim.attributes["POSITION"] = createAccessorBuffer<const Vector3f>(model, verts, true);
  setBufferView(model, prim.attributes["POSITION"], fx::gltf::BufferView::TargetType::ArrayBuffer);

  // need min/max for position accessor
  Box3f bbox;
  for (const auto& v : verts) {
    bbox.extend(v);
  }
  for (size_t d = 0; d < 3; d++) {
    model.accessors.at(prim.attributes["POSITION"]).min.emplace_back(bbox.min()[d]);
    model.accessors.at(prim.attributes["POSITION"]).max.emplace_back(bbox.max()[d]);
  }

  // add normal attribute
  if (mesh.normals.size() == mesh.vertices.size()) {
    prim.attributes["NORMAL"] = createAccessorBuffer<const Vector3f>(model, mesh.normals, true);
    setBufferView(model, prim.attributes["NORMAL"], fx::gltf::BufferView::TargetType::ArrayBuffer);
  }

  // add texcoord attribute
  if (!mesh.texcoords.empty()) {
    // NOTE: GLTF doesn't support storing multiple texture coordinates per vertex, so instead of
    //       using the standard GLTF texture arrays we store the data in a custom location
    auto& def = addMomentumExtension(prim.extensionsAndExtras);
    def["texcoords"] = createAccessorBuffer<const Vector2f>(model, mesh.texcoords, true);
    setBufferView(model, def["texcoords"], fx::gltf::BufferView::TargetType::ArrayBuffer);

    const std::span<const int32_t> texFaces(
        &mesh.texcoord_faces[0][0], mesh.texcoord_faces.size() * 3);
    def["texfaces"] = createAccessorBuffer<const int32_t>(model, texFaces);
    setBufferView(model, def["texfaces"], fx::gltf::BufferView::TargetType::ArrayBuffer);
  }

  if (mesh.colors.size() == mesh.vertices.size() && addColor) {
    prim.attributes["COLOR_0"] =
        createAccessorBuffer<const Vector3b>(model, mesh.colors, true, true);
    setBufferView(model, prim.attributes["COLOR_0"], fx::gltf::BufferView::TargetType::ArrayBuffer);
  }

  return model.meshes.size() - 1;
}

void addMorphTargetsToModel(
    fx::gltf::Document& model,
    const BlendShape& shapes,
    size_t numShapes,
    size_t meshIndex) {
  if (model.meshes.empty() || meshIndex == kInvalidIndex) {
    MT_LOGW("No mesh is saved in the file when adding blendshapes");
    return;
  }

  // assume there is only one primitive on the character mesh.
  auto& prim = model.meshes.at(model.nodes.at(meshIndex).mesh).primitives.at(0);
  size_t numShapesAdded = prim.targets.size();
  if (numShapesAdded >= numShapes) {
    // enough shapes already exist; nothing to be done.
    return;
  }
  auto& def = addMomentumExtension(prim.extensionsAndExtras);
  def["shapeNames"] = shapes.getShapeNames();

  const MatrixXf& shapeVecs = shapes.getShapeVectors();
  for (size_t iShape = numShapesAdded; iShape < numShapes; ++iShape) {
    // make a copy so we can scale it
    VectorXf deltas = shapeVecs.col(iShape);
    fromMomentumVec3f(deltas);
    prim.targets.emplace_back();
    auto& attr = prim.targets.back();

    std::span<const Eigen::Vector3f> span(
        reinterpret_cast<Eigen::Vector3f*>(deltas.data()), static_cast<size_t>(deltas.size() / 3));
    attr["POSITION"] = createAccessorBuffer<const Vector3f>(model, span, true);
    setBufferView(model, attr["POSITION"], fx::gltf::BufferView::TargetType::ArrayBuffer);

    // set bbox
    Box3f bbox;
    for (const Vector3f& v : span) {
      bbox.extend(v);
    }
    for (size_t d = 0; d < 3; ++d) {
      model.accessors.at(attr["POSITION"]).min.emplace_back(bbox.min()[d]);
      model.accessors.at(attr["POSITION"]).max.emplace_back(bbox.max()[d]);
    }
  }
}

void addMorphWeightsToModel(
    fx::gltf::Document& model,
    const Character& character,
    const float fps,
    const MatrixXf& modelParams,
    const size_t meshIndex,
    const std::string& motionName = "default") {
  const size_t numFrames = modelParams.cols();
  if (numFrames == 0) {
    return;
  }

  const auto& pt = character.parameterTransform;
  const size_t numWeights = pt.numBlendShapeParameters();
  if (numWeights == 0) {
    return;
  }

  // validate the mesh node
  if (model.meshes[model.nodes[meshIndex].mesh].primitives.size() != 1 ||
      model.meshes[model.nodes[meshIndex].mesh].primitives.at(0).targets.size() != numWeights) {
    MT_LOGW("No valid mesh to add blendshape animation for");
    return;
  }

  // extract weight values
  std::vector<float> timestamps(numFrames, 0);
  bool useChannel = false;
  // Important to make sure row vs col is what createSampler expects
  MatrixXf weights(numWeights, numFrames);

  for (size_t iFrame = 0; iFrame < numFrames; ++iFrame) {
    const VectorXf& pose = modelParams.col(iFrame);
    const VectorXf w = extractBlendWeights(pt, ModelParameters(pose)).v;
    if (w.norm() > 1e-5f) {
      useChannel = true;
    }
    weights.col(iFrame) = w;
    timestamps[iFrame] = static_cast<float>(iFrame) / fps;
  }
  if (!useChannel) {
    return;
  }

  // store weight values in gltf document
  auto& animation = addMomentumAnimationToModel(model, motionName);
  const auto timestampIdx = createAccessorBuffer<const float>(model, timestamps);
  if (!timestamps.empty() && !model.accessors.empty() && timestampIdx < model.accessors.size()) {
    model.accessors[timestampIdx].min.emplace_back(timestamps.front());
    model.accessors[timestampIdx].max.emplace_back(timestamps.back());
  }

  animation.channels.emplace_back();
  auto& channel = animation.channels.back();
  channel.sampler = createSampler<const float>(
      model,
      animation,
      std::span<const float>(weights.data(), numWeights * numFrames),
      timestampIdx);
  channel.target.node = meshIndex;
  channel.target.path = "weights";
}

Mesh createUnitCube(const Eigen::Vector3b& color) {
  Mesh cube;
  cube.vertices = {
      Vector3f(-1.0f, -1.0f, 1.0f),
      Vector3f(1.0f, -1.0f, 1.0f),
      Vector3f(1.0f, 1.0f, 1.0f),
      Vector3f(-1.0f, 1.0f, 1.0f),
      Vector3f(-1.0f, -1.0f, -1.0f),
      Vector3f(1.0f, -1.0f, -1.0f),
      Vector3f(1.0f, 1.0f, -1.0f),
      Vector3f(-1.0f, 1.0f, -1.0f)};
  cube.faces = {
      Vector3i(0, 1, 2),
      Vector3i(2, 3, 0),
      Vector3i(1, 5, 6),
      Vector3i(6, 2, 1),
      Vector3i(7, 6, 5),
      Vector3i(5, 4, 7),
      Vector3i(4, 0, 3),
      Vector3i(3, 7, 4),
      Vector3i(4, 5, 1),
      Vector3i(1, 0, 4),
      Vector3i(3, 2, 6),
      Vector3i(6, 7, 3)};
  cube.colors = std::vector<Eigen::Vector3b>(8, color);
  return cube;
}
const auto kUnitCubeRed = createUnitCube(Eigen::Vector3b(255, 0, 0));
const auto kUnitCubeGreen = createUnitCube(Eigen::Vector3b(0, 255, 0));

void addActorAnimationToModel(
    fx::gltf::Document& model,
    const float fps,
    std::span<const std::vector<momentum::Marker>> markerSequence,
    const GltfBuilder::MarkerMesh markerMesh,
    const std::string& animName) {
  if (markerSequence.empty()) {
    return;
  }

  // create arrays for data
  std::map<std::string, size_t> markerNameToIndex;
  std::vector<std::string> markerNames;
  std::vector<std::vector<float>> timestamps;
  std::vector<std::vector<Vector3f>> markerPositions;
  for (size_t i = 0; i < markerSequence.size(); i++) {
    const float timestamp = static_cast<float>(i) / fps;
    for (const auto& marker : markerSequence[i]) {
      if (marker.occluded) {
        continue;
      }

      // create new array if marker is unknown
      if (markerNameToIndex.count(marker.name) == 0) {
        const auto index = timestamps.size();
        timestamps.emplace_back();
        markerPositions.emplace_back();
        markerNameToIndex[marker.name] = index;
        markerNames.emplace_back(marker.name);
      }

      // get index
      const auto& index = markerNameToIndex.at(marker.name);
      timestamps[index].push_back(timestamp);
      markerPositions[index].push_back(fromMomentumVec3f(marker.pos.cast<float>()));
    }
  }

  // add data to gltf document
  auto& scene = getDefaultScene(model);
  auto& animation = addMomentumAnimationToModel(model, animName);

  MT_CHECK(
      (markerNames.size() == timestamps.size()) && (markerNames.size() == markerPositions.size()) &&
      (markerNames.size() == markerNameToIndex.size()));

  for (size_t j = 0; j < timestamps.size(); j++) {
    // create marker
    const auto nodeIndex = model.nodes.size();
    model.nodes.emplace_back();
    auto& node = model.nodes.back();
    scene.nodes.push_back(nodeIndex);
    node.name = markerNames[j];
    node.translation = {0.0f, 0.0f, 0.0f};
    node.extensionsAndExtras["extensions"]["FB_momentum"]["type"] = "marker";
    switch (markerMesh) {
      case GltfBuilder::MarkerMesh::UnitCube: {
        auto newMeshIdx = addMeshToModel(model, kUnitCubeRed, true);
        if (newMeshIdx < model.meshes.size()) {
          node.mesh = newMeshIdx;
        }
        break;
      }
      default:
        break;
    }

    // create animation channel
    const auto timestampIdx = createAccessorBuffer<const float>(model, timestamps[j]);
    if (!timestamps[j].empty() && !model.accessors.empty() &&
        timestampIdx < model.accessors.size()) {
      model.accessors[timestampIdx].min.emplace_back(timestamps[j].front());
      model.accessors[timestampIdx].max.emplace_back(timestamps[j].back());
    }

    animation.channels.emplace_back();
    auto& tchannel = animation.channels.back();
    tchannel.sampler =
        createSampler<const Vector3f>(model, animation, markerPositions[j], timestampIdx);
    tchannel.target.node = nodeIndex;
    tchannel.target.path = "translation";
  }
}

size_t addMeshToModel(
    fx::gltf::Document& model,
    const Character& character,
    const std::vector<size_t>& jointToNodeMap,
    const size_t rootNodeIdx) {
  // write mesh node/skinning data
  if (character.mesh && character.skinWeights) {
    // create new mesh node
    const auto nodeIdx = model.nodes.size();
    model.nodes.emplace_back();
    auto& node = model.nodes.back();
    model.nodes[rootNodeIdx].children.push_back(nodeIdx);

    // create model mesh
    const auto& mesh = *character.mesh;
    auto newMeshIdx = addMeshToModel(model, mesh);
    if (newMeshIdx >= model.meshes.size()) {
      // Mesh not valid, skip
      return kInvalidIndex;
    }
    node.mesh = newMeshIdx;
    auto& m = model.meshes[node.mesh];
    MT_CHECK(m.primitives.size() == 1);
    auto& prim = m.primitives.back();

    const auto& skin = *character.skinWeights;
    model.skins.emplace_back();
    auto& sk = model.skins.back();

    // add skin node
    for (const auto& ji : jointToNodeMap) {
      sk.joints.push_back(ji);
    }
    sk.skeleton = sk.joints[0];

    node.name = "mesh";
    node.skin = model.skins.size() - 1;

    // add inverse bind matrix
    std::vector<Matrix4f> ibm;
    for (const auto& mat : character.inverseBindPose) {
      auto data = mat;
      data.translation() = fromMomentumVec3f(data.translation());
      ibm.emplace_back(data.matrix());
    }
    sk.inverseBindMatrices = createAccessorBuffer<const Matrix4f>(model, ibm);

    // add skinning data
    int weightSets = 1;
    if (skin.weight.block(0, 4, skin.weight.rows(), 4).maxCoeff() > 0.001f) {
      // only export two sets of weights if more than one set is being used
      // only one vertex on the second set would have to have more than 0.1% influence to qualify
      weightSets = 2;
    }
    for (size_t b = 0; b < weightSets; b++) {
      std::vector<Vector4s> indices(skin.index.rows());
      std::vector<Vector4f> weights(skin.index.rows());
      for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = Vector4s(
            skin.index(i, b * 4 + 0),
            skin.index(i, b * 4 + 1),
            skin.index(i, b * 4 + 2),
            skin.index(i, b * 4 + 3));
        weights[i] = Vector4f(
            skin.weight(i, b * 4 + 0),
            skin.weight(i, b * 4 + 1),
            skin.weight(i, b * 4 + 2),
            skin.weight(i, b * 4 + 3));
      }

      const auto kIndicesAttribute = std::string("JOINTS_") + std::to_string(b);
      prim.attributes[kIndicesAttribute] = createAccessorBuffer<const Vector4s>(model, indices);
      setBufferView(
          model, prim.attributes[kIndicesAttribute], fx::gltf::BufferView::TargetType::ArrayBuffer);

      const auto kWeightsAttribute = std::string("WEIGHTS_") + std::to_string(b);
      prim.attributes[kWeightsAttribute] = createAccessorBuffer<const Vector4f>(model, weights);
      setBufferView(
          model, prim.attributes[kWeightsAttribute], fx::gltf::BufferView::TargetType::ArrayBuffer);
    }
    return nodeIdx;
  }
  return kInvalidIndex;
}

void addSkeletonStatesToModel(
    fx::gltf::Document& model,
    const Character& character,
    const float fps,
    std::span<const SkeletonState> skeletonStates,
    std::span<const size_t> jointToNodeMap,
    const std::string& motionName = "default") {
  const auto numFrames = skeletonStates.size();
  if (numFrames == 0) {
    return;
  }

  // check for valid character and stuff before adding things
  MT_THROW_IF(
      character.skeleton.joints.empty() || character.parameterTransform.numJointParameters() == 0,
      "Character is not valid when trying to add skeleton states to model");

  // store parameterized motion into joints for actual animation
  const auto numJoints = character.skeleton.joints.size();
  std::vector<float> timestamps(numFrames);
  std::vector<Vector3<bool>> useChannel(numJoints, Vector3<bool>::Constant(false));
  std::vector<std::vector<Vector3f>> translation(numJoints, std::vector<Vector3f>(numFrames));
  std::vector<std::vector<Vector4f>> rotation(numJoints, std::vector<Vector4f>(numFrames));
  std::vector<std::vector<Vector3f>> scale(numJoints, std::vector<Vector3f>(numFrames));

  // create motion data using the mapped data
  bool foundAnyChannel = false;
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(numFrames); i++) {
    for (size_t j = 0; j < numJoints; j++) {
      auto localT = skeletonStates[i].jointState[j].localTranslation().cast<float>();
      if (!(localT - character.skeleton.joints[j].translationOffset).isZero(1e-5f)) {
        useChannel[j][0] = true;
        foundAnyChannel = true;
      }
      translation[j][i] = fromMomentumVec3f(localT);

      auto localR = skeletonStates[i].jointState[j].localRotation().cast<float>();
      rotation[j][i] = {localR.x(), localR.y(), localR.z(), localR.w()};
      if (!(localR.toRotationMatrix() - character.skeleton.joints[j].preRotation.toRotationMatrix())
               .isZero(1e-5f)) {
        useChannel[j][1] = true;
        foundAnyChannel = true;
      }

      float localScale = skeletonStates[i].jointState[j].localScale();
      scale[j][i] = Vector3f::Ones() * localScale;
      if (fabs(localScale - 1.0f) > 1e-5f) {
        useChannel[j][2] = true;
        foundAnyChannel = true;
      }
    }
    timestamps[i] = static_cast<float>(i) / fps;
  }

  // fx-gltf has a bug where if you write an animation that has no actual
  // animated channels, it throws an error "Required field not found : channels".
  // So we'll just make sure there is at least one channel in this case.
  if (!foundAnyChannel && !useChannel.empty()) {
    useChannel[0][0] = true;
  }

  // add data to gltf document
  auto& animation = addMomentumAnimationToModel(model, motionName);
  const auto timestampIdx = createAccessorBuffer<const float>(model, timestamps);
  if (!timestamps.empty() && !model.accessors.empty() && timestampIdx < model.accessors.size()) {
    model.accessors[timestampIdx].min.emplace_back(timestamps.front());
    model.accessors[timestampIdx].max.emplace_back(timestamps.back());
  }

  // only save the animation channels that actually have any useful data
  for (size_t j = 0; j < numJoints; j++) {
    if (useChannel[j][0]) {
      animation.channels.emplace_back();
      auto& tchannel = animation.channels.back();
      tchannel.sampler =
          createSampler<const Vector3f>(model, animation, translation[j], timestampIdx);
      tchannel.target.node = jointToNodeMap[j];
      tchannel.target.path = "translation";
    }

    if (useChannel[j][1]) {
      animation.channels.emplace_back();
      auto& rchannel = animation.channels.back();
      rchannel.sampler = createSampler<const Vector4f>(model, animation, rotation[j], timestampIdx);
      rchannel.target.node = jointToNodeMap[j];
      rchannel.target.path = "rotation";
    }

    if (useChannel[j][2]) {
      animation.channels.emplace_back();
      auto& schannel = animation.channels.back();
      schannel.sampler = createSampler<const Vector3f>(model, animation, scale[j], timestampIdx);
      schannel.target.node = jointToNodeMap[j];
      schannel.target.path = "scale";
    }
  }
}

void addMotionToModel(
    fx::gltf::Document& model,
    const Character& character,
    const float fps,
    const MotionParameters& inputMotion,
    const IdentityParameters& inputIdentity,
    const std::vector<size_t>& jointToNodeMap,
    const size_t meshIndex,
    const bool addExtension = true,
    const std::string& motionName = "default") {
  // disentangle input values for storage
  const auto& parameterNames = std::get<0>(inputMotion);
  const auto& motion = std::get<1>(inputMotion);
  const auto& jointNames = std::get<0>(inputIdentity);
  const auto& identity = std::get<1>(inputIdentity);

  const auto numFrames = motion.cols();
  if (numFrames == 0) {
    return;
  }

  // **** Save animation to Momentum extension ****
  // add motion if we have poses
  bool fullAnimation = false;
  if (addExtension) {
    auto& def = addMomentumExtension(model.extensionsAndExtras);
    // store motion array and offsets as metadata
    def["motion"]["nframes"] = numFrames;
    if (!parameterNames.empty() && motion.size() > 0) {
      def["motion"]["parameterNames"] = parameterNames;
      def["motion"]["poses"] = createAccessorBuffer<const float>(
          model, std::span<const float>(motion.data(), motion.size()));
      fullAnimation = true;
    }

    if (!jointNames.empty() && identity.size() > 0) {
      def["motion"]["jointNames"] = jointNames;
      def["motion"]["offsets"] = createAccessorBuffer<const float>(
          model, std::span<const float>(identity.data(), identity.size()));
      fullAnimation = true;
    }
  } else {
    // There may be animation outside of the momentum machinery
    fullAnimation = true;
  }

  // **** Save animation to gltf ****
  // add blenshapes (that are used) if available
  if (meshIndex != kInvalidIndex && character.blendShape != nullptr) {
    addMorphTargetsToModel(
        model,
        *character.blendShape.get(),
        character.parameterTransform.numBlendShapeParameters(),
        meshIndex);

    addMorphWeightsToModel(model, character, fps, motion, meshIndex, motionName);
  }

  // check for valid character and motion before adding joint animation
  if (!fullAnimation || character.skeleton.joints.empty()) {
    return;
  }

  // map data to character
  const auto inputPoses = mapMotionToCharacter(inputMotion, character);
  const auto inputOffset = mapIdentityToCharacter(inputIdentity, character);

  // create skeleton states from model params
  std::vector<momentum::SkeletonState> skeletonStates(numFrames);
  CharacterParameters params;
  params.offsets = inputOffset;
  for (Eigen::Index i = 0; i < numFrames; i++) {
    params.pose = inputPoses.col(i);
    const CharacterState state(params, character, false, false, false);
    skeletonStates[i] = state.skeletonState;
  }

  // add skeleton states to model
  addSkeletonStatesToModel(model, character, fps, skeletonStates, jointToNodeMap, motionName);
}

size_t appendNode(fx::gltf::Document& document, const std::string& name) {
  auto nodeIdx = document.nodes.size();
  document.nodes.emplace_back();
  auto& node = document.nodes.back();
  node.name = name;
  return nodeIdx;
}

std::vector<size_t> addSkeletonToModel(
    fx::gltf::Document& model,
    const Character& character,
    const bool updateExtension = true,
    size_t modelRootNodeIndex = kInvalidIndex) {
  // add all joints to node list
  std::vector<size_t> jointToNodeMap;

  for (size_t i = 0; i < character.skeleton.joints.size(); i++) {
    const auto& joint = character.skeleton.joints[i];

    // create new node
    const auto nodeIndex = appendNode(model, joint.name);
    auto& node = model.nodes[nodeIndex];
    if (i == 0) {
      if (modelRootNodeIndex != kInvalidIndex && modelRootNodeIndex < model.nodes.size()) {
        model.nodes[modelRootNodeIndex].children.push_back(nodeIndex);
      } else {
        auto& scene = getDefaultScene(model);
        scene.nodes.push_back(nodeIndex);
      }
    }

    // set joint values in node
    node.rotation = fromMomentumQuaternionf(joint.preRotation);
    const auto translation = fromMomentumVec3f(joint.translationOffset);
    node.translation = {translation[0], translation[1], translation[2]};

    if (updateExtension) {
      auto& extension = addMomentumExtension(node.extensionsAndExtras);
      extension["type"] = std::string("skeleton_joint");
    }

    // add node as child to parent joint
    if (joint.parent != kInvalidIndex) {
      model.nodes.at(jointToNodeMap.at(joint.parent)).children.push_back(nodeIndex);
    }

    // add node to map
    jointToNodeMap.push_back(nodeIndex);
  }

  return jointToNodeMap;
}

void addCollisionsToModel(
    fx::gltf::Document& model,
    const Character& character,
    const std::vector<size_t>& jointToNodeMap,
    bool addExtensions) {
  // add all collision objects to node list
  if (character.collision) {
    for (size_t i = 0; i < character.collision->size(); i++) {
      const auto& col = (*character.collision)[i];

      MT_CHECK(col.parent != kInvalidIndex);

      // create new node
      const auto nodeIndex = model.nodes.size();
      model.nodes.emplace_back();
      auto& node = model.nodes.back();

      // add node as child to parent joint
      model.nodes.at(jointToNodeMap.at(col.parent)).children.push_back(nodeIndex);

      // add values for the tapered capsule
      node.name = character.skeleton.joints.at(col.parent).name + "_col";

      node.rotation = fromMomentumQuaternionf(col.transformation.rotation);
      const auto translation = fromMomentumVec3f(col.transformation.translation);
      node.translation = {translation[0], translation[1], translation[2]};

      // add extra properties
      if (addExtensions) {
        auto& extension = addMomentumExtension(node.extensionsAndExtras);
        extension["type"] = "collision_capsule";
        extension["length"] = col.length;
        extension["radius"] = col.radius;
      }
    }
  }
}

void addLocatorsToModel(
    fx::gltf::Document& model,
    const Character& character,
    const std::vector<size_t>& jointToNodeMap,
    bool addExtensions) {
  // add locators
  for (const auto& loc : character.locators) {
    MT_CHECK(loc.parent != kInvalidIndex);

    // create new node
    const auto nodeIndex = model.nodes.size();
    model.nodes.emplace_back();
    auto& node = model.nodes.back();

    // add node as child to parent joint
    model.nodes[jointToNodeMap[loc.parent]].children.push_back(nodeIndex);

    // add values for the tapered capsule
    node.name = loc.name;
    const auto translation = fromMomentumVec3f(loc.offset);
    node.translation = {translation[0], translation[1], translation[2]};

    // add extra properties
    if (addExtensions) {
      auto& extension = addMomentumExtension(node.extensionsAndExtras);
      extension["type"] = "locator";
      extension["weight"] = loc.weight;
      extension["limitWeight"] = loc.limitWeight;
      extension["limitOrigin"] = loc.limitOrigin;
      extension["locked"] = loc.locked;
      if (loc.attachedToSkin) {
        extension["attachedToSkin"] = loc.attachedToSkin;
      }
      if (loc.skinOffset != 0.0f) {
        extension["skinOffset"] = loc.skinOffset;
      }
    }

    auto newMeshIdx = addMeshToModel(model, kUnitCubeGreen, true);
    if (newMeshIdx < model.meshes.size()) {
      node.mesh = newMeshIdx;
    }
  }
}

void saveDocument(
    const filesystem::path& filename,
    fx::gltf::Document& model,
    const GltfFileFormat fileFormat) {
  // save model
  auto deducedFileFormat = fileFormat;
  if (fileFormat == GltfFileFormat::Auto && filename.extension() == filesystem::path(".gltf")) {
    deducedFileFormat = GltfFileFormat::Ascii;
  }

  try {
    // Check if the parent directory of the file exists. If the parent directory does not exist
    // and is not an empty string (which means the file is in the current working directory),
    // create it. This is necessary because fx::gltf::Save throws an exception if the directory
    // does not exist.
    const filesystem::path parentDir = filename.parent_path();
    if (!parentDir.empty() && !filesystem::exists(parentDir)) {
      filesystem::create_directories(parentDir);
    }

    switch (deducedFileFormat) {
      case GltfFileFormat::Ascii: {
        // we can't embed binary if we writing to ascii format.
        // so we need to provide exported with a uri name relative to json root.
        filesystem::path new_filename = filename;
        new_filename.replace_extension(filesystem::path("glbin"));
        if (!model.buffers.empty()) {
          model.buffers.front().uri = new_filename.filename().string();
        }

        MT_LOGE(
            "There are some compatibility issues with .gltf files. Not all extension data might be exported correctly.");
        fx::gltf::Save(model, filename.string(), false);
        break;
      }
      case GltfFileFormat::Auto:
        [[fallthrough]];
      case GltfFileFormat::Binary:
        fx::gltf::Save(model, filename.string(), true);
        break;
    }
  } catch (const std::exception& e) {
    MT_THROW("Failed to save file to: {}. Error: {}", filename.string(), e.what());
  } catch (...) {
    std::rethrow_exception(std::current_exception());
  }

  MT_THROW_IF(!filesystem::exists(filename), "Unable to save: {}", filename.string());
}

} // namespace

namespace momentum {

struct GltfBuilder::Impl {
  struct CharacterData {
    std::vector<size_t> nodeMap = {};
    size_t rootIndex = kInvalidIndex;
    // node index of the mesh node; needed for creating blendshape morph targets
    size_t meshIndex = kInvalidIndex;
    std::vector<size_t> animationIndices = {};
  };

  Impl();
  fx::gltf::Document document;
  std::unordered_map<std::string, CharacterData> characterData;
  MarkerMesh markerMesh = MarkerMesh::None;
};

GltfBuilder::Impl::Impl() : document(fx::gltf::Document()) {
  document.extensionsUsed.push_back("FB_momentum");
}

GltfBuilder::GltfBuilder() {
  impl_ = std::make_unique<GltfBuilder::Impl>();
  initDefaultScene(impl_->document);
}

GltfBuilder::~GltfBuilder() = default;

void GltfBuilder::addCharacter(
    const Character& character,
    const Vector3f& positionOffset /*= Vector3f::Zero()*/,
    const Quaternionf& rotationOffset /*= Quaternionf::Identity()*/,
    const FileSaveOptions& options) {
  if (impl_->characterData.find(character.name) != impl_->characterData.end()) {
    // Character already exist. Doesn't allow character with the same name to be saved.
    // #TODO: proper warning
    return;
  }
  auto& scene = getDefaultScene(impl_->document);

  // Add character root node
  const auto rootNodeIdx = appendNode(impl_->document, character.name);
  MT_CHECK(
      rootNodeIdx >= 0 && rootNodeIdx < impl_->document.nodes.size(),
      "{}: {}",
      rootNodeIdx,
      impl_->document.nodes.size());
  auto& node = impl_->document.nodes[rootNodeIdx];
  const auto translation = fromMomentumVec3f(positionOffset);
  node.translation = {translation[0], translation[1], translation[2]};
  node.rotation = fromMomentumQuaternionf(rotationOffset);
  scene.nodes.push_back(rootNodeIdx);
  auto& characterData = impl_->characterData[character.name];
  characterData.rootIndex = rootNodeIdx;

  // fill with data
  characterData.nodeMap =
      addSkeletonToModel(impl_->document, character, options.extensions, rootNodeIdx);
  if (options.collisions) {
    addCollisionsToModel(impl_->document, character, characterData.nodeMap, options.extensions);
  }
  if (options.locators) {
    addLocatorsToModel(impl_->document, character, characterData.nodeMap, options.extensions);
  }
  if (options.mesh) {
    characterData.meshIndex =
        addMeshToModel(impl_->document, character, characterData.nodeMap, rootNodeIdx);

    // Add blendshapes if both addMesh and addBlendShapes are true
    if (options.blendShapes && character.blendShape != nullptr) {
      addMorphTargetsToModel(
          impl_->document,
          *character.blendShape,
          character.blendShape->shapeSize(),
          characterData.meshIndex);
    }
  }
  if (options.extensions) {
    auto& def = addMomentumExtension(impl_->document.extensionsAndExtras);
    parameterTransformToJson(character, def["transform"]);
    parameterLimitsToJson(character, def["parameterLimits"]);
    parameterSetsToJson(character, def["parameterSet"]);
    poseConstraintsToJson(character, def["poseConstraints"]);
  }
}

void GltfBuilder::addMesh(const Mesh& mesh, const std::string& name, bool addColor) {
  auto& model = impl_->document;
  auto meshIdx = addMeshToModel(model, mesh, addColor);
  if (meshIdx >= model.meshes.size()) {
    MT_LOGW("Failed to add mesh");
    return;
  }

  const auto nodeIndex = model.nodes.size();
  model.nodes.emplace_back();
  auto& node = model.nodes.back();
  node.name = name;
  node.translation = {0.0f, 0.0f, 0.0f};
  node.mesh = meshIdx;

  auto& scene = getDefaultScene(model);
  scene.nodes.push_back(nodeIndex);
}

size_t GltfBuilder::getNumCharacters() {
  return impl_->characterData.size();
}

size_t GltfBuilder::getCharacterRootIndex(const std::string& name) {
  if (impl_->characterData.count(name) == 0) {
    return kInvalidIndex;
  }
  return impl_->characterData[name].rootIndex;
}

size_t GltfBuilder::getNumJoints(const std::string& name) {
  if (impl_->characterData.count(name) == 0) {
    return kInvalidIndex;
  }
  return impl_->characterData[name].nodeMap.size();
}

void GltfBuilder::setFps(float fps) {
  // set frame-rate
  auto& def = addMomentumExtension(impl_->document.extensionsAndExtras);
  constexpr auto kFpsStr = "fps";
  if (!def.contains(kFpsStr)) {
    def[kFpsStr] = fps;
  } else if (def[kFpsStr] != fps) {
    MT_LOGW(
        "Attempting to set FPS to {} but it already exists as {}. Ignoring the new FPS.",
        fps,
        def[kFpsStr].get<float>());
  }
}

void GltfBuilder::addMotion(
    const Character& character,
    const float fps /*= 120.0f*/,
    const MotionParameters& motion /*= {}*/,
    const IdentityParameters& offsets /*= {}*/,
    const bool addExtensions /*= true*/,
    const std::string& customName /*= "default"*/) {
  setFps(fps);
  if (impl_->characterData.find(character.name) == impl_->characterData.end()) {
    // #TODO: Warn about this addition
    addCharacter(character);
  }

  const auto& jointToNodeMap = impl_->characterData[character.name].nodeMap;
  const size_t meshIndex = impl_->characterData[character.name].meshIndex;
  // add motion
  addMotionToModel(
      impl_->document,
      character,
      fps,
      motion,
      offsets,
      jointToNodeMap,
      meshIndex,
      addExtensions,
      customName);

  auto animIter = std::find_if(
      impl_->document.animations.begin(),
      impl_->document.animations.end(),
      [&customName](const fx::gltf::Animation& anim) { return anim.name == customName; });
  if (animIter != impl_->document.animations.end()) {
    const auto animIdx = static_cast<size_t>(animIter - impl_->document.animations.begin());
    impl_->characterData[character.name].animationIndices.push_back(animIdx);
  }
}

size_t GltfBuilder::getNumMotions() {
  return impl_->document.animations.size();
}

float GltfBuilder::getFps() {
  auto& def = addMomentumExtension(impl_->document.extensionsAndExtras);
  constexpr auto kFpsStr = "fps";
  if (!def.contains(kFpsStr)) {
    return 0.0f;
  }
  return def[kFpsStr];
}

std::vector<size_t> GltfBuilder::getCharacterMotions(const std::string& characterName) {
  if (impl_->characterData.count(characterName) == 0) {
    return {};
  }
  return impl_->characterData[characterName].animationIndices;
}

void GltfBuilder::addMarkerSequence(
    const float fps,
    std::span<const std::vector<momentum::Marker>> markers,
    const MarkerMesh markerMesh,
    const std::string& animName) {
  setFps(fps);
  addActorAnimationToModel(impl_->document, fps, markers, markerMesh, animName);
}

void GltfBuilder::save(
    const filesystem::path& filename,
    const GltfFileFormat fileFormat /*= GltfFileFormat::Extension*/,
    bool embedResources /*= false*/) {
  GltfBuilder::save(impl_->document, filename, fileFormat, embedResources);
}

void GltfBuilder::forceEmbedResources(fx::gltf::Document& document) {
  for (auto& buffer : document.buffers) {
    buffer.SetEmbeddedResource();
  }
}

const fx::gltf::Document& GltfBuilder::getDocument() {
  return impl_->document;
}

void GltfBuilder::forceEmbedResources() {
  forceEmbedResources(impl_->document);
}

void GltfBuilder::save(
    fx::gltf::Document& document,
    const filesystem::path& filename,
    const GltfFileFormat fileFormat /*= GltfFileFormat::Extension*/,
    bool embedResources /*= false*/) {
  if (embedResources) {
    forceEmbedResources(document);
  }
  saveDocument(filename, document, fileFormat);
}

void GltfBuilder::addSkeletonStates(
    const Character& character,
    const float fps,
    std::span<const SkeletonState> skeletonStates,
    const std::string& customName) {
  setFps(fps);
  if (impl_->characterData.find(character.name) == impl_->characterData.end()) {
    // #TODO: Warn about this addition
    addCharacter(character);
  }

  const auto& jointToNodeMap = impl_->characterData[character.name].nodeMap;

  // add motion from skeleton states
  addSkeletonStatesToModel(
      impl_->document, character, fps, skeletonStates, jointToNodeMap, customName);

  auto animIter = std::find_if(
      impl_->document.animations.begin(),
      impl_->document.animations.end(),
      [&customName](const fx::gltf::Animation& anim) { return anim.name == customName; });
  if (animIter != impl_->document.animations.end()) {
    const auto animIdx = static_cast<size_t>(animIter - impl_->document.animations.begin());
    impl_->characterData[character.name].animationIndices.push_back(animIdx);
  }
}

} // namespace momentum
