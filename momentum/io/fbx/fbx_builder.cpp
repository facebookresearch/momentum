/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/fbx_builder.h"

#include "momentum/common/exception.h"
#include "momentum/common/log.h"

#ifdef MOMENTUM_WITH_FBX_SDK

#include "momentum/character/character.h"
#include "momentum/character/character_state.h"
#include "momentum/common/name_utils.h"
#include "momentum/io/fbx/fbx_io_internal.h"

#include <unordered_map>

namespace momentum {

using namespace fbx_internal; // NOLINT(google-build-using-namespace)

namespace {

void prefixSkeletonNodeNames(
    const std::string& prefix,
    const std::vector<::fbxsdk::FbxNode*>& nodes) {
  for (auto* node : nodes) {
    MT_THROW_IF(node == nullptr, "Cannot rename a null FBX node");
    const std::string prefixedName = prefix + ":" + node->GetName();
    node->SetName(prefixedName.c_str());
    if (auto* attribute = node->GetNodeAttribute()) {
      attribute->SetName(prefixedName.c_str());
    }
  }
}

} // namespace

struct FbxBuilder::Impl {
  struct CharacterData {
    SkeletonNodeResult skeletonResult;
    MeshBlendShapeResult meshResult;
    std::string name;
  };

  ::fbxsdk::FbxManager* manager = nullptr;
  ::fbxsdk::FbxScene* scene = nullptr;

  // --- Character name model (read before changing name handling) ---
  // Every added character/rigid body is stored in `characters` under a UNIQUE *exported* name (the
  // map key). The exported name is derived from a requested name -- character.name for
  // addCharacter, or the `name` argument (falling back to character.name) for addRigidBody -- and
  // is made unique by makeUniqueName(), which appends a numeric suffix ("name_1", ...) on
  // collision. Because of the addRigidBody `name` override AND suffixing, the exported name is
  // frequently NOT equal to character.name: character.name is the *source* name, not the identity.
  // addCharacter/ addRigidBody RETURN the exported name; callers pass it back as `characterName` to
  // addMotion* to target a specific character -- required for instancing (the same Character object
  // added more than once) or whenever the name was overridden/suffixed.
  //
  // `exportedNameBySourceName` maps source name (character.name) -> exported name, FIRST-WINS
  // (never overwritten). It lets addMotion* resolve an EMPTY characterName back to a character even
  // when the exported name differs from character.name (e.g. an addRigidBody `name` override).
  // First-wins (not latest-wins) makes an empty characterName deterministically resolve to the
  // *name-owner* -- the first character added under that source name -- never a later duplicate.
  std::unordered_map<std::string, CharacterData> characters;
  std::unordered_map<std::string, std::string> exportedNameBySourceName;

  Impl() {
    manager = ::fbxsdk::FbxManager::Create();
    auto* ios = ::fbxsdk::FbxIOSettings::Create(manager, IOSROOT);
    manager->SetIOSettings(ios);
    scene = ::fbxsdk::FbxScene::Create(manager, "momentum_scene");
  }

  ~Impl() {
    if (scene != nullptr) {
      scene->Destroy();
      scene = nullptr;
    }
    if (manager != nullptr) {
      manager->Destroy();
      manager = nullptr;
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  // Resolve which stored character an addMotion* call targets (see the "Character name model" note
  // above) and validate that its skeleton matches `character`. Returns the stored CharacterData.
  CharacterData& resolveCharacterForMotion(
      const Character& character,
      const std::string& characterName) {
    std::string resolvedCharacterName = characterName;
    if (resolvedCharacterName.empty()) {
      const auto sourceIt = exportedNameBySourceName.find(character.name);
      MT_THROW_IF(
          sourceIt == exportedNameBySourceName.end(),
          "Character '{}' has not been added to the builder",
          character.name);
      resolvedCharacterName = sourceIt->second;
    }

    const auto it = characters.find(resolvedCharacterName);
    MT_THROW_IF(
        it == characters.end(),
        "Character '{}' has not been added to the builder",
        resolvedCharacterName);

    MT_THROW_IF(
        it->second.skeletonResult.nodes.size() != character.skeleton.joints.size(),
        "Character '{}' ({} joints) does not match exported character '{}' ({} skeleton nodes). When multiple characters share a name, pass the exact name returned by addCharacter/addRigidBody.",
        character.name,
        character.skeleton.joints.size(),
        resolvedCharacterName,
        it->second.skeletonResult.nodes.size());

    return it->second;
  }
};

FbxBuilder::FbxBuilder() : impl_(std::make_unique<Impl>()) {}

FbxBuilder::~FbxBuilder() = default;

FbxBuilder::FbxBuilder(FbxBuilder&&) noexcept = default;

FbxBuilder& FbxBuilder::operator=(FbxBuilder&&) noexcept = default;

std::string FbxBuilder::addCharacter(const Character& character, const FileSaveOptions& options) {
  MT_THROW_IF(!impl_, "FbxBuilder has been moved from or already saved");
  MT_THROW_IF(!impl_->scene, "FBX scene is null");

  std::string characterName = makeUniqueName(character.name, impl_->characters);
  if (characterName != character.name) {
    MT_LOGW(
        "Character name '{}' is already in use; exporting as '{}'.", character.name, characterName);
  }

  auto* root = impl_->scene->GetRootNode();
  MT_THROW_IF(root == nullptr, "Unable to get root node from FBX scene");

  // Set coordinate system
  ::fbxsdk::FbxAxisSystem axis = toFbx(options.coordSystemInfo);
  axis.ConvertScene(impl_->scene);

  // Create skeleton hierarchy
  auto skeletonResult = createSkeletonNodes(character, impl_->scene);
  if (characterName != character.name) {
    prefixSkeletonNodeNames(characterName, skeletonResult.nodes);
  }
  addMetaData(skeletonResult.rootNode, character, characterName);
  addPhysicalProperties(character, skeletonResult.jointToNodeMap);

  if (options.locators) {
    createLocatorNodes(character, impl_->scene, skeletonResult.nodes);
  }
  if (options.collisions) {
    createCollisionGeometryNodes(character, impl_->scene, skeletonResult.nodes);
  }

  // Create mesh with blend shapes (parented to scene root = skinned)
  MeshBlendShapeResult meshResult;
  if (options.mesh && character.mesh != nullptr) {
    const std::string meshName =
        characterName == character.name ? "body_mesh" : characterName + "_mesh";
    meshResult = createMeshNode(
        character, impl_->scene, root, skeletonResult.jointToNodeMap, options.permissive, meshName);
  }

  // Add skeleton to scene root
  if (!skeletonResult.nodes.empty()) {
    root->AddChild(skeletonResult.rootNode);
  }

  // Store character data for later animation
  impl_->characters[characterName] = {
      .skeletonResult = std::move(skeletonResult),
      .meshResult = std::move(meshResult),
      .name = characterName};
  // First-wins: record source name -> exported name so an empty characterName in addMotion*
  // resolves to the first character added under this source name (see FbxBuilder::Impl).
  impl_->exportedNameBySourceName.emplace(character.name, characterName);
  return characterName;
}

std::string FbxBuilder::addRigidBody(
    const Character& character,
    const std::string& name,
    size_t parentJoint,
    const FileSaveOptions& options) {
  MT_THROW_IF(!impl_, "FbxBuilder has been moved from or already saved");
  MT_THROW_IF(!impl_->scene, "FBX scene is null");

  auto* root = impl_->scene->GetRootNode();
  MT_THROW_IF(root == nullptr, "Unable to get root node from FBX scene");

  const std::string requestedName = name.empty() ? character.name : name;
  std::string characterName = makeUniqueName(requestedName, impl_->characters);
  if (characterName != requestedName) {
    MT_LOGW(
        "Character name '{}' is already in use; exporting as '{}'.", requestedName, characterName);
  }

  // Create skeleton hierarchy
  auto skeletonResult = createSkeletonNodes(character, impl_->scene);
  addMetaData(skeletonResult.rootNode, character, characterName);
  addPhysicalProperties(character, skeletonResult.jointToNodeMap);

  // Prefix skeleton joint names with characterName to avoid collisions when
  // multiple rigid bodies share the scene (e.g. "root" -> "controller_l:root").
  if (!name.empty() || characterName != character.name) {
    prefixSkeletonNodeNames(characterName, skeletonResult.nodes);
  }

  // Resolve which joint node to parent the mesh under
  MT_THROW_IF(
      parentJoint >= skeletonResult.nodes.size(),
      "parentJoint index {} out of range for skeleton with {} joints",
      parentJoint,
      skeletonResult.nodes.size());
  ::fbxsdk::FbxNode* meshParentNode = skeletonResult.nodes[parentJoint];

  // Create mesh parented under the target joint (no skin deformer = rigid body)
  MeshBlendShapeResult meshResult;
  if (options.mesh && character.mesh != nullptr) {
    // Create mesh node with vertices, normals, and UVs but no skin deformer.
    // Parent it under the target joint so it moves rigidly with that joint.
    const auto numVertices = static_cast<int>(character.mesh->vertices.size());
    ::fbxsdk::FbxNode* meshNode =
        ::fbxsdk::FbxNode::Create(impl_->scene, (characterName + "_mesh").c_str());
    ::fbxsdk::FbxMesh* lMesh = ::fbxsdk::FbxMesh::Create(impl_->scene, "mesh");
    lMesh->SetControlPointCount(numVertices);
    lMesh->InitNormals(numVertices);
    for (int i = 0; i < numVertices; i++) {
      FbxVector4 point(
          character.mesh->vertices[i].x(),
          character.mesh->vertices[i].y(),
          character.mesh->vertices[i].z());
      FbxVector4 normal(
          character.mesh->normals[i].x(),
          character.mesh->normals[i].y(),
          character.mesh->normals[i].z());
      lMesh->SetControlPointAt(point, normal, i);
    }
    writePolygonsToFbxMesh(*character.mesh, lMesh);
    lMesh->BuildMeshEdgeArray();
    meshNode->SetNodeAttribute(lMesh);

    // Add texture coordinates
    if (!character.mesh->texcoords.empty()) {
      const fbxsdk::FbxLayerElement::EType uvType = fbxsdk::FbxLayerElement::eTextureDiffuse;
      lMesh->InitTextureUV(0, uvType);
      lMesh->InitTextureUVIndices(
          ::fbxsdk::FbxLayerElement::EMappingMode::eByPolygonVertex, uvType);
      for (const auto& texcoords : character.mesh->texcoords) {
        lMesh->AddTextureUV(::fbxsdk::FbxVector2(texcoords[0], 1.0f - texcoords[1]), uvType);
      }
      writeTextureUVIndicesToFbxMesh(*character.mesh, lMesh, uvType);
    }

    // No skin deformer — parent mesh under target joint for rigid body behavior
    if (meshParentNode != nullptr) {
      meshParentNode->AddChild(meshNode);
    }
  }

  // Add skeleton to scene root
  if (!skeletonResult.nodes.empty()) {
    root->AddChild(skeletonResult.rootNode);
  }

  impl_->characters[characterName] = {
      .skeletonResult = std::move(skeletonResult),
      .meshResult = std::move(meshResult),
      .name = characterName};
  // First-wins: record source name -> exported name so an empty characterName in addMotion*
  // resolves to the first character added under this source name (see FbxBuilder::Impl).
  impl_->exportedNameBySourceName.emplace(character.name, characterName);
  return characterName;
}

void FbxBuilder::addMotion(
    const Character& character,
    float fps,
    const MatrixXf& motion,
    const VectorXf& offsets,
    const std::string& characterName) {
  MT_THROW_IF(!impl_, "FbxBuilder has been moved from or already saved");

  const auto& charData = impl_->resolveCharacterForMotion(character, characterName);

  if (motion.cols() == 0) {
    return;
  }

  // Convert model params to joint params via CharacterState
  CharacterParameters params;
  if (offsets.size() == character.parameterTransform.numJointParameters()) {
    params.offsets = offsets;
  } else {
    params.offsets = character.parameterTransform.bindPose();
  }

  CharacterState state;
  MatrixXf jointValues;
  params.pose = motion.col(0);
  state.set(params, character, false, false, false);

  jointValues.resize(state.skeletonState.jointParameters.v.size(), motion.cols());
  jointValues.col(0) = state.skeletonState.jointParameters.v;

  for (Eigen::Index f = 1; f < motion.cols(); f++) {
    params.pose = motion.col(f);
    state.set(params, character, false, false, false);
    jointValues.col(f) = state.skeletonState.jointParameters.v;
  }

  if (jointValues.rows() == character.parameterTransform.numJointParameters()) {
    createAnimationCurves(
        character, impl_->scene, charData.skeletonResult.nodes, jointValues, fps, false);
  }

  // Blend shape animation
  const MatrixXf blendShapeWeights = extractBlendShapeWeights(character, motion);
  if (blendShapeWeights.cols() > 0 && !charData.meshResult.blendShapeChannels.empty()) {
    createBlendShapeAnimationCurves(
        impl_->scene, charData.meshResult.blendShapeChannels, blendShapeWeights, fps);
  }
  const MatrixXf faceExpressionWeights = extractFaceExpressionWeights(character, motion);
  if (faceExpressionWeights.cols() > 0 && !charData.meshResult.faceExprChannels.empty()) {
    createBlendShapeAnimationCurves(
        impl_->scene, charData.meshResult.faceExprChannels, faceExpressionWeights, fps);
  }
}

void FbxBuilder::addMotionWithJointParams(
    const Character& character,
    float fps,
    const MatrixXf& jointParams,
    const std::string& characterName) {
  MT_THROW_IF(!impl_, "FbxBuilder has been moved from or already saved");

  const auto& charData = impl_->resolveCharacterForMotion(character, characterName);

  if (jointParams.cols() == 0) {
    return;
  }

  createAnimationCurves(
      character,
      impl_->scene,
      charData.skeletonResult.nodes,
      jointParams,
      fps,
      true); // skipActiveJointParamCheck = true
}

void FbxBuilder::addAnimatedMesh(
    const Character& character,
    const std::string& name,
    float fps,
    const MatrixXf& jointParams) {
  MT_THROW_IF(character.mesh == nullptr, "Character has no mesh");
  addAnimatedMesh(
      *character.mesh,
      name,
      fps,
      jointParams,
      character.skeleton.joints.empty() ? Vector3f::Zero()
                                        : character.skeleton.joints[0].translationOffset);
}

void FbxBuilder::addAnimatedMesh(
    const Mesh& mesh,
    const std::string& name,
    float fps,
    const MatrixXf& jointParams,
    const Vector3f& translationOffset) {
  MT_THROW_IF(!impl_, "FbxBuilder has been moved from or already saved");
  MT_THROW_IF(!impl_->scene, "FBX scene is null");
  MT_THROW_IF(jointParams.cols() == 0, "jointParams is empty");

  auto* root = impl_->scene->GetRootNode();
  MT_THROW_IF(root == nullptr, "Unable to get root node from FBX scene");

  // Create mesh node directly under scene root (no skeleton)
  const auto numVertices = static_cast<int>(mesh.vertices.size());
  ::fbxsdk::FbxNode* meshNode = ::fbxsdk::FbxNode::Create(impl_->scene, name.c_str());
  ::fbxsdk::FbxMesh* lMesh = ::fbxsdk::FbxMesh::Create(impl_->scene, (name + "_geo").c_str());
  lMesh->SetControlPointCount(numVertices);
  lMesh->InitNormals(numVertices);
  for (int i = 0; i < numVertices; i++) {
    lMesh->SetControlPointAt(
        FbxVector4(mesh.vertices[i].x(), mesh.vertices[i].y(), mesh.vertices[i].z()),
        FbxVector4(mesh.normals[i].x(), mesh.normals[i].y(), mesh.normals[i].z()),
        i);
  }
  writePolygonsToFbxMesh(mesh, lMesh);
  lMesh->BuildMeshEdgeArray();
  meshNode->SetNodeAttribute(lMesh);

  if (!mesh.texcoords.empty()) {
    const fbxsdk::FbxLayerElement::EType uvType = fbxsdk::FbxLayerElement::eTextureDiffuse;
    lMesh->InitTextureUV(0, uvType);
    lMesh->InitTextureUVIndices(::fbxsdk::FbxLayerElement::EMappingMode::eByPolygonVertex, uvType);
    for (const auto& texcoords : mesh.texcoords) {
      lMesh->AddTextureUV(::fbxsdk::FbxVector2(texcoords[0], 1.0f - texcoords[1]), uvType);
    }
    writeTextureUVIndicesToFbxMesh(mesh, lMesh, uvType);
  }

  root->AddChild(meshNode);

  // Animate the mesh node's transform using the root joint parameters.
  setFrameRate(impl_->scene, fps);
  auto [animStack, animBaseLayer] =
      getOrCreateAnimStackAndLayer(impl_->scene, "Skeleton Animation Stack");

  // jointParams is (nJointParams x nFrames) in C++ convention.
  // FBX uses 9 channels; momentum uses 7 (uniform scale maps to all three).
  Joint rootJoint;
  rootJoint.translationOffset = translationOffset;

  const auto nFrames = jointParams.cols();

  // Set static default transform (used for constant channels)
  FbxDouble3 staticTrans(
      jointParamToFbx(jointParams(0, 0), 0, rootJoint),
      jointParamToFbx(jointParams(1, 0), 1, rootJoint),
      jointParamToFbx(jointParams(2, 0), 2, rootJoint));
  meshNode->LclTranslation.Set(staticTrans);
  FbxDouble3 staticRot(
      jointParamToFbx(jointParams(3, 0), 3, rootJoint),
      jointParamToFbx(jointParams(4, 0), 4, rootJoint),
      jointParamToFbx(jointParams(5, 0), 5, rootJoint));
  meshNode->LclRotation.Set(staticRot);
  const Eigen::Index scaleRow = std::min(static_cast<Eigen::Index>(6), jointParams.rows() - 1);
  const float staticScale = jointParamToFbx(jointParams(scaleRow, 0), 6, rootJoint);
  meshNode->LclScaling.Set(FbxDouble3(staticScale, staticScale, staticScale));

  // Only create curve nodes and curves for non-constant channels to avoid
  // empty curves overriding the static property values in DCC tools.
  const std::array<::fbxsdk::FbxProperty*, 3> properties = {
      &meshNode->LclTranslation,
      &meshNode->LclRotation,
      &meshNode->LclScaling,
  };
  const std::array<const char*, 3> components = {
      FBXSDK_CURVENODE_COMPONENT_X,
      FBXSDK_CURVENODE_COMPONENT_Y,
      FBXSDK_CURVENODE_COMPONENT_Z,
  };

  ::fbxsdk::FbxTime time;
  for (size_t c = 0; c < 9; c++) {
    const auto paramRow = static_cast<Eigen::Index>(std::min(c, size_t(6)));
    if (paramRow >= jointParams.rows()) {
      break;
    }

    bool isConstant = true;
    const float firstVal = jointParams(paramRow, 0);
    for (Eigen::Index f = 1; f < nFrames; f++) {
      if (jointParams(paramRow, f) != firstVal) {
        isConstant = false;
        break;
      }
    }
    if (isConstant) {
      continue;
    }

    auto* prop = properties[c / 3];
    prop->GetCurveNode(true);
    auto* curve = prop->GetCurve(animBaseLayer, components[c % 3], true);

    curve->KeyModifyBegin();
    for (Eigen::Index f = 0; f < nFrames; f++) {
      const float val = jointParamToFbx(jointParams(paramRow, f), c, rootJoint);
      time.SetSecondDouble(static_cast<double>(f) / fps);
      const auto keyIndex = curve->KeyAdd(time);
      curve->KeySet(keyIndex, time, val);
    }
    curve->KeyModifyEnd();
  }
}

void FbxBuilder::addMarkerSequence(float fps, std::span<const std::vector<Marker>> markerSequence) {
  MT_THROW_IF(!impl_, "FbxBuilder has been moved from or already saved");

  if (!markerSequence.empty()) {
    createMarkerNodes(impl_->scene, markerSequence, fps);
  }
}

void FbxBuilder::save(const filesystem::path& filename) {
  MT_THROW_IF(!impl_, "FbxBuilder has been moved from or already saved");

  auto* lExporter = ::fbxsdk::FbxExporter::Create(impl_->manager, "");

  std::string sFilename = filename.string();
  bool lExportStatus =
      lExporter->Initialize(sFilename.c_str(), -1, impl_->manager->GetIOSettings());

  MT_THROW_IF(
      !lExportStatus,
      "Unable to initialize fbx exporter: {}",
      lExporter->GetStatus().GetErrorString());

  lExporter->Export(impl_->scene);
  lExporter->Destroy();

  // Clean up — the builder is consumed after save
  impl_.reset();
}

} // namespace momentum

#else // !MOMENTUM_WITH_FBX_SDK

namespace momentum {

struct FbxBuilder::Impl {};

FbxBuilder::FbxBuilder() {
  MT_THROW(
      "FbxBuilder requires the Autodesk FBX SDK. FBX loading is available via OpenFBX, but building scenes requires the full SDK.");
}

FbxBuilder::~FbxBuilder() = default;

FbxBuilder::FbxBuilder(FbxBuilder&&) noexcept = default;

FbxBuilder& FbxBuilder::operator=(FbxBuilder&&) noexcept = default;

std::string FbxBuilder::addCharacter(const Character&, const FileSaveOptions&) {
  MT_THROW("FbxBuilder requires the Autodesk FBX SDK.");
}

std::string
FbxBuilder::addRigidBody(const Character&, const std::string&, size_t, const FileSaveOptions&) {
  MT_THROW("FbxBuilder requires the Autodesk FBX SDK.");
}

void FbxBuilder::addMotion(
    const Character&,
    float,
    const MatrixXf&,
    const VectorXf&,
    const std::string&) {
  MT_THROW("FbxBuilder requires the Autodesk FBX SDK.");
}

void FbxBuilder::addMotionWithJointParams(
    const Character&,
    float,
    const MatrixXf&,
    const std::string&) {
  MT_THROW("FbxBuilder requires the Autodesk FBX SDK.");
}

void FbxBuilder::addAnimatedMesh(const Character&, const std::string&, float, const MatrixXf&) {
  MT_THROW("FbxBuilder requires the Autodesk FBX SDK.");
}

void FbxBuilder::addAnimatedMesh(
    const Mesh&,
    const std::string&,
    float,
    const MatrixXf&,
    const Vector3f&) {
  MT_THROW("FbxBuilder requires the Autodesk FBX SDK.");
}

void FbxBuilder::addMarkerSequence(float, std::span<const std::vector<Marker>>) {
  MT_THROW("FbxBuilder requires the Autodesk FBX SDK.");
}

void FbxBuilder::save(const filesystem::path&) {
  MT_THROW("FbxBuilder requires the Autodesk FBX SDK.");
}

} // namespace momentum

#endif // MOMENTUM_WITH_FBX_SDK
