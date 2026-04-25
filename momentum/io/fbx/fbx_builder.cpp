/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/fbx_builder.h"

#include "momentum/common/exception.h"

#ifdef MOMENTUM_WITH_FBX_SDK

#include "momentum/character/character.h"
#include "momentum/character/character_state.h"
#include "momentum/io/fbx/fbx_io_internal.h"

#include <unordered_map>

namespace momentum {

using namespace fbx_internal; // NOLINT(google-build-using-namespace)

struct FbxBuilder::Impl {
  struct CharacterData {
    SkeletonNodeResult skeletonResult;
    MeshBlendShapeResult meshResult;
    std::string name;
  };

  ::fbxsdk::FbxManager* manager = nullptr;
  ::fbxsdk::FbxScene* scene = nullptr;
  std::unordered_map<std::string, CharacterData> characters;

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
};

FbxBuilder::FbxBuilder() : impl_(std::make_unique<Impl>()) {}

FbxBuilder::~FbxBuilder() = default;

FbxBuilder::FbxBuilder(FbxBuilder&&) noexcept = default;

FbxBuilder& FbxBuilder::operator=(FbxBuilder&&) noexcept = default;

void FbxBuilder::addCharacter(const Character& character, const FileSaveOptions& options) {
  MT_THROW_IF(!impl_, "FbxBuilder has been moved from or already saved");
  MT_THROW_IF(!impl_->scene, "FBX scene is null");

  auto* root = impl_->scene->GetRootNode();
  MT_THROW_IF(root == nullptr, "Unable to get root node from FBX scene");

  // Set coordinate system
  ::fbxsdk::FbxAxisSystem axis = toFbx(options.coordSystemInfo);
  axis.ConvertScene(impl_->scene);

  // Create skeleton hierarchy
  auto skeletonResult = createSkeletonNodes(character, impl_->scene);
  addMetaData(skeletonResult.rootNode, character);

  if (options.locators) {
    createLocatorNodes(character, impl_->scene, skeletonResult.nodes);
  }
  if (options.collisions) {
    createCollisionGeometryNodes(character, impl_->scene, skeletonResult.nodes);
  }

  // Create mesh with blend shapes (parented to scene root = skinned)
  MeshBlendShapeResult meshResult;
  if (options.mesh && character.mesh != nullptr) {
    meshResult = createMeshNode(
        character, impl_->scene, root, skeletonResult.jointToNodeMap, options.permissive);
  }

  // Add skeleton to scene root
  if (!skeletonResult.nodes.empty()) {
    root->AddChild(skeletonResult.rootNode);
  }

  // Store character data for later animation
  const std::string& name = character.name;
  impl_->characters[name] = {std::move(skeletonResult), std::move(meshResult), name};
}

void FbxBuilder::addRigidBody(
    const Character& character,
    const std::string& name,
    size_t parentJoint,
    const FileSaveOptions& options) {
  MT_THROW_IF(!impl_, "FbxBuilder has been moved from or already saved");
  MT_THROW_IF(!impl_->scene, "FBX scene is null");

  auto* root = impl_->scene->GetRootNode();
  MT_THROW_IF(root == nullptr, "Unable to get root node from FBX scene");

  const std::string charName = name.empty() ? character.name : name;

  // Create skeleton hierarchy
  auto skeletonResult = createSkeletonNodes(character, impl_->scene);
  addMetaData(skeletonResult.rootNode, character);

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
        ::fbxsdk::FbxNode::Create(impl_->scene, (charName + "_mesh").c_str());
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

  impl_->characters[character.name] = {std::move(skeletonResult), std::move(meshResult), charName};
}

void FbxBuilder::addMotion(
    const Character& character,
    float fps,
    const MatrixXf& motion,
    const VectorXf& offsets) {
  MT_THROW_IF(!impl_, "FbxBuilder has been moved from or already saved");

  auto it = impl_->characters.find(character.name);
  MT_THROW_IF(
      it == impl_->characters.end(),
      "Character '{}' has not been added to the builder",
      character.name);

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

  const auto& charData = it->second;
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
    const MatrixXf& jointParams) {
  MT_THROW_IF(!impl_, "FbxBuilder has been moved from or already saved");

  auto it = impl_->characters.find(character.name);
  MT_THROW_IF(
      it == impl_->characters.end(),
      "Character '{}' has not been added to the builder",
      character.name);

  if (jointParams.cols() == 0) {
    return;
  }

  const auto& charData = it->second;
  createAnimationCurves(
      character,
      impl_->scene,
      charData.skeletonResult.nodes,
      jointParams,
      fps,
      true); // skipActiveJointParamCheck = true
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

void FbxBuilder::addCharacter(const Character&, const FileSaveOptions&) {
  MT_THROW("FbxBuilder requires the Autodesk FBX SDK.");
}

void FbxBuilder::addRigidBody(
    const Character&,
    const std::string&,
    size_t,
    const FileSaveOptions&) {
  MT_THROW("FbxBuilder requires the Autodesk FBX SDK.");
}

void FbxBuilder::addMotion(const Character&, float, const MatrixXf&, const VectorXf&) {
  MT_THROW("FbxBuilder requires the Autodesk FBX SDK.");
}

void FbxBuilder::addMotionWithJointParams(const Character&, float, const MatrixXf&) {
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
