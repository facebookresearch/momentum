/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/gltf/gltf_mesh_io.h"

#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/io/common/json_utils.h"
#include "momentum/io/gltf/utils/accessor_utils.h"
#include "momentum/io/gltf/utils/coordinate_utils.h"

namespace momentum {

namespace {

nlohmann::json getMomentumExtension(const nlohmann::json& extensionsAndExtras) {
  if (extensionsAndExtras.count("extensions") != 0 &&
      extensionsAndExtras["extensions"].count("FB_momentum") != 0) {
    return extensionsAndExtras["extensions"]["FB_momentum"];
  } else {
    return nlohmann::json::object();
  }
}

} // namespace

size_t
addMesh(const fx::gltf::Document& model, const fx::gltf::Primitive& primitive, Mesh_u& mesh) {
  if (primitive.mode != fx::gltf::Primitive::Mode::Triangles) {
    MT_LOGW("Mesh not loaded because it's not a triangle mesh");
    return 0;
  }

  // load index buffer
  auto idxDense = copyAccessorBuffer<uint32_t>(model, primitive.indices);
  if (idxDense.empty()) {
    // Try fallback with short indices.
    auto a = copyAccessorBuffer<uint16_t>(model, primitive.indices);
    for (const auto& ae : a) {
      idxDense.push_back(ae);
    }
  }
  MT_CHECK(idxDense.size() % 3 == 0, "{} % 3 = {}", idxDense.size(), idxDense.size() % 3);
  std::vector<Vector3i> idx(idxDense.size() / 3);
  if (!idx.empty()) {
    std::copy_n(idxDense.data(), idxDense.size(), &idx[0][0]);
  }

  // load vertex position buffer
  auto pos = copyAccessorBuffer<Vector3f>(model, primitive.attributes.at("POSITION"));
  toMomentumVec3f(pos);

  // if we have no points or indices, skip loading
  if (idx.empty() || pos.empty()) {
    return 0;
  }

  // load optional normal buffer
  std::vector<Vector3f> nml;
  const auto normId = primitive.attributes.find("NORMAL");
  if (normId != primitive.attributes.end()) {
    nml = copyAccessorBuffer<Vector3f>(model, normId->second);
  }
  MT_CHECK(nml.empty() || nml.size() == pos.size(), "nml: {}, pos: {}", nml.size(), pos.size());
  MT_LOGT_IF(nml.empty(), "no vertex normal found");

  // load optional color buffer
  std::vector<Vector3b> col;
  const auto colorId = primitive.attributes.find("COLOR_0");
  if (colorId != primitive.attributes.end()) {
    col = copyAccessorBuffer<Vector3b>(model, colorId->second);
  }
  MT_CHECK(col.empty() || col.size() == pos.size(), "col: {}, pos: {}", col.size(), pos.size());
  MT_LOGT_IF(col.empty(), "no vertex color found");

  // NOTE: gltf does not support multiple texcoords per vertex, so we only load the GLTF texcoords
  //       if we don't have our own custom extension
  const auto& extension = getMomentumExtension(primitive);
  std::vector<Vector2f> texcoord;
  std::vector<Vector3i> texfaces;
  if (extension.count("texcoords") > 0 && extension.count("texfaces") > 0) {
    texcoord = copyAccessorBuffer<Vector2f>(model, extension.at("texcoords"));

    // load "face index buffer
    auto fidxDense = copyAccessorBuffer<uint32_t>(model, extension.at("texfaces"));
    MT_CHECK(fidxDense.size() % 3 == 0, "{} % 3 = {}", fidxDense.size(), fidxDense.size() % 3);
    texfaces.resize(fidxDense.size() / 3);
    if (!fidxDense.empty()) {
      MT_THROW_IF(texfaces.empty(), "texfaces is empty but fidxDense is not empty");
      std::copy_n(fidxDense.data(), fidxDense.size(), &texfaces[0][0]);
    }
  } else {
    // load optional standard GLTF texcoord buffer
    auto texcoordId = primitive.attributes.find("TEXCOORD_0");
    if (texcoordId != primitive.attributes.end()) {
      texcoord = copyAccessorBuffer<Vector2f>(model, texcoordId->second);
    }
    const auto kVertexOffset = mesh->vertices.size();
    // Update vertex indices of the faces!!!
    if (kVertexOffset > 0) {
      for (auto&& iFace : idx) {
        iFace[0] += kVertexOffset;
        iFace[1] += kVertexOffset;
        iFace[2] += kVertexOffset;
      }
    }
    if (texcoord.size() == pos.size()) {
      texfaces = idx;
    }
    MT_CHECK(
        texcoord.empty() || texcoord.size() == pos.size(),
        "texcoord: {}, pos: {}",
        texcoord.size(),
        pos.size());
  }
  MT_LOGT_IF(texcoord.empty(), "no texture coords found");

  // append new faces
  mesh->faces.insert(mesh->faces.end(), idx.begin(), idx.end());
  mesh->vertices.insert(mesh->vertices.end(), pos.begin(), pos.end());
  mesh->normals.insert(mesh->normals.end(), nml.begin(), nml.end());
  mesh->colors.insert(mesh->colors.end(), col.begin(), col.end());
  mesh->texcoords.insert(mesh->texcoords.end(), texcoord.begin(), texcoord.end());
  mesh->texcoord_faces.insert(mesh->texcoord_faces.end(), texfaces.begin(), texfaces.end());

  // make sure we have enough normals and colors
  mesh->normals.resize(mesh->vertices.size(), Vector3f::Zero());
  mesh->colors.resize(mesh->vertices.size(), Vector3b::Zero());
  mesh->confidence.resize(mesh->vertices.size(), 1.0f);
  return pos.size();
}

size_t addBlendShapes(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive,
    BlendShape_u& blendShape) {
  if (primitive.targets.empty()) {
    return 0;
  }

  const size_t numNewTargets = primitive.targets.size();

  // Read the base mesh vertices from the primitive
  auto baseVertices = copyAccessorBuffer<Vector3f>(model, primitive.attributes.at("POSITION"));
  toMomentumVec3f(baseVertices);
  const size_t kNumNewVertices = baseVertices.size();

  // Create the BlendShape if it doesn't exist yet
  if (blendShape == nullptr) {
    // Initialize with the base mesh vertices
    blendShape =
        std::make_unique<BlendShape>(std::span<const Vector3f>(baseVertices), numNewTargets);

    // load blendshape names from extension
    const auto& extension = getMomentumExtension(primitive.extensionsAndExtras);
    std::vector<std::string> blendShapeNames =
        extension.value("shapeNames", std::vector<std::string>());

    // Load each morph target
    for (size_t iTarget = 0; iTarget < numNewTargets; ++iTarget) {
      const auto& target = primitive.targets[iTarget];

      // Look for POSITION attribute in the morph target
      auto posId = target.find("POSITION");
      if (posId == target.end()) {
        MT_LOGW("Morph target {} has no POSITION attribute", iTarget);
        continue;
      }

      // Load the position deltas
      auto deltas = copyAccessorBuffer<Vector3f>(model, posId->second);
      toMomentumVec3f(deltas);

      MT_CHECK(
          deltas.size() == kNumNewVertices,
          "Morph target {} has {} vertices but mesh has {} vertices",
          iTarget,
          deltas.size(),
          kNumNewVertices);

      // Set the shape vector for this target
      if (iTarget < blendShapeNames.size()) {
        blendShape->setShapeVector(
            iTarget, std::span<const Vector3f>(deltas), blendShapeNames[iTarget]);
      } else {
        blendShape->setShapeVector(iTarget, std::span<const Vector3f>(deltas));
      }
    }
  } else {
    // Append to existing BlendShape
    const size_t kNumExistingVertices = blendShape->modelSize();
    const size_t kNumExistingTargets = blendShape->shapeSize();
    const size_t kTotalVertices = kNumExistingVertices + kNumNewVertices;
    const size_t kMaxTargets = std::max(kNumExistingTargets, numNewTargets);

    // Get the existing data
    auto existingBaseShape = blendShape->getBaseShape();
    const auto& existingShapeVectors = blendShape->getShapeVectors();

    // Create new base shape by appending new vertices to existing ones
    std::vector<Vector3f> newBaseShape;
    newBaseShape.reserve(kTotalVertices);
    newBaseShape.insert(newBaseShape.end(), existingBaseShape.begin(), existingBaseShape.end());
    newBaseShape.insert(newBaseShape.end(), baseVertices.begin(), baseVertices.end());

    // Create new shape vectors matrix
    MatrixXf newShapeVectors(kTotalVertices * 3, kMaxTargets);
    newShapeVectors.setZero();

    // Copy existing shape vectors to the top rows
    newShapeVectors.topRows(kNumExistingVertices * 3).leftCols(kNumExistingTargets) =
        existingShapeVectors;

    // Load and append new morph target deltas
    for (size_t iTarget = 0; iTarget < numNewTargets; ++iTarget) {
      const auto& target = primitive.targets[iTarget];

      // Look for POSITION attribute in the morph target
      auto posId = target.find("POSITION");
      if (posId == target.end()) {
        MT_LOGW("Morph target {} has no POSITION attribute", iTarget);
        continue;
      }

      // Load the position deltas
      auto deltas = copyAccessorBuffer<Vector3f>(model, posId->second);
      toMomentumVec3f(deltas);

      MT_CHECK(
          deltas.size() == kNumNewVertices,
          "Morph target {} has {} vertices but mesh has {} vertices",
          iTarget,
          deltas.size(),
          kNumNewVertices);

      // Copy deltas into the new shape vectors matrix
      for (size_t iVert = 0; iVert < kNumNewVertices; ++iVert) {
        const size_t rowOffset = (kNumExistingVertices + iVert) * 3;
        newShapeVectors(rowOffset + 0, iTarget) = deltas[iVert].x();
        newShapeVectors(rowOffset + 1, iTarget) = deltas[iVert].y();
        newShapeVectors(rowOffset + 2, iTarget) = deltas[iVert].z();
      }
    }

    // Create a new BlendShape with the combined data
    blendShape = std::make_unique<BlendShape>(std::span<const Vector3f>(newBaseShape), kMaxTargets);
    blendShape->setShapeVectors(newShapeVectors);
  }

  return numNewTargets;
}

std::vector<Vector4f> loadWeightsFromAccessor(
    const fx::gltf::Document& model,
    int32_t accessorIdx) {
  std::vector<Vector4f> weightsData = copyAlignedAccessorBuffer<Vector4f>(model, accessorIdx);
  if (weightsData.empty()) {
    // Try fallback with normalized unsigned short weights (per glTF 2.0 spec, WEIGHTS_n can be
    // FLOAT, UNSIGNED_BYTE normalized, or UNSIGNED_SHORT normalized).
    auto weightsShort = copyAccessorBuffer<Vector4s>(model, accessorIdx);
    if (!weightsShort.empty()) {
      weightsData.reserve(weightsShort.size());
      for (const auto& value : weightsShort) {
        weightsData.emplace_back(value.cast<float>() / 65535.0f);
      }
    }
  }
  if (weightsData.empty()) {
    // Try fallback with normalized unsigned byte weights.
    auto weightsByte = copyAccessorBuffer<Vector4b>(model, accessorIdx);
    if (!weightsByte.empty()) {
      weightsData.reserve(weightsByte.size());
      for (const auto& value : weightsByte) {
        weightsData.emplace_back(value.cast<float>() / 255.0f);
      }
    }
  }
  return weightsData;
}

void addSkinWeights(
    const fx::gltf::Document& model,
    const fx::gltf::Skin& skin,
    const fx::gltf::Primitive& primitive,
    const std::vector<size_t>& nodeToObjectMap,
    const size_t kNumVertices,
    SkinWeights_u& skinWeights) {
  const auto kVertexOffset = skinWeights->index.rows();
  skinWeights->index.conservativeResize(kVertexOffset + kNumVertices, Eigen::NoChange);
  skinWeights->weight.conservativeResize(kVertexOffset + kNumVertices, Eigen::NoChange);
  skinWeights->index.bottomRows(kNumVertices).setZero();
  skinWeights->weight.bottomRows(kNumVertices).setZero();

  // load skinning in batches of 4 indices/weights at a time (up to 8)
  for (size_t i = 0; i < 2; i++) {
    // load skinning index buffer
    auto jointAttribute = primitive.attributes.find(std::string("JOINTS_") + std::to_string(i));
    std::vector<Vector4s> jointIndices;
    if (jointAttribute != primitive.attributes.end()) {
      jointIndices = copyAccessorBuffer<Vector4s>(model, jointAttribute->second);
      if (jointIndices.empty()) {
        // Try fallback with short indices.
        auto jointIndicesShort = copyAccessorBuffer<Vector4b>(model, jointAttribute->second);
        for (const auto& value : jointIndicesShort) {
          jointIndices.emplace_back(value.cast<uint16_t>());
        }
      }
    }

    // load skinning weight buffer
    auto weightsAttribute = primitive.attributes.find(std::string("WEIGHTS_") + std::to_string(i));
    std::vector<Vector4f> weightsData;
    if (weightsAttribute != primitive.attributes.end()) {
      weightsData = loadWeightsFromAccessor(model, weightsAttribute->second);
      if (weightsData.empty()) {
        MT_LOGW("No skinning weights read");
        return;
      }
    } else {
      MT_LOGW_IF(i == 0, "No skinning weights stored on primitive");
      return;
    }

    if (jointIndices.empty() || weightsData.empty() || jointIndices.size() != weightsData.size() ||
        jointIndices.size() != kNumVertices) {
      MT_LOGW_IF(jointIndices.empty() || weightsData.empty(), "Mesh is not skinned to any joint");
      MT_LOGW_IF(
          !jointIndices.empty() && !weightsData.empty() &&
              jointIndices.size() != weightsData.size(),
          "Inconsistent data: {} vertices are skinned but {} weights found",
          jointIndices.size(),
          weightsData.size());
      MT_LOGW_IF(
          jointIndices.size() != kNumVertices,
          "Inconsistent data: {} vertices are skinned but mesh has {} vertices",
          jointIndices.size(),
          kNumVertices);
      return;
    }

    // copy indices/vertices into our buffer
    for (Eigen::Index v = 0; v < static_cast<int>(kNumVertices); v++) {
      for (size_t d = 0; d < 4; d++) {
        const auto& skinWeight = weightsData[v][d];
        const auto& jointIdxInSkin = jointIndices[v][d];
        MT_CHECK(
            jointIdxInSkin >= 0 && jointIdxInSkin < skin.joints.size(),
            "{}: {}",
            jointIdxInSkin,
            skin.joints.size());
        const auto& nodeIdx = skin.joints[jointIdxInSkin];
        const auto& jointIndex = nodeToObjectMap[nodeIdx];
        if (jointIndex != kInvalidIndex) {
          skinWeights->index(kVertexOffset + v, i * 4 + d) = (uint32_t)jointIndex;
          skinWeights->weight(kVertexOffset + v, i * 4 + d) = skinWeight;
        } else {
          MT_LOGW("Invalid joint index encountered when reading skinning weights");
        }
      }
    }
  }
}

std::tuple<Mesh_u, SkinWeights_u, BlendShape_u> loadSkinnedMesh(
    const fx::gltf::Document& model,
    const std::vector<size_t>& meshNodes,
    const std::vector<size_t>& nodeToObjectMap) {
  std::vector<size_t> skinnedNodes;
  skinnedNodes.reserve(meshNodes.size());
  std::vector<size_t> unskinnedNodes;
  unskinnedNodes.reserve(meshNodes.size());
  for (auto nodeId : meshNodes) {
    MT_CHECK(nodeId >= 0 && nodeId < model.nodes.size(), "{}: {}", nodeId, model.nodes.size());
    const auto& node = model.nodes[nodeId];
    if ((node.mesh >= 0) && (node.skin >= 0)) {
      MT_CHECK(node.mesh < model.meshes.size(), "{}: {}", node.mesh, model.meshes.size());
      MT_CHECK(node.skin < model.skins.size(), "{}: {}", node.skin, model.skins.size());
      skinnedNodes.push_back(nodeId);
    } else if (node.mesh >= 0) {
      MT_CHECK(node.mesh < model.meshes.size(), "{}: {}", node.mesh, model.meshes.size());
      unskinnedNodes.push_back(nodeId);
    }
  }

  MT_LOGW_IF(
      !skinnedNodes.empty() && !unskinnedNodes.empty(),
      "Found both skinned meshes {} and meshes without skinning {}. Unskinned meshes will be ignored.",
      skinnedNodes.size(),
      unskinnedNodes.size());
  auto resultMesh = std::make_unique<Mesh>();
  auto skinWeights = std::make_unique<SkinWeights>();
  BlendShape_u blendShape = nullptr;

  if (!skinnedNodes.empty()) {
    for (auto nodeId : skinnedNodes) {
      MT_CHECK(
          nodeId < model.nodes.size(), "nodeId: {}, nodes size: {}", nodeId, model.nodes.size());
      const auto& node = model.nodes[nodeId];
      MT_CHECK(
          node.mesh >= 0 && static_cast<size_t>(node.mesh) < model.meshes.size(),
          "mesh: {}, meshes size: {}",
          node.mesh,
          model.meshes.size());
      const auto& mesh = model.meshes[node.mesh];
      MT_CHECK(
          node.skin >= 0 && static_cast<size_t>(node.skin) < model.skins.size(),
          "skin: {}, skins size: {}",
          node.skin,
          model.skins.size());
      const auto& skin = model.skins[node.skin];
      for (const auto& primitive : mesh.primitives) {
        const auto kNumVertices = addMesh(model, primitive, resultMesh);
        addSkinWeights(model, skin, primitive, nodeToObjectMap, kNumVertices, skinWeights);
        addBlendShapes(model, primitive, blendShape);
        MT_CHECK(
            resultMesh->vertices.size() == skinWeights->index.rows(),
            "vertices: {}, skinWeights: {}",
            resultMesh->vertices.size(),
            skinWeights->index.rows());
      }
    }
  } else if (!unskinnedNodes.empty()) {
    for (auto nodeId : unskinnedNodes) {
      MT_CHECK(
          nodeId < model.nodes.size(), "nodeId: {}, nodes size: {}", nodeId, model.nodes.size());
      const auto& node = model.nodes[nodeId];
      MT_CHECK(
          node.mesh >= 0 && static_cast<size_t>(node.mesh) < model.meshes.size(),
          "mesh: {}, meshes size: {}",
          node.mesh,
          model.meshes.size());
      const auto& mesh = model.meshes[node.mesh];
      for (const auto& primitive : mesh.primitives) {
        addMesh(model, primitive, resultMesh);
        addBlendShapes(model, primitive, blendShape);
      }
    }
    return {std::move(resultMesh), nullptr, std::move(blendShape)};
  } else {
    return {};
  }
  return {std::move(resultMesh), std::move(skinWeights), std::move(blendShape)};
}

SkinWeights_u bindMeshToJoint(const Mesh_u& mesh, size_t jointId) {
  auto skinWeights = std::make_unique<SkinWeights>();
  const auto kNumVertices = mesh->vertices.size();
  skinWeights->index.conservativeResize(kNumVertices, Eigen::NoChange);
  skinWeights->weight.conservativeResize(kNumVertices, Eigen::NoChange);
  skinWeights->index.bottomRows(kNumVertices).setZero();
  skinWeights->weight.bottomRows(kNumVertices).setZero();
  for (auto vertId = 0; vertId < kNumVertices; vertId++) {
    skinWeights->index(vertId, 0) = static_cast<uint32_t>(jointId);
    skinWeights->weight(vertId, 0) = 1.0f;
  }

  return skinWeights;
}

} // namespace momentum
