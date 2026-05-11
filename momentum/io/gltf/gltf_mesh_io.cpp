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

#include <algorithm>

namespace momentum {

namespace {

struct TextureBuffers {
  std::vector<Vector2f> coords;
  std::vector<Vector3i> faces;
};

struct MeshBuffers {
  std::vector<Vector3i> faces;
  std::vector<Vector3f> positions;
  std::vector<Vector3f> normals;
  std::vector<Vector3b> colors;
  TextureBuffers texture;
};

nlohmann::json getMomentumExtension(const nlohmann::json& extensionsAndExtras) {
  if (extensionsAndExtras.count("extensions") != 0 &&
      extensionsAndExtras["extensions"].count("FB_momentum") != 0) {
    return extensionsAndExtras["extensions"]["FB_momentum"];
  } else {
    return nlohmann::json::object();
  }
}

std::vector<uint32_t> loadIndexBuffer(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive) {
  auto indices = copyAccessorBuffer<uint32_t>(model, primitive.indices);
  if (!indices.empty()) {
    return indices;
  }

  const auto shortIndices = copyAccessorBuffer<uint16_t>(model, primitive.indices);
  indices.reserve(shortIndices.size());
  for (const auto index : shortIndices) {
    indices.push_back(index);
  }
  return indices;
}

std::vector<Vector3i> makeTriangleFaces(const std::vector<uint32_t>& indices) {
  MT_CHECK(indices.size() % 3 == 0, "{} % 3 = {}", indices.size(), indices.size() % 3);

  std::vector<Vector3i> faces(indices.size() / 3);
  if (!faces.empty()) {
    std::copy_n(indices.data(), indices.size(), &faces[0][0]);
  }
  return faces;
}

std::vector<Vector3i> loadTriangleFaces(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive) {
  return makeTriangleFaces(loadIndexBuffer(model, primitive));
}

template <typename T>
std::vector<T> loadOptionalAttribute(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive,
    const char* attributeName) {
  const auto attribute = primitive.attributes.find(attributeName);
  if (attribute == primitive.attributes.end()) {
    return {};
  }
  return copyAccessorBuffer<T>(model, attribute->second);
}

std::vector<Vector3f> loadPositions(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive) {
  auto positions = copyAccessorBuffer<Vector3f>(model, primitive.attributes.at("POSITION"));
  toMomentumVec3f(positions);
  return positions;
}

void offsetFaces(std::vector<Vector3i>& faces, const size_t vertexOffset) {
  if (vertexOffset == 0) {
    return;
  }

  for (auto& face : faces) {
    face[0] += vertexOffset;
    face[1] += vertexOffset;
    face[2] += vertexOffset;
  }
}

TextureBuffers loadMomentumTextureBuffers(
    const fx::gltf::Document& model,
    const nlohmann::json& extension) {
  TextureBuffers texture;
  texture.coords = copyAccessorBuffer<Vector2f>(model, extension.at("texcoords"));
  texture.faces = makeTriangleFaces(copyAccessorBuffer<uint32_t>(model, extension.at("texfaces")));
  return texture;
}

TextureBuffers loadStandardTextureBuffers(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive,
    const size_t numPositions,
    const size_t vertexOffset,
    std::vector<Vector3i>& faces) {
  TextureBuffers texture;
  texture.coords = loadOptionalAttribute<Vector2f>(model, primitive, "TEXCOORD_0");

  offsetFaces(faces, vertexOffset);
  if (texture.coords.size() == numPositions) {
    texture.faces = faces;
  }

  MT_CHECK(
      texture.coords.empty() || texture.coords.size() == numPositions,
      "texcoord: {}, pos: {}",
      texture.coords.size(),
      numPositions);

  return texture;
}

TextureBuffers loadTextureBuffers(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive,
    const size_t numPositions,
    const size_t vertexOffset,
    std::vector<Vector3i>& faces) {
  const auto extension = getMomentumExtension(primitive);
  if (extension.count("texcoords") > 0 && extension.count("texfaces") > 0) {
    return loadMomentumTextureBuffers(model, extension);
  }

  return loadStandardTextureBuffers(model, primitive, numPositions, vertexOffset, faces);
}

MeshBuffers loadMeshBuffers(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive,
    const size_t vertexOffset) {
  MeshBuffers buffers;
  buffers.faces = loadTriangleFaces(model, primitive);
  buffers.positions = loadPositions(model, primitive);

  if (buffers.faces.empty() || buffers.positions.empty()) {
    return buffers;
  }

  buffers.normals = loadOptionalAttribute<Vector3f>(model, primitive, "NORMAL");
  MT_CHECK(
      buffers.normals.empty() || buffers.normals.size() == buffers.positions.size(),
      "nml: {}, pos: {}",
      buffers.normals.size(),
      buffers.positions.size());
  MT_LOGT_IF(buffers.normals.empty(), "no vertex normal found");

  buffers.colors = loadOptionalAttribute<Vector3b>(model, primitive, "COLOR_0");
  MT_CHECK(
      buffers.colors.empty() || buffers.colors.size() == buffers.positions.size(),
      "col: {}, pos: {}",
      buffers.colors.size(),
      buffers.positions.size());
  MT_LOGT_IF(buffers.colors.empty(), "no vertex color found");

  // NOTE: gltf does not support multiple texcoords per vertex, so we only load the GLTF texcoords
  //       if we don't have our own custom extension
  buffers.texture =
      loadTextureBuffers(model, primitive, buffers.positions.size(), vertexOffset, buffers.faces);
  MT_LOGT_IF(buffers.texture.coords.empty(), "no texture coords found");

  return buffers;
}

void appendColors(Mesh& mesh, const std::vector<Vector3b>& colors, const size_t vertexOffset) {
  const bool hadColors = !mesh.colors.empty();
  if (!colors.empty()) {
    if (!hadColors && vertexOffset > 0) {
      mesh.colors.resize(vertexOffset, Vector3b::Constant(255));
    }
    mesh.colors.insert(mesh.colors.end(), colors.begin(), colors.end());
    return;
  }

  if (hadColors) {
    mesh.colors.resize(mesh.vertices.size(), Vector3b::Constant(255));
  }
}

void appendMeshBuffers(Mesh& mesh, const MeshBuffers& buffers, const size_t vertexOffset) {
  mesh.faces.insert(mesh.faces.end(), buffers.faces.begin(), buffers.faces.end());
  mesh.vertices.insert(mesh.vertices.end(), buffers.positions.begin(), buffers.positions.end());
  mesh.normals.insert(mesh.normals.end(), buffers.normals.begin(), buffers.normals.end());
  appendColors(mesh, buffers.colors, vertexOffset);
  mesh.texcoords.insert(
      mesh.texcoords.end(), buffers.texture.coords.begin(), buffers.texture.coords.end());
  mesh.texcoord_faces.insert(
      mesh.texcoord_faces.end(), buffers.texture.faces.begin(), buffers.texture.faces.end());

  // make sure we have enough normals
  mesh.normals.resize(mesh.vertices.size(), Vector3f::Zero());
  mesh.confidence.resize(mesh.vertices.size(), 1.0f);
}

} // namespace

size_t
addMesh(const fx::gltf::Document& model, const fx::gltf::Primitive& primitive, Mesh_u& mesh) {
  if (primitive.mode != fx::gltf::Primitive::Mode::Triangles) {
    MT_LOGW("Mesh not loaded because it's not a triangle mesh");
    return 0;
  }

  const size_t vertexOffset = mesh->vertices.size();
  const auto buffers = loadMeshBuffers(model, primitive, vertexOffset);

  // if we have no points or indices, skip loading
  if (buffers.faces.empty() || buffers.positions.empty()) {
    return 0;
  }

  appendMeshBuffers(*mesh, buffers, vertexOffset);
  return buffers.positions.size();
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
        MT_CHECK(nodeIdx < nodeToObjectMap.size(), "{}: {}", nodeIdx, nodeToObjectMap.size());
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
      // NOLINTBEGIN(facebook-hte-ParameterUncheckedArrayBounds)
      const auto& node = model.nodes[nodeId];
      const auto& mesh = model.meshes[node.mesh];
      const auto& skin = model.skins[node.skin];
      // NOLINTEND(facebook-hte-ParameterUncheckedArrayBounds)
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
      // NOLINTBEGIN(facebook-hte-ParameterUncheckedArrayBounds)
      const auto& node = model.nodes[nodeId];
      const auto& mesh = model.meshes[node.mesh];
      // NOLINTEND(facebook-hte-ParameterUncheckedArrayBounds)
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
