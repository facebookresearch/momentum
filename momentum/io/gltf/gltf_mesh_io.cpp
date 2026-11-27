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
#include "momentum/io/gltf/utils/momentum_extension.h"

#include <algorithm>
#include <array>
#include <exception>
#include <numeric>
#include <span>
#include <string>
#include <tuple>
#include <utility>

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

std::vector<uint32_t> loadIndexBuffer(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive) {
  return copyAccessorBuffer<uint32_t>(model, primitive.indices);
}

std::vector<Vector3i> makeTriangleFaces(const std::vector<uint32_t>& indices) {
  MT_CHECK(indices.size() % 3 == 0, "{} % 3 = {}", indices.size(), indices.size() % 3);

  std::vector<Vector3i> faces(indices.size() / 3);
  if (!faces.empty()) {
    std::copy_n(indices.data(), indices.size(), &faces[0][0]);
  }
  return faces;
}

// The triangles of `primitive`, a triangle list, strip or fan, with or without an index buffer, as
// the glTF format defines them. The triangles of a strip or fan that repeat a vertex have no area
// and are skipped.
std::vector<Vector3i> loadTriangleFaces(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive) {
  std::vector<uint32_t> indices;
  if (primitive.indices >= 0) {
    indices = loadIndexBuffer(model, primitive);
  } else {
    // Without an index buffer, the vertices come in order.
    indices.resize(model.accessors.at(primitive.attributes.at("POSITION")).count);
    std::iota(indices.begin(), indices.end(), 0u);
  }
  if (primitive.mode == fx::gltf::Primitive::Mode::Triangles) {
    return makeTriangleFaces(indices);
  }
  const bool strip = primitive.mode == fx::gltf::Primitive::Mode::TriangleStrip;
  const auto vertex = [&indices](size_t i) { return static_cast<int>(indices.at(i)); };
  std::vector<Vector3i> faces;
  for (size_t i = 0; i + 2 < indices.size(); ++i) {
    Vector3i face;
    if (!strip) {
      face = Vector3i(vertex(i + 1), vertex(i + 2), vertex(0));
    } else if (i % 2 == 0) {
      face = Vector3i(vertex(i), vertex(i + 1), vertex(i + 2));
    } else {
      // A strip alternates the order of its triangles so that they all face the same side.
      face = Vector3i(vertex(i), vertex(i + 2), vertex(i + 1));
    }
    if (face[0] != face[1] && face[1] != face[2] && face[0] != face[2]) {
      faces.push_back(face);
    }
  }
  return faces;
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

// The colors of accessor `id`, which exists, without alpha: it has `Color4` elements when it has
// alpha, and `Color3` elements otherwise.
template <typename Color3, typename Color4>
std::vector<Color3> loadColorsWithoutAlpha(const fx::gltf::Document& model, int32_t id) {
  if (model.accessors.at(id).type != fx::gltf::Accessor::Type::Vec4) {
    return copyAccessorBuffer<Color3>(model, id);
  }
  const std::vector<Color4> colors = copyAccessorBuffer<Color4>(model, id);
  std::vector<Color3> result(colors.size());
  std::ranges::transform(
      colors, result.begin(), [](const Color4& color) { return color.template head<3>(); });
  return result;
}

// The vertex colors of `primitive`, from colors with or without alpha of any component type that
// the glTF format allows: normalized unsigned bytes and shorts, and floats. Unsigned bytes are
// read as they are stored.
std::vector<Vector3b> loadColors(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive) {
  const auto attribute = primitive.attributes.find("COLOR_0");
  if (attribute == primitive.attributes.end()) {
    return {};
  }
  const auto id = static_cast<int32_t>(attribute->second);
  MT_THROW_IF(
      attribute->second >= model.accessors.size(),
      "Accessor {} of the vertex colors does not exist; the file has {} accessors",
      attribute->second,
      model.accessors.size());
  const auto componentType = model.accessors.at(id).componentType;
  if (componentType == fx::gltf::Accessor::ComponentType::UnsignedByte) {
    return loadColorsWithoutAlpha<Vector3b, Vector4b>(model, id);
  }
  // Normalized unsigned shorts come in [0, 1], as floats do.
  const std::vector<Vector3f> colors = loadColorsWithoutAlpha<Vector3f, Vector4f>(model, id);
  std::vector<Vector3b> result(colors.size());
  std::ranges::transform(colors, result.begin(), [](const Vector3f& color) {
    return Vector3b(
        (255.0f * color.cwiseMax(0.0f).cwiseMin(1.0f)).array().round().cast<uint8_t>().matrix());
  });
  return result;
}

std::vector<Vector3f> loadPositions(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive) {
  auto positions = copyAccessorBuffer<Vector3f>(model, primitive.attributes.at("POSITION"));
  toMomentumVec3f(positions);
  return positions;
}

// The sum over the morph targets of `primitive` of the target's attribute `name` times its default
// weight in `weights`, which the glTF format adds to the primitive's attribute, of `count` values,
// in its rest shape. Empty when no target with a nonzero weight has `count` values of the
// attribute. A target with another count is skipped with a warning; for POSITION, the blend shape
// reader's own check of the counts then stops the load, as before.
std::vector<Vector3f> defaultMorphOffsets(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive,
    std::span<const float> weights,
    const std::string& name,
    size_t count) {
  std::vector<Vector3f> offsets;
  if (count == 0) {
    return offsets;
  }
  size_t nextTarget = 0;
  for (const float weight : weights.first(std::min(weights.size(), primitive.targets.size()))) {
    const size_t target = nextTarget++;
    const auto& attributes = primitive.targets.at(target);
    const auto attribute = attributes.find(name);
    if (weight == 0.0f || attribute == attributes.end()) {
      continue;
    }
    const auto deltas = copyAccessorBuffer<Vector3f>(model, attribute->second);
    if (deltas.size() != count) {
      MT_LOGW(
          "glTF morph target {} has {} {} values, but its primitive has {}; its default weight is "
          "not applied to them.",
          target,
          deltas.size(),
          name,
          count);
      continue;
    }
    offsets.resize(count, Vector3f::Zero());
    for (size_t i = 0; i < count; ++i) {
      offsets[i] += weight * deltas[i];
    }
  }
  return offsets;
}

// Adds to `positions`, the POSITION values of `primitive` in momentum units, the morph targets
// times their default weights `weights`: the positions of the primitive's rest shape.
void addDefaultMorphPositions(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive,
    std::span<const float> weights,
    std::vector<Vector3f>& positions) {
  auto offsets = defaultMorphOffsets(model, primitive, weights, "POSITION", positions.size());
  toMomentumVec3f(offsets);
  for (size_t i = 0; i < offsets.size(); ++i) {
    positions.at(i) += offsets[i];
  }
}

// The default weights of the morph targets of the mesh of `node`: the node's, which override the
// mesh's for that node, or else the mesh's. fx-gltf does not read the `weights` of a node from a
// file, so a node's weights come only from a document built in memory.
std::span<const float> morphWeightsOfNode(const fx::gltf::Node& node, const fx::gltf::Mesh& mesh) {
  return node.weights.empty() ? std::span<const float>(mesh.weights)
                              : std::span<const float>(node.weights);
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
  const auto& extension = findMomentumExtension(primitive.extensionsAndExtras);
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

  buffers.colors = loadColors(model, primitive);
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

size_t addMesh(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive,
    Mesh_u& mesh,
    std::span<const float> defaultMorphWeights) {
  if (primitive.mode != fx::gltf::Primitive::Mode::Triangles &&
      primitive.mode != fx::gltf::Primitive::Mode::TriangleStrip &&
      primitive.mode != fx::gltf::Primitive::Mode::TriangleFan) {
    MT_LOGW("Mesh not loaded because it has points or lines instead of triangles");
    return 0;
  }
  if (primitive.attributes.count("POSITION") == 0) {
    MT_LOGW("Mesh not loaded because it has no vertex positions");
    return 0;
  }

  const size_t vertexOffset = mesh->vertices.size();
  auto buffers = loadMeshBuffers(model, primitive, vertexOffset);

  // if we have no points or indices, skip loading
  if (buffers.faces.empty() || buffers.positions.empty()) {
    return 0;
  }

  // The rest shape of a primitive includes its morph targets times their default weights.
  addDefaultMorphPositions(model, primitive, defaultMorphWeights, buffers.positions);
  // The loader did not read the normals of morph targets before it applied their default weights,
  // so a file whose normal deltas cannot be read keeps loading, with the normals of its base shape.
  std::vector<Vector3f> normalOffsets;
  try {
    normalOffsets = defaultMorphOffsets(
        model, primitive, defaultMorphWeights, "NORMAL", buffers.normals.size());
  } catch (const std::exception& e) {
    MT_LOGW(
        "The default weights of glTF morph targets are not applied to the normals: {}", e.what());
  }
  for (size_t i = 0; i < normalOffsets.size(); ++i) {
    buffers.normals[i] = (buffers.normals[i] + normalOffsets[i]).normalized();
  }

  appendMeshBuffers(*mesh, buffers, vertexOffset);
  return buffers.positions.size();
}

size_t addBlendShapes(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive,
    BlendShape_u& blendShape,
    std::span<const float> defaultMorphWeights) {
  if (primitive.targets.empty() || primitive.attributes.count("POSITION") == 0) {
    return 0;
  }

  const size_t numNewTargets = primitive.targets.size();

  // The base shape is the rest shape that addMesh() gives.
  auto baseVertices = loadPositions(model, primitive);
  addDefaultMorphPositions(model, primitive, defaultMorphWeights, baseVertices);
  const size_t kNumNewVertices = baseVertices.size();

  // Create the BlendShape if it doesn't exist yet
  if (blendShape == nullptr) {
    // Initialize with the base mesh vertices
    blendShape =
        std::make_unique<BlendShape>(std::span<const Vector3f>(baseVertices), numNewTargets);

    // load blendshape names from extension
    const auto& extension = findMomentumExtension(primitive.extensionsAndExtras);
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
  // The glTF format allows float, normalized unsigned byte and normalized unsigned short weights.
  // Unsigned byte and short weights are read as normalized also when the file does not mark them
  // so, as momentum always read them.
  std::vector<Vector4f> weights = copyAlignedAccessorBuffer<Vector4f>(model, accessorIdx);
  if (weights.empty()) {
    return weights;
  }
  const auto& accessor = model.accessors.at(accessorIdx);
  if (accessor.normalized ||
      (accessor.componentType != fx::gltf::Accessor::ComponentType::UnsignedByte &&
       accessor.componentType != fx::gltf::Accessor::ComponentType::UnsignedShort)) {
    return weights;
  }
  for (auto& weight : weights) {
    weight = weight.unaryExpr(
        [&](float value) { return detail::normalizedComponent(value, accessor.componentType); });
  }
  return weights;
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
    }

    // load skinning weight buffer
    auto weightsAttribute = primitive.attributes.find(std::string("WEIGHTS_") + std::to_string(i));
    std::vector<Vector4f> weightsData;
    if (weightsAttribute != primitive.attributes.end()) {
      weightsData = loadWeightsFromAccessor(model, weightsAttribute->second);
      if (weightsData.empty()) {
        MT_LOGW("No skinning weights read");
        break;
      }
    } else {
      MT_LOGW_IF(i == 0, "No skinning weights stored on primitive");
      break;
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
      break;
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

  // The glTF format allows a zero weight before a nonzero one, but applySSD() and the other
  // skinning code stop at the first zero weight of a vertex, so the nonzero weights go first. The
  // order of the weights is kept otherwise.
  for (auto v = kVertexOffset; v < kVertexOffset + static_cast<Eigen::Index>(kNumVertices); ++v) {
    std::array<std::pair<uint32_t, float>, kMaxSkinJoints> slots;
    size_t next = 0;
    for (const bool nonzero : {true, false}) {
      for (size_t slot = 0; slot < kMaxSkinJoints; ++slot) {
        if ((skinWeights->weight(v, slot) != 0.0f) == nonzero) {
          slots[next++] = {skinWeights->index(v, slot), skinWeights->weight(v, slot)};
        }
      }
    }
    for (size_t slot = 0; slot < kMaxSkinJoints; ++slot) {
      std::tie(skinWeights->index(v, slot), skinWeights->weight(v, slot)) = slots[slot];
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
        const auto weights = morphWeightsOfNode(node, mesh);
        const auto kNumVertices = addMesh(model, primitive, resultMesh, weights);
        addSkinWeights(model, skin, primitive, nodeToObjectMap, kNumVertices, skinWeights);
        addBlendShapes(model, primitive, blendShape, weights);
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
        const auto weights = morphWeightsOfNode(node, mesh);
        addMesh(model, primitive, resultMesh, weights);
        addBlendShapes(model, primitive, blendShape, weights);
      }
    }
    return {std::move(resultMesh), nullptr, std::move(blendShape)};
  } else {
    return {};
  }
  return {std::move(resultMesh), std::move(skinWeights), std::move(blendShape)};
}

namespace {

// Throws where the mesh reader would stop the process on `primitive` (its MT_CHECKs), so that an
// object with such a mesh is skipped with a warning (see buildObjectCharacter()): attributes and
// morph targets whose element counts differ from the positions, and triangles whose vertex count is
// not a multiple of 3. These checks are stricter than the reader's, which skips attributes it
// cannot read.
void checkObjectPrimitive(const fx::gltf::Document& model, const fx::gltf::Primitive& primitive) {
  const auto countOf = [&](uint32_t id) {
    MT_THROW_IF(
        id >= model.accessors.size(),
        "Accessor {} does not exist; the file has {} accessors",
        id,
        model.accessors.size());
    return model.accessors[id].count;
  };
  const auto position = primitive.attributes.find("POSITION");
  if (position == primitive.attributes.end()) {
    return;
  }
  const uint32_t numPositions = countOf(position->second);
  for (const char* name : {"NORMAL", "COLOR_0", "TEXCOORD_0"}) {
    const auto attribute = primitive.attributes.find(name);
    if (attribute == primitive.attributes.end()) {
      continue;
    }
    const uint32_t count = countOf(attribute->second);
    MT_THROW_IF(
        count != numPositions,
        "The {} attribute of a glTF mesh has {} elements for {} positions",
        name,
        count,
        numPositions);
  }
  if (primitive.mode == fx::gltf::Primitive::Mode::Triangles) {
    const uint32_t numVertices =
        primitive.indices >= 0 ? countOf(static_cast<uint32_t>(primitive.indices)) : numPositions;
    MT_THROW_IF(
        numVertices % 3 != 0,
        "A glTF triangle mesh has {} vertices, which is not a multiple of 3",
        numVertices);
  }
  if (primitive.targets.empty()) {
    return;
  }
  const size_t numBasePositions = copyAccessorBuffer<Vector3f>(model, position->second).size();
  for (const auto& target : primitive.targets) {
    const auto targetPosition = target.find("POSITION");
    if (targetPosition == target.end()) {
      continue;
    }
    countOf(targetPosition->second);
    const size_t count = copyAccessorBuffer<Vector3f>(model, targetPosition->second).size();
    MT_THROW_IF(
        count != numBasePositions,
        "A morph target of a glTF mesh has {} positions for {}",
        count,
        numBasePositions);
  }
}

} // namespace

std::tuple<Mesh_u, SkinWeights_u, BlendShape_u> loadRigidMeshes(
    const fx::gltf::Document& model,
    std::span<const size_t> meshNodes,
    std::span<const size_t> nodeToObjectMap) {
  auto mesh = std::make_unique<Mesh>();
  auto skinWeights = std::make_unique<SkinWeights>();
  BlendShape_u blendShape;
  for (const size_t nodeId : meshNodes) {
    const auto& node = model.nodes.at(nodeId);
    MT_THROW_IF(
        nodeId >= nodeToObjectMap.size(),
        "glTF node {} is not in the node map of {} nodes",
        nodeId,
        nodeToObjectMap.size());
    const auto jointId = static_cast<uint32_t>(nodeToObjectMap[nodeId]);
    const auto& gltfMesh = model.meshes.at(node.mesh);
    const auto weights = morphWeightsOfNode(node, gltfMesh);
    for (const auto& primitive : gltfMesh.primitives) {
      checkObjectPrimitive(model, primitive);
      const auto numVertices = static_cast<Eigen::Index>(addMesh(model, primitive, mesh, weights));
      const Eigen::Index start = skinWeights->index.rows();
      skinWeights->index.conservativeResize(start + numVertices, Eigen::NoChange);
      skinWeights->weight.conservativeResize(start + numVertices, Eigen::NoChange);
      skinWeights->index.bottomRows(numVertices).setZero();
      skinWeights->weight.bottomRows(numVertices).setZero();
      skinWeights->index.bottomRows(numVertices).col(0).setConstant(jointId);
      skinWeights->weight.bottomRows(numVertices).col(0).setOnes();
      addBlendShapes(model, primitive, blendShape, weights);
    }
  }
  return {std::move(mesh), std::move(skinWeights), std::move(blendShape)};
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
