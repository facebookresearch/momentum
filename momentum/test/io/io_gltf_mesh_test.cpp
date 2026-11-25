/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/character.h"
#include "momentum/character/skin_weights.h"
#include "momentum/io/gltf/gltf_io.h"
#include "momentum/math/mesh.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

using namespace momentum;

namespace {

template <typename T>
void appendBytes(std::vector<std::byte>& out, const T& value) {
  const auto* bytes = reinterpret_cast<const std::byte*>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(T));
}

/// Build a binary glTF holding two triangles (six vertices, no shared corners), optionally
/// skinned to a single joint and optionally with an index buffer.
std::vector<std::byte> makeTwoTriangleGlb(bool indexed, bool skinned) {
  // Each triangle has its own three vertices; positions are in meters (glTF units).
  const std::vector<std::array<float, 3>> positions = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  constexpr uint32_t kNumVertices = 6;

  std::vector<std::byte> bin;
  std::string views;
  std::string accessors;
  auto addView = [&](const void* data, size_t size) {
    if (!views.empty()) {
      views += ",";
    }
    views += R"({"buffer":0,"byteOffset":)" + std::to_string(bin.size()) + R"(,"byteLength":)" +
        std::to_string(size) + "}";
    const auto* bytes = static_cast<const std::byte*>(data);
    bin.insert(bin.end(), bytes, bytes + size);
  };
  int numAccessors = 0;
  auto addAccessor = [&](int componentType, const char* type, const std::string& extra = "") {
    if (!accessors.empty()) {
      accessors += ",";
    }
    accessors += R"({"bufferView":)" + std::to_string(numAccessors) + R"(,"componentType":)" +
        std::to_string(componentType) + R"(,"count":)" + std::to_string(kNumVertices) +
        R"(,"type":")" + type + "\"" + extra + "}";
    return numAccessors++;
  };

  addView(positions.data(), positions.size() * sizeof(positions[0]));
  const int position = addAccessor(5126, "VEC3", R"(,"min":[0,0,0],"max":[1,1,0])");
  std::string attributes = R"("POSITION":)" + std::to_string(position);

  const std::vector<uint16_t> joints(kNumVertices * 4, 0);
  std::vector<float> weights(kNumVertices * 4, 0.0f);
  if (skinned) {
    for (uint32_t v = 0; v < kNumVertices; ++v) {
      weights[v * 4] = 1.0f;
    }
    addView(joints.data(), joints.size() * sizeof(uint16_t));
    const int jointsAccessor = addAccessor(5123, "VEC4");
    addView(weights.data(), weights.size() * sizeof(float));
    const int weightsAccessor = addAccessor(5126, "VEC4");
    attributes += R"(,"JOINTS_0":)" + std::to_string(jointsAccessor) + R"(,"WEIGHTS_0":)" +
        std::to_string(weightsAccessor);
  }

  std::string primitive = "{\"attributes\":{" + attributes + "},\"mode\":4";
  const std::vector<uint16_t> indices = {0, 1, 2, 3, 4, 5};
  if (indexed) {
    addView(indices.data(), indices.size() * sizeof(uint16_t));
    primitive += R"(,"indices":)" + std::to_string(addAccessor(5123, "SCALAR"));
  }
  primitive += "}";

  const std::string nodes = skinned ? R"([{"name":"root"},{"name":"mesh","mesh":0,"skin":0}])"
                                    : R"([{"name":"mesh","mesh":0}])";
  const std::string sceneNodes = skinned ? "[0,1]" : "[0]";
  std::string json = R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":)" + sceneNodes +
      R"(}],"nodes":)" + nodes + R"(,"meshes":[{"primitives":[)" + primitive +
      R"(]}],"accessors":[)" + accessors + R"(],"bufferViews":[)" + views +
      R"(],"buffers":[{"byteLength":)" + std::to_string(bin.size()) + "}]";
  if (skinned) {
    json += R"(,"skins":[{"joints":[0]}])";
  }
  json += "}";

  // GLB chunks must be 4-byte aligned: pad JSON with spaces and the binary chunk with zeros.
  json.append((4 - json.size() % 4) % 4, ' ');
  bin.resize(bin.size() + (4 - bin.size() % 4) % 4, std::byte{0});

  std::vector<std::byte> glb;
  appendBytes(glb, uint32_t{0x46546C67}); // "glTF"
  appendBytes(glb, uint32_t{2});
  appendBytes(glb, static_cast<uint32_t>(12 + 8 + json.size() + 8 + bin.size()));
  appendBytes(glb, static_cast<uint32_t>(json.size()));
  appendBytes(glb, uint32_t{0x4E4F534A}); // "JSON"
  const auto* jsonBytes = reinterpret_cast<const std::byte*>(json.data());
  glb.insert(glb.end(), jsonBytes, jsonBytes + json.size());
  appendBytes(glb, static_cast<uint32_t>(bin.size()));
  appendBytes(glb, uint32_t{0x004E4942}); // "BIN\0"
  glb.insert(glb.end(), bin.begin(), bin.end());
  return glb;
}

void expectSameMesh(const Character& actual, const Character& expected) {
  ASSERT_NE(actual.mesh, nullptr);
  ASSERT_NE(expected.mesh, nullptr);
  ASSERT_EQ(actual.mesh->vertices.size(), expected.mesh->vertices.size());
  for (size_t i = 0; i < expected.mesh->vertices.size(); ++i) {
    EXPECT_TRUE(actual.mesh->vertices[i].isApprox(expected.mesh->vertices[i])) << "vertex " << i;
  }
  EXPECT_EQ(actual.mesh->faces, expected.mesh->faces);
  ASSERT_NE(actual.skinWeights, nullptr);
  ASSERT_NE(expected.skinWeights, nullptr);
  EXPECT_EQ(actual.skinWeights->index, expected.skinWeights->index);
  EXPECT_EQ(actual.skinWeights->weight, expected.skinWeights->weight);
}

class GltfMeshTest : public testing::TestWithParam<bool> {};

// A glTF primitive without "indices" is non-indexed geometry: every three consecutive vertices
// form a triangle (as the Khronos "Fox" sample stores its mesh). It must load like the same
// triangles written with an index buffer.
TEST_P(GltfMeshTest, NonIndexedTrianglesLoadLikeIndexedOnes) {
  const bool skinned = GetParam();
  const auto indexedGlb = makeTwoTriangleGlb(/*indexed=*/true, skinned);
  const auto nonIndexedGlb = makeTwoTriangleGlb(/*indexed=*/false, skinned);
  const Character indexed = loadGltfCharacter(std::span<const std::byte>(indexedGlb));
  const Character nonIndexed = loadGltfCharacter(std::span<const std::byte>(nonIndexedGlb));

  ASSERT_NE(indexed.mesh, nullptr);
  ASSERT_EQ(indexed.mesh->vertices.size(), 6u);
  ASSERT_NE(nonIndexed.mesh, nullptr);
  EXPECT_EQ(nonIndexed.mesh->vertices.size(), 6u);
  ASSERT_EQ(nonIndexed.mesh->faces.size(), 2u);
  EXPECT_EQ(nonIndexed.mesh->faces[0], Eigen::Vector3i(0, 1, 2));
  EXPECT_EQ(nonIndexed.mesh->faces[1], Eigen::Vector3i(3, 4, 5));
  expectSameMesh(nonIndexed, indexed);
}

INSTANTIATE_TEST_SUITE_P(
    SkinnedAndUnskinned,
    GltfMeshTest,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) { return info.param ? "Skinned" : "Unskinned"; });

} // namespace
