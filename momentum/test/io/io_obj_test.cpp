/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/obj/obj_io.h"
#include "momentum/test/io/io_helpers.h"

#include <gtest/gtest.h>

#include <fstream>
#include <string>

using namespace momentum;

namespace {

/// Write `contents` to a temporary .obj and load it.
Mesh loadFromText(const std::string& name, const std::string& contents) {
  const TemporaryFile file = temporaryFile(name, "obj");
  {
    std::ofstream out(file.path());
    out << contents;
  }
  return loadObj(file.path());
}

// Two triangles sharing an edge, each corner with its own texture coordinate.
constexpr const char* kQuadAsTriangles = R"(# comment line
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3
f 1/1 3/3 4/4
)";

} // namespace

TEST(IoObjTest, ReadsPositionsFacesAndTexcoords) {
  const Mesh mesh = loadFromText("obj_basic", kQuadAsTriangles);

  EXPECT_EQ(mesh.vertices.size(), 4);
  EXPECT_EQ(mesh.faces.size(), 2);
  EXPECT_EQ(mesh.texcoords.size(), 4);
  // texcoord_faces must stay index-aligned with faces or downstream indexing breaks.
  EXPECT_EQ(mesh.texcoord_faces.size(), mesh.faces.size());

  EXPECT_EQ(mesh.vertices[2], Eigen::Vector3f(1.0f, 1.0f, 0.0f));
  // OBJ indices are 1-based; the loader must rebase them to 0.
  EXPECT_EQ(mesh.faces[1], Eigen::Vector3i(0, 2, 3));
  EXPECT_EQ(mesh.texcoord_faces[1], Eigen::Vector3i(0, 2, 3));
  EXPECT_EQ(mesh.texcoords[3], Eigen::Vector2f(0.0f, 1.0f));

  // Per-vertex arrays the rest of momentum assumes are populated.
  EXPECT_EQ(mesh.normals.size(), mesh.vertices.size());
  EXPECT_EQ(mesh.colors.size(), mesh.vertices.size());
}

TEST(IoObjTest, FanTriangulatesPolygonsAndKeepsOriginalTopology) {
  const Mesh mesh = loadFromText(
      "obj_quad",
      "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 0\n"
      "f 1 2 3 4\n"
      "f 2 5 3\n");

  // A quad fans into 2 triangles, plus the standalone triangle.
  EXPECT_EQ(mesh.faces.size(), 3);
  EXPECT_EQ(mesh.faces[0], Eigen::Vector3i(0, 1, 2));
  EXPECT_EQ(mesh.faces[1], Eigen::Vector3i(0, 2, 3));

  // The un-triangulated topology is preserved alongside it.
  const std::vector<uint32_t> expectedSizes{4, 3};
  EXPECT_EQ(mesh.polyFaceSizes, expectedSizes);
  const std::vector<uint32_t> expectedPolyFaces{0, 1, 2, 3, 1, 4, 2};
  EXPECT_EQ(mesh.polyFaces, expectedPolyFaces);
}

TEST(IoObjTest, AcceptsEveryCornerFormAndNegativeIndices) {
  // `v//vn` carries a normal but no texture coordinate, and -1 means "the vertex most
  // recently read", so both faces below address the same three vertices.
  const Mesh mesh = loadFromText(
      "obj_forms",
      "v 0 0 0\nv 1 0 0\nv 1 1 0\n"
      "vn 0 0 1\n"
      "f 1 2 3\n"
      "f 1//1 2//1 3//1\n"
      "f -3 -2 -1\n");

  EXPECT_EQ(mesh.faces.size(), 3);
  for (const auto& face : mesh.faces) {
    EXPECT_EQ(face, Eigen::Vector3i(0, 1, 2));
  }
  // No face carried a vt index, so the mesh is untextured rather than partly textured.
  EXPECT_TRUE(mesh.texcoords.empty());
  EXPECT_TRUE(mesh.texcoord_faces.empty());
}

TEST(IoObjTest, DropsTexcoordsWhenOnlySomeFacesHaveThem) {
  // texcoord_faces cannot be partially populated, so a mixed file loads untextured.
  const Mesh mesh = loadFromText(
      "obj_mixed",
      "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
      "vt 0 0\nvt 1 0\nvt 1 1\n"
      "f 1/1 2/2 3/3\n"
      "f 1 3 4\n");

  EXPECT_EQ(mesh.faces.size(), 2);
  EXPECT_TRUE(mesh.texcoords.empty());
  EXPECT_TRUE(mesh.texcoord_faces.empty());
}

TEST(IoObjTest, ReadsVertexColorsFromTheSixFloatForm) {
  const Mesh mesh = loadFromText(
      "obj_colors",
      "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 1 1 0 0 0 1\n"
      "f 1 2 3\n");

  ASSERT_EQ(mesh.colors.size(), 3);
  EXPECT_EQ(mesh.colors[0], Vector3b(255, 0, 0));
  EXPECT_EQ(mesh.colors[2], Vector3b(0, 0, 255));
}

TEST(IoObjTest, RejectsOutOfRangeAndEmptyFiles) {
  EXPECT_THROW(loadFromText("obj_oor", "v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 9\n"), std::runtime_error);
  EXPECT_THROW(loadFromText("obj_novertex", "# nothing here\n"), std::runtime_error);
  EXPECT_THROW(loadFromText("obj_noface", "v 0 0 0\nv 1 0 0\n"), std::runtime_error);
}

TEST(IoObjTest, ReportsTheLineAndTokenForAMalformedIndex) {
  // std::stoll alone throws a bare std::invalid_argument naming neither the line
  // nor the token, unlike every other diagnostic this parser emits.
  const auto expectMentions =
      [](const std::string& name, const std::string& text, const std::string& needle) {
        try {
          loadFromText(name, text);
          FAIL() << "expected a throw for " << name;
        } catch (const std::runtime_error& e) {
          const std::string what = e.what();
          EXPECT_NE(what.find("line 4"), std::string::npos) << what;
          EXPECT_NE(what.find(needle), std::string::npos) << what;
        }
      };
  const std::string prefix = "v 0 0 0\nv 1 0 0\nv 1 1 0\n";
  expectMentions("obj_bad_vertex_index", prefix + "f abc 2 3\n", "abc");
  expectMentions("obj_bad_texcoord_index", prefix + "f 1/xy 2 3\n", "xy");
  expectMentions("obj_trailing_chars", prefix + "f 1x 2 3\n", "1x");
}

TEST(IoObjTest, RejectsAMalformedTextureCoordinate) {
  // A short or non-numeric vt used to fall through as (0, 0), silently corrupting
  // UVs rather than failing the way a malformed vertex or face does.
  EXPECT_THROW(loadFromText("obj_vt_short", "v 0 0 0\nvt 0.5\nf 1 1 1\n"), std::runtime_error);
  EXPECT_THROW(
      loadFromText("obj_vt_nonnumeric", "v 0 0 0\nvt foo bar\nf 1 1 1\n"), std::runtime_error);
}

TEST(IoObjTest, RecomputesNormalsRatherThanReadingThem) {
  // Normals come from updateNormals(), not from vn: a Mesh stores one normal per
  // vertex while OBJ indexes them per face corner, so the file's values are junk
  // here and must not survive.
  const Mesh mesh = loadFromText(
      "obj_vn",
      "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
      "vn 1 0 0\nvn 1 0 0\nvn 1 0 0\n"
      "f 1//1 2//2 3//3\n");
  ASSERT_EQ(mesh.normals.size(), 3);
  // The triangle lies in z=0, so a geometric normal is +/-Z, never the +X in the file.
  for (const auto& n : mesh.normals) {
    EXPECT_NEAR(std::abs(n.z()), 1.0f, 1e-5f) << n.transpose();
    EXPECT_NEAR(n.x(), 0.0f, 1e-5f) << n.transpose();
  }
}

TEST(IoObjTest, LoadsFromMemoryIdenticallyToDisk) {
  const Mesh fromDisk = loadFromText("obj_mem", kQuadAsTriangles);
  const std::string text(kQuadAsTriangles);
  const Mesh fromMemory = loadObj(std::as_bytes(std::span<const char>(text.data(), text.size())));

  ASSERT_EQ(fromMemory.vertices.size(), fromDisk.vertices.size());
  EXPECT_EQ(fromMemory.faces, fromDisk.faces);
  EXPECT_EQ(fromMemory.texcoord_faces, fromDisk.texcoord_faces);
}
