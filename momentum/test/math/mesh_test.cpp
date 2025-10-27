/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/math/utility.h"

using namespace momentum;

// Test fixture for mesh tests
template <typename T>
class MeshTest : public testing::Test {
 protected:
  using MeshType = MeshT<T>;
};

using Types = testing::Types<float, double>;
TYPED_TEST_SUITE(MeshTest, Types);

// Test default constructor
TYPED_TEST(MeshTest, DefaultConstructor) {
  using MeshType = typename TestFixture::MeshType;

  MeshType mesh;

  EXPECT_TRUE(mesh.vertices.empty());
  EXPECT_TRUE(mesh.normals.empty());
  EXPECT_TRUE(mesh.faces.empty());
  EXPECT_TRUE(mesh.lines.empty());
  EXPECT_TRUE(mesh.colors.empty());
  EXPECT_TRUE(mesh.confidence.empty());
  EXPECT_TRUE(mesh.texcoords.empty());
  EXPECT_TRUE(mesh.texcoord_faces.empty());
  EXPECT_TRUE(mesh.texcoord_lines.empty());
}

// Test updateNormals with a tetrahedron
TYPED_TEST(MeshTest, UpdateNormals) {
  using T = TypeParam;
  using MeshType = typename TestFixture::MeshType;
  using Vector3 = Eigen::Vector3<T>;

  MeshType mesh;

  // Create a tetrahedron
  mesh.vertices = {
      Vector3(0, 0, 0), // 0: origin
      Vector3(1, 0, 0), // 1: x-axis
      Vector3(0, 1, 0), // 2: y-axis
      Vector3(0, 0, 1) // 3: z-axis
  };

  mesh.faces = {
      Eigen::Vector3i(0, 1, 2), // base
      Eigen::Vector3i(0, 1, 3), // side 1
      Eigen::Vector3i(0, 2, 3), // side 2
      Eigen::Vector3i(1, 2, 3) // side 3
  };

  // Update normals
  mesh.updateNormals();

  // Check that we have the correct number of normals
  EXPECT_EQ(mesh.normals.size(), mesh.vertices.size());

  // Check that all normals are normalized
  for (const auto& normal : mesh.normals) {
    EXPECT_NEAR(normal.norm(), 1.0, Eps<T>(1e-5f, 1e-13));
  }

  // Check that all normals are valid
  for (const auto& normal : mesh.normals) {
    // Check that the normal is not NaN
    EXPECT_FALSE(IsNanNoOpt(normal[0]));
    EXPECT_FALSE(IsNanNoOpt(normal[1]));
    EXPECT_FALSE(IsNanNoOpt(normal[2]));
  }
}

// Test updateNormals with invalid face indices
TYPED_TEST(MeshTest, UpdateNormalsInvalidIndices) {
  using T = TypeParam;
  using MeshType = typename TestFixture::MeshType;
  using Vector3 = Eigen::Vector3<T>;

  MeshType mesh;

  // Create a simple mesh
  mesh.vertices = {Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)};

  // Add a face with an invalid index
  mesh.faces = {
      Eigen::Vector3i(0, 1, 2), // valid
      Eigen::Vector3i(0, 1, 3) // invalid (index 3 is out of bounds)
  };

  // Update normals should not crash
  EXPECT_NO_THROW(mesh.updateNormals());

  // Check that we have the correct number of normals
  EXPECT_EQ(mesh.normals.size(), mesh.vertices.size());

  // Check that the normal for the valid face is computed correctly
  Vector3 expectedNormal(0, 0, 1); // normal to the xy-plane
  EXPECT_TRUE(mesh.normals[0].isApprox(expectedNormal));
  EXPECT_TRUE(mesh.normals[1].isApprox(expectedNormal));
  EXPECT_TRUE(mesh.normals[2].isApprox(expectedNormal));
}

// Test updateNormals with degenerate faces
TYPED_TEST(MeshTest, UpdateNormalsDegenerate) {
  using T = TypeParam;
  using MeshType = typename TestFixture::MeshType;
  using Vector3 = Eigen::Vector3<T>;

  MeshType mesh;

  // Create a mesh with a degenerate face (colinear vertices)
  mesh.vertices = {
      Vector3(0, 0, 0),
      Vector3(1, 0, 0),
      Vector3(2, 0, 0), // colinear with the first two
      Vector3(0, 1, 0)};

  mesh.faces = {
      Eigen::Vector3i(0, 1, 2), // degenerate (colinear)
      Eigen::Vector3i(0, 1, 3) // valid
  };

  // Update normals should not crash
  EXPECT_NO_THROW(mesh.updateNormals());

  // Check that we have the correct number of normals
  EXPECT_EQ(mesh.normals.size(), mesh.vertices.size());

  // Check that the normal for the valid face is computed correctly
  Vector3 expectedNormal(0, 0, 1); // normal to the xy-plane
  EXPECT_TRUE(mesh.normals[0].isApprox(expectedNormal));
  EXPECT_TRUE(mesh.normals[1].isApprox(expectedNormal));
  EXPECT_TRUE(mesh.normals[3].isApprox(expectedNormal));

  // The normal for vertex 2 should be zero since it's only part of the degenerate face
  EXPECT_TRUE(mesh.normals[2].isZero());
}

// Test cast method
TYPED_TEST(MeshTest, Cast) {
  using T = TypeParam;
  using MeshType = typename TestFixture::MeshType;
  using Vector3 = Eigen::Vector3<T>;

  MeshType mesh;

  // Create a simple mesh
  mesh.vertices = {Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)};

  mesh.normals = {Vector3(0, 0, 1), Vector3(0, 0, 1), Vector3(0, 0, 1)};

  mesh.faces = {Eigen::Vector3i(0, 1, 2)};

  mesh.confidence = {0.5, 0.6, 0.7};

  // Cast to the same type (should be a copy)
  auto sameMesh = mesh.template cast<T>();

  EXPECT_EQ(sameMesh.vertices.size(), mesh.vertices.size());
  EXPECT_EQ(sameMesh.normals.size(), mesh.normals.size());
  EXPECT_EQ(sameMesh.faces.size(), mesh.faces.size());
  EXPECT_EQ(sameMesh.confidence.size(), mesh.confidence.size());

  for (size_t i = 0; i < mesh.vertices.size(); ++i) {
    EXPECT_TRUE(sameMesh.vertices[i].isApprox(mesh.vertices[i]));
    EXPECT_TRUE(sameMesh.normals[i].isApprox(mesh.normals[i]));
    EXPECT_NEAR(sameMesh.confidence[i], mesh.confidence[i], Eps<T>(1e-5f, 1e-13));
  }

  // Cast to the other type
  using OtherT = typename std::conditional<std::is_same<T, float>::value, double, float>::type;
  auto otherMesh = mesh.template cast<OtherT>();

  EXPECT_EQ(otherMesh.vertices.size(), mesh.vertices.size());
  EXPECT_EQ(otherMesh.normals.size(), mesh.normals.size());
  EXPECT_EQ(otherMesh.faces.size(), mesh.faces.size());
  EXPECT_EQ(otherMesh.confidence.size(), mesh.confidence.size());

  for (size_t i = 0; i < mesh.vertices.size(); ++i) {
    EXPECT_TRUE(otherMesh.vertices[i].template cast<T>().isApprox(mesh.vertices[i]));
    EXPECT_TRUE(otherMesh.normals[i].template cast<T>().isApprox(mesh.normals[i]));
    // Use a more relaxed tolerance for float-to-double or double-to-float conversions
    EXPECT_NEAR(static_cast<T>(otherMesh.confidence[i]), mesh.confidence[i], Eps<T>(1e-4f, 1e-6));
  }
}

// Test reset method
TYPED_TEST(MeshTest, Reset) {
  using T = TypeParam;
  using MeshType = typename TestFixture::MeshType;
  using Vector3 = Eigen::Vector3<T>;

  MeshType mesh;

  // Create a simple mesh
  mesh.vertices = {Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)};

  mesh.normals = {Vector3(0, 0, 1), Vector3(0, 0, 1), Vector3(0, 0, 1)};

  mesh.faces = {Eigen::Vector3i(0, 1, 2)};

  mesh.lines = {{0, 1}, {1, 2}, {2, 0}};

  mesh.colors = {
      Eigen::Vector3b(255, 0, 0), Eigen::Vector3b(0, 255, 0), Eigen::Vector3b(0, 0, 255)};

  mesh.confidence = {0.5, 0.6, 0.7};

  mesh.texcoords = {Eigen::Vector2f(0, 0), Eigen::Vector2f(1, 0), Eigen::Vector2f(0, 1)};

  mesh.texcoord_faces = {Eigen::Vector3i(0, 1, 2)};

  mesh.texcoord_lines = {{0, 1}, {1, 2}, {2, 0}};

  // Reset the mesh
  mesh.reset();

  // Check that all data members are empty
  EXPECT_TRUE(mesh.vertices.empty());
  EXPECT_TRUE(mesh.normals.empty());
  EXPECT_TRUE(mesh.faces.empty());
  EXPECT_TRUE(mesh.lines.empty());
  EXPECT_TRUE(mesh.colors.empty());
  EXPECT_TRUE(mesh.confidence.empty());
  EXPECT_TRUE(mesh.texcoords.empty());
  EXPECT_TRUE(mesh.texcoord_faces.empty());
  EXPECT_TRUE(mesh.texcoord_lines.empty());
}

// Test with a complex mesh
TYPED_TEST(MeshTest, ComplexMesh) {
  using T = TypeParam;
  using MeshType = typename TestFixture::MeshType;
  using Vector3 = Eigen::Vector3<T>;

  MeshType mesh;

  // Create a cube mesh
  mesh.vertices = {
      Vector3(0, 0, 0), // 0: bottom-left-front
      Vector3(1, 0, 0), // 1: bottom-right-front
      Vector3(1, 1, 0), // 2: top-right-front
      Vector3(0, 1, 0), // 3: top-left-front
      Vector3(0, 0, 1), // 4: bottom-left-back
      Vector3(1, 0, 1), // 5: bottom-right-back
      Vector3(1, 1, 1), // 6: top-right-back
      Vector3(0, 1, 1) // 7: top-left-back
  };

  // Define the faces of the cube (6 faces, 2 triangles each)
  mesh.faces = {
      // Front face
      Eigen::Vector3i(0, 1, 2),
      Eigen::Vector3i(0, 2, 3),
      // Right face
      Eigen::Vector3i(1, 5, 6),
      Eigen::Vector3i(1, 6, 2),
      // Back face
      Eigen::Vector3i(5, 4, 7),
      Eigen::Vector3i(5, 7, 6),
      // Left face
      Eigen::Vector3i(4, 0, 3),
      Eigen::Vector3i(4, 3, 7),
      // Top face
      Eigen::Vector3i(3, 2, 6),
      Eigen::Vector3i(3, 6, 7),
      // Bottom face
      Eigen::Vector3i(4, 5, 1),
      Eigen::Vector3i(4, 1, 0)};

  // Define the edges of the cube
  mesh.lines = {
      {0, 1},
      {1, 2},
      {2, 3},
      {3, 0}, // front face
      {4, 5},
      {5, 6},
      {6, 7},
      {7, 4}, // back face
      {0, 4},
      {1, 5},
      {2, 6},
      {3, 7} // connecting edges
  };

  // Update normals
  mesh.updateNormals();

  // Check that we have the correct number of normals
  EXPECT_EQ(mesh.normals.size(), mesh.vertices.size());

  // Check that all normals are normalized
  for (const auto& normal : mesh.normals) {
    EXPECT_NEAR(normal.norm(), 1.0, Eps<T>(1e-5f, 1e-13));
  }

  // Check that all normals are valid and normalized
  for (const auto& normal : mesh.normals) {
    // Check that the normal is not NaN
    EXPECT_FALSE(IsNanNoOpt(normal[0]));
    EXPECT_FALSE(IsNanNoOpt(normal[1]));
    EXPECT_FALSE(IsNanNoOpt(normal[2]));

    // Check that the normal is normalized
    EXPECT_NEAR(normal.norm(), 1.0, Eps<T>(1e-5f, 1e-13));
  }

  // Test reset
  mesh.reset();
  EXPECT_TRUE(mesh.vertices.empty());
  EXPECT_TRUE(mesh.normals.empty());
  EXPECT_TRUE(mesh.faces.empty());
  EXPECT_TRUE(mesh.lines.empty());
}

// Test with an empty mesh
TYPED_TEST(MeshTest, EmptyMesh) {
  using T = TypeParam;
  using MeshType = typename TestFixture::MeshType;

  MeshType mesh;

  // Update normals on an empty mesh should not crash
  EXPECT_NO_THROW(mesh.updateNormals());

  // Check that normals is still empty
  EXPECT_TRUE(mesh.normals.empty());

  // Cast an empty mesh
  auto castedMesh = mesh.template cast<T>();
  EXPECT_TRUE(castedMesh.vertices.empty());
  EXPECT_TRUE(castedMesh.normals.empty());
  EXPECT_TRUE(castedMesh.faces.empty());
  EXPECT_TRUE(castedMesh.lines.empty());
  EXPECT_TRUE(castedMesh.colors.empty());
  EXPECT_TRUE(castedMesh.confidence.empty());
  EXPECT_TRUE(castedMesh.texcoords.empty());
  EXPECT_TRUE(castedMesh.texcoord_faces.empty());
  EXPECT_TRUE(castedMesh.texcoord_lines.empty());

  // Reset an empty mesh
  EXPECT_NO_THROW(mesh.reset());
  EXPECT_TRUE(mesh.vertices.empty());
}

// Test with texture coordinates
TYPED_TEST(MeshTest, TextureCoordinates) {
  using T = TypeParam;
  using MeshType = typename TestFixture::MeshType;
  using Vector3 = Eigen::Vector3<T>;

  MeshType mesh;

  // Create a simple mesh with texture coordinates
  mesh.vertices = {Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)};

  mesh.faces = {Eigen::Vector3i(0, 1, 2)};

  mesh.texcoords = {Eigen::Vector2f(0, 0), Eigen::Vector2f(1, 0), Eigen::Vector2f(0, 1)};

  mesh.texcoord_faces = {Eigen::Vector3i(0, 1, 2)};

  // Cast to another type
  using OtherT = typename std::conditional<std::is_same<T, float>::value, double, float>::type;
  auto otherMesh = mesh.template cast<OtherT>();

  // Check that texture coordinates are preserved
  EXPECT_EQ(otherMesh.texcoords.size(), mesh.texcoords.size());
  EXPECT_EQ(otherMesh.texcoord_faces.size(), mesh.texcoord_faces.size());

  for (size_t i = 0; i < mesh.texcoords.size(); ++i) {
    EXPECT_TRUE(otherMesh.texcoords[i].isApprox(mesh.texcoords[i]));
  }

  for (size_t i = 0; i < mesh.texcoord_faces.size(); ++i) {
    EXPECT_EQ(otherMesh.texcoord_faces[i], mesh.texcoord_faces[i]);
  }

  // Reset and check that texture coordinates are cleared
  mesh.reset();
  EXPECT_TRUE(mesh.texcoords.empty());
  EXPECT_TRUE(mesh.texcoord_faces.empty());
}

// Test with colors and confidence
TYPED_TEST(MeshTest, ColorsAndConfidence) {
  using T = TypeParam;
  using MeshType = typename TestFixture::MeshType;
  using Vector3 = Eigen::Vector3<T>;

  MeshType mesh;

  // Create a simple mesh with colors and confidence
  mesh.vertices = {Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)};

  mesh.colors = {
      Eigen::Vector3b(255, 0, 0), Eigen::Vector3b(0, 255, 0), Eigen::Vector3b(0, 0, 255)};

  mesh.confidence = {0.5, 0.6, 0.7};

  // Cast to another type
  using OtherT = typename std::conditional<std::is_same<T, float>::value, double, float>::type;
  auto otherMesh = mesh.template cast<OtherT>();

  // Check that colors are preserved
  EXPECT_EQ(otherMesh.colors.size(), mesh.colors.size());
  for (size_t i = 0; i < mesh.colors.size(); ++i) {
    EXPECT_EQ(otherMesh.colors[i], mesh.colors[i]);
  }

  // Check that confidence values are preserved
  EXPECT_EQ(otherMesh.confidence.size(), mesh.confidence.size());
  for (size_t i = 0; i < mesh.confidence.size(); ++i) {
    // Use a more relaxed tolerance for float-to-double or double-to-float conversions
    EXPECT_NEAR(static_cast<T>(otherMesh.confidence[i]), mesh.confidence[i], Eps<T>(1e-4f, 1e-6));
  }

  // Reset and check that colors and confidence are cleared
  mesh.reset();
  EXPECT_TRUE(mesh.colors.empty());
  EXPECT_TRUE(mesh.confidence.empty());
}

// Test with lines
TYPED_TEST(MeshTest, Lines) {
  using T = TypeParam;
  using MeshType = typename TestFixture::MeshType;
  using Vector3 = Eigen::Vector3<T>;

  MeshType mesh;

  // Create a simple mesh with lines
  mesh.vertices = {Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(1, 1, 0)};

  mesh.lines = {{0, 1}, {1, 3}, {3, 2}, {2, 0}};

  // Cast to another type
  using OtherT = typename std::conditional<std::is_same<T, float>::value, double, float>::type;
  auto otherMesh = mesh.template cast<OtherT>();

  // Check that lines are preserved
  EXPECT_EQ(otherMesh.lines.size(), mesh.lines.size());
  for (size_t i = 0; i < mesh.lines.size(); ++i) {
    EXPECT_EQ(otherMesh.lines[i], mesh.lines[i]);
  }

  // Reset and check that lines are cleared
  mesh.reset();
  EXPECT_TRUE(mesh.lines.empty());
}
