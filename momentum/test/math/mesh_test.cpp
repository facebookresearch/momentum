/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/math/mesh.h"

using namespace momentum;

TEST(Momentum_Mesh, Construction) {
  Mesh m;
  EXPECT_TRUE(m.vertices.empty());
  EXPECT_TRUE(m.normals.empty());
  EXPECT_TRUE(m.faces.empty());
  EXPECT_TRUE(m.lines.empty());
  EXPECT_TRUE(m.colors.empty());
  EXPECT_TRUE(m.confidence.empty());
  EXPECT_TRUE(m.texcoords.empty());
  EXPECT_TRUE(m.texcoord_faces.empty());
  EXPECT_TRUE(m.texcoord_lines.empty());
}

TEST(Momentum_Mesh, UpdateNormals) {
  // Construct simple mesh with 3 vertices and 1 face
  Mesh m;
  m.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}};
  m.faces = {{0, 1, 2}};

  // Compute mesh normals
  m.updateNormals();
  EXPECT_EQ(m.normals.size(), 3);

  // Ensure all normals are (0, 0, 1)
  const float eps = 1e-5;
  for (int i = 0; i < m.normals.size(); ++i) {
    EXPECT_NEAR(m.normals[i][0], 0.f, eps);
    EXPECT_NEAR(m.normals[i][1], 0.f, eps);
    EXPECT_NEAR(m.normals[i][2], 1.f, eps);
  }
}

TEST(Momentum_Mesh, UpdateNormalsMt) {
  // Construct simple mesh with 3 vertices and 1 face
  Mesh m;
  m.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}};
  m.faces = {{0, 1, 2}};

  // Compute mesh normals
  m.updateNormalsMt();
  EXPECT_EQ(m.normals.size(), 3);

  // Ensure all normals are (0, 0, 1)
  const float eps = 1e-5;
  for (int i = 0; i < m.normals.size(); ++i) {
    EXPECT_NEAR(m.normals[i][0], 0.f, eps);
    EXPECT_NEAR(m.normals[i][1], 0.f, eps);
    EXPECT_NEAR(m.normals[i][2], 1.f, eps);
  }
}

TEST(Momentum_Mesh, Reset) {
  // Construct non-empty mesh
  Mesh m = {
      {{1.0, 1.0, 1.0}}, // vertices
      {{1.0, 1.0, 1.0}}, // normals
      {{0, 1, 2}}, // faces
      {{0, 1, 2}}, // lines
      {{255, 0, 0}}, // colors
      {1.0f}, // confidence
      {{1.0, 1.0}}, // texcoords
      {{1, 1, 1}}, // texcoord_faces
      {{0, 1, 2}} // texcoord_lines
  };
  // Reset mesh and ensure it is empty
  m.reset();
  EXPECT_TRUE(m.vertices.empty());
  EXPECT_TRUE(m.normals.empty());
  EXPECT_TRUE(m.faces.empty());
  EXPECT_TRUE(m.lines.empty());
  EXPECT_TRUE(m.colors.empty());
  EXPECT_TRUE(m.confidence.empty());
  EXPECT_TRUE(m.texcoords.empty());
  EXPECT_TRUE(m.texcoord_faces.empty());
  EXPECT_TRUE(m.texcoord_lines.empty());
}
