/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/urdf/urdf_io.h"
#include "momentum/io/urdf/urdf_mesh_io.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character/character_helpers_gtest.h"
#include "momentum/test/io/io_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <fstream>

using namespace momentum;

namespace {

std::optional<filesystem::path> getTestFilePath(const std::string& filename) {
  auto envVar = GetEnvVar("TEST_MOMENTUM_MODELS_PATH");
  if (!envVar.has_value()) {
    return std::nullopt;
  }
  return filesystem::path(envVar.value()) / filename;
}

TEST(IoUrdfTest, LoadCharacter) {
  auto urdfPath = getTestFilePath("character.urdf");
  if (!urdfPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_MOMENTUM_MODELS_PATH' is not set.";
  }

  const auto& character = loadUrdfCharacter(*urdfPath);
  const auto& skeleton = character.skeleton;
  EXPECT_EQ(skeleton.joints.size(), 45);
  EXPECT_EQ(skeleton.joints[0].name, "b_root");
  EXPECT_EQ(skeleton.joints[0].parent, kInvalidIndex);
  EXPECT_TRUE(skeleton.joints[0].preRotation.isApprox(Quaternionf::Identity()));
  EXPECT_TRUE(skeleton.joints[0].translationOffset.isApprox(Vector3f::Zero()));

  const auto& parameterTransform = character.parameterTransform;
  const auto totalDofs = parameterTransform.transform.cols();
  EXPECT_EQ(totalDofs, 6 + 34); // 6 global parameters (root) + 34 joint parameters
  const size_t numJoints = skeleton.joints.size();
  const size_t numJointParameters = numJoints * kParametersPerJoint;
  EXPECT_EQ(parameterTransform.name.size(), totalDofs);
  EXPECT_EQ(parameterTransform.offsets.size(), (numJointParameters));
  EXPECT_EQ(parameterTransform.transform.rows(), numJointParameters);
  EXPECT_EQ(parameterTransform.transform.cols(), totalDofs);

  const auto& parameterLimits = character.parameterLimits;
  EXPECT_EQ(parameterLimits.size(), 34); // the number of revolute and prismatic joints
  EXPECT_EQ(parameterLimits[0].type, MinMax);
  EXPECT_EQ(parameterLimits[0].data.minMax.parameterIndex, 6); // the first joint parameter
  EXPECT_FLOAT_EQ(parameterLimits[0].data.minMax.limits[0], -2.095);
  EXPECT_FLOAT_EQ(parameterLimits[0].data.minMax.limits[1], 0.785);
}

// Test that the existing character.urdf (which has no visual meshes) still works
// and produces a character without a mesh.
TEST(IoUrdfTest, LoadCharacterNoMesh) {
  auto urdfPath = getTestFilePath("character.urdf");
  if (!urdfPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_MOMENTUM_MODELS_PATH' is not set.";
  }

  const auto& character = loadUrdfCharacter(*urdfPath);
  EXPECT_EQ(character.mesh, nullptr);
  EXPECT_EQ(character.skinWeights, nullptr);
}

TEST(IoUrdfTest, GenerateBoxMesh) {
  const auto mesh = generateBoxMesh(2.0f, 3.0f, 4.0f);
  EXPECT_EQ(mesh.vertices.size(), 8);
  EXPECT_EQ(mesh.faces.size(), 12);
  EXPECT_EQ(mesh.normals.size(), 8);

  // Check that all vertices are within the expected bounds
  for (const auto& v : mesh.vertices) {
    EXPECT_NEAR(v.x(), 0.0f, 1.0f + 1e-6f);
    EXPECT_NEAR(v.y(), 0.0f, 1.5f + 1e-6f);
    EXPECT_NEAR(v.z(), 0.0f, 2.0f + 1e-6f);
  }
}

TEST(IoUrdfTest, GenerateCylinderMesh) {
  const int segments = 16;
  const auto mesh = generateCylinderMesh(1.0f, 2.0f, segments);

  // 2 center vertices + 2 * segments ring vertices
  EXPECT_EQ(mesh.vertices.size(), 2 + 2 * segments);
  // segments bottom cap + segments top cap + 2 * segments side
  EXPECT_EQ(mesh.faces.size(), 4 * segments);
  EXPECT_EQ(mesh.normals.size(), mesh.vertices.size());
}

TEST(IoUrdfTest, GenerateSphereMesh) {
  const int lat = 8;
  const int lon = 16;
  const auto mesh = generateSphereMesh(1.0f, lat, lon);

  // 2 poles + (lat - 1) * lon ring vertices
  EXPECT_EQ(mesh.vertices.size(), 2 + (lat - 1) * lon);
  // lon top cap + lon bottom cap + (lat - 2) * lon * 2 middle
  EXPECT_EQ(mesh.faces.size(), 2 * lon + (lat - 2) * lon * 2);
  EXPECT_EQ(mesh.normals.size(), mesh.vertices.size());
}

TEST(IoUrdfTest, LoadUrdfWithPrimitiveVisuals) {
  // Create a simple URDF with primitive visual geometries
  const std::string urdfContent = R"(
    <?xml version="1.0"?>
    <robot name="test_robot">
      <link name="base_link">
        <visual>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <geometry>
            <box size="0.1 0.2 0.3"/>
          </geometry>
        </visual>
      </link>
      <link name="child_link">
        <visual>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <geometry>
            <cylinder radius="0.05" length="0.2"/>
          </geometry>
        </visual>
      </link>
      <joint name="joint1" type="revolute">
        <parent link="base_link"/>
        <child link="child_link"/>
        <origin xyz="0.1 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
      </joint>
    </robot>
  )";

  const auto bytes = std::as_bytes(std::span(urdfContent));
  const auto character = loadUrdfCharacter(bytes);

  // Should have 2 joints
  EXPECT_EQ(character.skeleton.joints.size(), 2);

  // Should have a mesh (from the primitive visuals)
  ASSERT_NE(character.mesh, nullptr);
  EXPECT_GT(character.mesh->vertices.size(), 0);
  EXPECT_GT(character.mesh->faces.size(), 0);
  EXPECT_EQ(character.mesh->normals.size(), character.mesh->vertices.size());

  // Should have skin weights
  ASSERT_NE(character.skinWeights, nullptr);
  EXPECT_EQ(character.skinWeights->index.rows(), character.mesh->vertices.size());
  EXPECT_EQ(character.skinWeights->weight.rows(), character.mesh->vertices.size());

  // Box has 8 vertices, cylinder (32 segments) has 66 vertices
  const size_t boxVertices = 8;
  const size_t cylinderVertices = 2 + 2 * 32; // 2 centers + 2 rings of 32
  EXPECT_EQ(character.mesh->vertices.size(), boxVertices + cylinderVertices);

  // Verify skin weights: first 8 vertices should be bound to joint 0 (base_link),
  // remaining to joint 1 (child_link)
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(boxVertices); ++i) {
    EXPECT_EQ(character.skinWeights->index(i, 0), 0);
    EXPECT_FLOAT_EQ(character.skinWeights->weight(i, 0), 1.0f);
  }
  for (auto i = static_cast<Eigen::Index>(boxVertices); i < character.skinWeights->index.rows();
       ++i) {
    EXPECT_EQ(character.skinWeights->index(i, 0), 1);
    EXPECT_FLOAT_EQ(character.skinWeights->weight(i, 0), 1.0f);
  }
}

TEST(IoUrdfTest, LoadUrdfWithSphereVisual) {
  const std::string urdfContent = R"(
    <?xml version="1.0"?>
    <robot name="test_robot">
      <link name="base_link">
        <visual>
          <origin xyz="0 0 0.5" rpy="0 0 0"/>
          <geometry>
            <sphere radius="0.1"/>
          </geometry>
        </visual>
      </link>
    </robot>
  )";

  const auto bytes = std::as_bytes(std::span(urdfContent));
  const auto character = loadUrdfCharacter(bytes);

  EXPECT_EQ(character.skeleton.joints.size(), 1);
  ASSERT_NE(character.mesh, nullptr);
  EXPECT_GT(character.mesh->vertices.size(), 0);
  EXPECT_GT(character.mesh->faces.size(), 0);

  // Sphere with default 16 lat, 32 lon: 2 + 15*32 = 482 vertices
  EXPECT_EQ(character.mesh->vertices.size(), 2 + 15 * 32);
}

TEST(IoUrdfTest, LoadStlMesh) {
  auto tempDir = temporaryDirectory("urdf_stl_test");

  // Write a minimal binary STL file with 1 triangle
  const auto stlPath = tempDir.path() / "test.stl";
  {
    std::ofstream file(stlPath, std::ios::binary);
    // 80-byte header (zeros)
    std::array<char, 80> header = {};
    file.write(header.data(), 80);
    // Triangle count
    uint32_t numTriangles = 1;
    file.write(reinterpret_cast<const char*>(&numTriangles), sizeof(uint32_t));
    // Normal
    std::array<float, 3> normal = {0.0f, 0.0f, 1.0f};
    file.write(reinterpret_cast<const char*>(normal.data()), sizeof(float) * 3);
    // Vertex 1
    std::array<float, 3> v1 = {0.0f, 0.0f, 0.0f};
    file.write(reinterpret_cast<const char*>(v1.data()), sizeof(float) * 3);
    // Vertex 2
    std::array<float, 3> v2 = {1.0f, 0.0f, 0.0f};
    file.write(reinterpret_cast<const char*>(v2.data()), sizeof(float) * 3);
    // Vertex 3
    std::array<float, 3> v3 = {0.0f, 1.0f, 0.0f};
    file.write(reinterpret_cast<const char*>(v3.data()), sizeof(float) * 3);
    // Attribute byte count
    uint16_t attrByteCount = 0;
    file.write(reinterpret_cast<const char*>(&attrByteCount), sizeof(uint16_t));
  }

  const auto meshOpt = loadStlMesh(stlPath);
  ASSERT_TRUE(meshOpt.has_value());
  EXPECT_EQ(meshOpt->vertices.size(), 3);
  EXPECT_EQ(meshOpt->faces.size(), 1);
  EXPECT_EQ(meshOpt->normals.size(), 3);
}

TEST(IoUrdfTest, LoadObjMesh) {
  auto tempDir = temporaryDirectory("urdf_obj_test");

  // Write a minimal OBJ file with 1 triangle
  const auto objPath = tempDir.path() / "test.obj";
  {
    std::ofstream file(objPath);
    file << "# Simple triangle\n";
    file << "v 0.0 0.0 0.0\n";
    file << "v 1.0 0.0 0.0\n";
    file << "v 0.0 1.0 0.0\n";
    file << "f 1 2 3\n";
  }

  const auto meshOpt = loadObjMesh(objPath);
  ASSERT_TRUE(meshOpt.has_value());
  EXPECT_EQ(meshOpt->vertices.size(), 3);
  EXPECT_EQ(meshOpt->faces.size(), 1);
  EXPECT_EQ(meshOpt->normals.size(), 3);

  // Verify vertex positions
  EXPECT_TRUE(meshOpt->vertices[0].isApprox(Eigen::Vector3f(0, 0, 0)));
  EXPECT_TRUE(meshOpt->vertices[1].isApprox(Eigen::Vector3f(1, 0, 0)));
  EXPECT_TRUE(meshOpt->vertices[2].isApprox(Eigen::Vector3f(0, 1, 0)));
}

TEST(IoUrdfTest, LoadObjMeshWithSlashNotation) {
  auto tempDir = temporaryDirectory("urdf_obj_slash_test");

  const auto objPath = tempDir.path() / "test.obj";
  {
    std::ofstream file(objPath);
    file << "v 0 0 0\n";
    file << "v 1 0 0\n";
    file << "v 0 1 0\n";
    file << "v 1 1 0\n";
    file << "vn 0 0 1\n";
    file << "f 1//1 2//1 3//1\n";
    file << "f 2//1 4//1 3//1\n";
  }

  const auto meshOpt = loadObjMesh(objPath);
  ASSERT_TRUE(meshOpt.has_value());
  EXPECT_EQ(meshOpt->vertices.size(), 4);
  EXPECT_EQ(meshOpt->faces.size(), 2);
}

TEST(IoUrdfTest, LoadUrdfWithStlMesh) {
  auto tempDir = temporaryDirectory("urdf_stl_load_test");

  // Write a binary STL triangle
  const auto stlPath = tempDir.path() / "mesh.stl";
  {
    std::ofstream file(stlPath, std::ios::binary);
    std::array<char, 80> header = {};
    file.write(header.data(), 80);
    uint32_t numTriangles = 1;
    file.write(reinterpret_cast<const char*>(&numTriangles), sizeof(uint32_t));
    std::array<float, 3> normal = {0, 0, 1};
    file.write(reinterpret_cast<const char*>(normal.data()), sizeof(float) * 3);
    std::array<float, 3> v1 = {0, 0, 0};
    file.write(reinterpret_cast<const char*>(v1.data()), sizeof(float) * 3);
    std::array<float, 3> v2 = {1, 0, 0};
    file.write(reinterpret_cast<const char*>(v2.data()), sizeof(float) * 3);
    std::array<float, 3> v3 = {0, 1, 0};
    file.write(reinterpret_cast<const char*>(v3.data()), sizeof(float) * 3);
    uint16_t attr = 0;
    file.write(reinterpret_cast<const char*>(&attr), sizeof(uint16_t));
  }

  // Write a URDF that references the STL file
  const auto urdfPath = tempDir.path() / "robot.urdf";
  {
    std::ofstream file(urdfPath);
    file << R"(<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="mesh.stl"/>
      </geometry>
    </visual>
  </link>
</robot>
)";
  }

  const auto character = loadUrdfCharacter(urdfPath);
  EXPECT_EQ(character.skeleton.joints.size(), 1);
  ASSERT_NE(character.mesh, nullptr);
  EXPECT_EQ(character.mesh->vertices.size(), 3);
  EXPECT_EQ(character.mesh->faces.size(), 1);
  ASSERT_NE(character.skinWeights, nullptr);
  EXPECT_EQ(character.skinWeights->index.rows(), 3);
}

TEST(IoUrdfTest, LoadUrdfWithMeshScale) {
  auto tempDir = temporaryDirectory("urdf_mesh_scale_test");

  // Write a simple OBJ
  const auto objPath = tempDir.path() / "mesh.obj";
  {
    std::ofstream file(objPath);
    file << "v 1 0 0\n";
    file << "v 0 1 0\n";
    file << "v 0 0 1\n";
    file << "f 1 2 3\n";
  }

  // URDF with scale
  const auto urdfPath = tempDir.path() / "robot.urdf";
  {
    std::ofstream file(urdfPath);
    file << R"(<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="mesh.obj" scale="2 3 4"/>
      </geometry>
    </visual>
  </link>
</robot>
)";
  }

  const auto character = loadUrdfCharacter(urdfPath);
  ASSERT_NE(character.mesh, nullptr);
  EXPECT_EQ(character.mesh->vertices.size(), 3);

  // Vertices should be scaled (from meters to cm), then multiplied by scale.
  // Original: (1,0,0) -> scaled: (2,0,0) -> in cm: (200,0,0)
  // Original: (0,1,0) -> scaled: (0,3,0) -> in cm: (0,300,0)
  // Original: (0,0,1) -> scaled: (0,0,4) -> in cm: (0,0,400)
  // (These are then transformed to bind-pose world space, which for a root joint
  // with identity transform is the same.)
  EXPECT_NEAR(character.mesh->vertices[0].x(), 200.0f, 1e-3f);
  EXPECT_NEAR(character.mesh->vertices[1].y(), 300.0f, 1e-3f);
  EXPECT_NEAR(character.mesh->vertices[2].z(), 400.0f, 1e-3f);
}

TEST(IoUrdfTest, LoadUrdfBytesWithMeshBasePath) {
  auto tempDir = temporaryDirectory("urdf_bytes_mesh_test");

  // Write a simple OBJ
  const auto objPath = tempDir.path() / "mesh.obj";
  {
    std::ofstream file(objPath);
    file << "v 0 0 0\n";
    file << "v 1 0 0\n";
    file << "v 0 1 0\n";
    file << "f 1 2 3\n";
  }

  const std::string urdfContent = R"(<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="mesh.obj"/>
      </geometry>
    </visual>
  </link>
</robot>
)";

  // Without base path — mesh should not be loaded
  {
    const auto bytes = std::as_bytes(std::span(urdfContent));
    const auto character = loadUrdfCharacter(bytes);
    EXPECT_EQ(character.mesh, nullptr);
  }

  // With base path — mesh should be loaded
  {
    const auto bytes = std::as_bytes(std::span(urdfContent));
    const auto character = loadUrdfCharacter(bytes, tempDir.path());
    ASSERT_NE(character.mesh, nullptr);
    EXPECT_EQ(character.mesh->vertices.size(), 3);
    EXPECT_EQ(character.mesh->faces.size(), 1);
  }
}

TEST(IoUrdfTest, ResolveMeshPath) {
  auto tempDir = temporaryDirectory("urdf_resolve_path_test");

  // Create a subdirectory and mesh file
  const auto meshDir = tempDir.path() / "meshes";
  filesystem::create_directory(meshDir);
  const auto meshFile = meshDir / "part.stl";
  {
    std::ofstream file(meshFile);
    file << "placeholder";
  }

  // Relative path
  EXPECT_TRUE(resolveMeshPath("meshes/part.stl", tempDir.path()).has_value());

  // Absolute path
  EXPECT_TRUE(resolveMeshPath(meshFile.string(), tempDir.path()).has_value());

  // Non-existent file
  EXPECT_FALSE(resolveMeshPath("nonexistent.stl", tempDir.path()).has_value());

  // package:// URI — resolve relative to URDF dir
  EXPECT_TRUE(resolveMeshPath("package://robot/meshes/part.stl", tempDir.path()).has_value());
}

TEST(IoUrdfTest, LoadUrdfWithMultipleVisuals) {
  const std::string urdfContent = R"(
    <?xml version="1.0"?>
    <robot name="test_robot">
      <link name="base_link">
        <visual>
          <geometry>
            <box size="0.1 0.1 0.1"/>
          </geometry>
        </visual>
        <visual>
          <geometry>
            <sphere radius="0.05"/>
          </geometry>
        </visual>
      </link>
    </robot>
  )";

  const auto bytes = std::as_bytes(std::span(urdfContent));
  const auto character = loadUrdfCharacter(bytes);

  ASSERT_NE(character.mesh, nullptr);

  // Box: 8 vertices, Sphere (16 lat, 32 lon): 482 vertices
  const size_t expectedVertices = 8 + (2 + 15 * 32);
  EXPECT_EQ(character.mesh->vertices.size(), expectedVertices);

  // All vertices should be bound to joint 0
  ASSERT_NE(character.skinWeights, nullptr);
  for (Eigen::Index i = 0; i < character.skinWeights->index.rows(); ++i) {
    EXPECT_EQ(character.skinWeights->index(i, 0), 0);
    EXPECT_FLOAT_EQ(character.skinWeights->weight(i, 0), 1.0f);
  }
}

TEST(IoUrdfTest, FloatingJoint) {
  const std::string urdfContent = R"(
    <?xml version="1.0"?>
    <robot name="floating_robot">
      <link name="world"/>
      <link name="base_link"/>
      <link name="child_link"/>
      <joint name="floating_base" type="floating">
        <parent link="world"/>
        <child link="base_link"/>
      </joint>
      <joint name="arm_joint" type="revolute">
        <parent link="base_link"/>
        <child link="child_link"/>
        <origin xyz="0 0 0.5" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
      </joint>
    </robot>
  )";

  const auto bytes = std::as_bytes(std::span(urdfContent));
  const auto character = loadUrdfCharacter(bytes);

  // 2 joints: base_link (floating) and child_link (revolute)
  EXPECT_EQ(character.skeleton.joints.size(), 2);

  // 7 model parameters: 6 floating + 1 revolute
  const auto& pt = character.parameterTransform;
  EXPECT_EQ(pt.name.size(), 7);

  // Floating joint parameter names use the URDF joint name
  EXPECT_EQ(pt.name[0], "floating_base_tx");
  EXPECT_EQ(pt.name[1], "floating_base_ty");
  EXPECT_EQ(pt.name[2], "floating_base_tz");
  EXPECT_EQ(pt.name[3], "floating_base_rx");
  EXPECT_EQ(pt.name[4], "floating_base_ry");
  EXPECT_EQ(pt.name[5], "floating_base_rz");
  EXPECT_EQ(pt.name[6], "arm_joint");

  // Translation parameters have 10x scaling factor
  constexpr float kTranslationScale = 10.0f;
  EXPECT_FLOAT_EQ(pt.transform.coeff(0 * kParametersPerJoint + 0, 0), kTranslationScale); // tx
  EXPECT_FLOAT_EQ(pt.transform.coeff(0 * kParametersPerJoint + 1, 1), kTranslationScale); // ty
  EXPECT_FLOAT_EQ(pt.transform.coeff(0 * kParametersPerJoint + 2, 2), kTranslationScale); // tz

  // Rotation parameters have 1x scaling
  EXPECT_FLOAT_EQ(pt.transform.coeff(0 * kParametersPerJoint + 3, 3), 1.0f); // rx
  EXPECT_FLOAT_EQ(pt.transform.coeff(0 * kParametersPerJoint + 4, 4), 1.0f); // ry
  EXPECT_FLOAT_EQ(pt.transform.coeff(0 * kParametersPerJoint + 5, 5), 1.0f); // rz

  // Only 1 parameter limit (the revolute joint)
  EXPECT_EQ(character.parameterLimits.size(), 1);
}

TEST(IoUrdfTest, RobotName) {
  const std::string urdfContent = R"(
    <?xml version="1.0"?>
    <robot name="my_robot">
      <link name="base_link"/>
    </robot>
  )";

  const auto bytes = std::as_bytes(std::span(urdfContent));
  const auto character = loadUrdfCharacter(bytes);
  EXPECT_EQ(character.name, "my_robot");
}

TEST(IoUrdfTest, MimicJoint) {
  // Create a URDF with a mimic joint: joint2 mimics joint1 with multiplier=2, offset=0.1
  const std::string urdfContent = R"(
    <?xml version="1.0"?>
    <robot name="mimic_robot">
      <link name="base_link"/>
      <link name="link1"/>
      <link name="link2"/>
      <joint name="joint1" type="revolute">
        <parent link="base_link"/>
        <child link="link1"/>
        <origin xyz="0.1 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
      </joint>
      <joint name="joint2" type="revolute">
        <parent link="base_link"/>
        <child link="link2"/>
        <origin xyz="-0.1 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
        <mimic joint="joint1" multiplier="2" offset="0.1"/>
      </joint>
    </robot>
  )";

  const auto bytes = std::as_bytes(std::span(urdfContent));
  const auto character = loadUrdfCharacter(bytes);

  // Should have 3 joints (base_link, link1, link2)
  EXPECT_EQ(character.skeleton.joints.size(), 3);

  // Only 1 model parameter (joint1's rz) — joint2 is a mimic, no independent DOF
  const auto& pt = character.parameterTransform;
  EXPECT_EQ(pt.name.size(), 1);
  EXPECT_EQ(pt.name[0], "joint1");

  // Only 1 parameter limit (joint1's limits) — mimic joints don't get limits
  EXPECT_EQ(character.parameterLimits.size(), 1);

  // The mimic joint's offset should be stored in parameterTransform.offsets.
  // joint2 is skeleton joint index 2, its rz is at joint param index 2*7+5 = 19
  EXPECT_FLOAT_EQ(pt.offsets[2 * kParametersPerJoint + 5], 0.1f);

  // The transform matrix should have an entry mapping model param 0 to joint2's rz
  // with coefficient = sign * multiplier = 1 * 2 = 2
  EXPECT_FLOAT_EQ(pt.transform.coeff(2 * kParametersPerJoint + 5, 0), 2.0f);
}

TEST(IoUrdfTest, MaterialColors) {
  const std::string urdfContent = R"(
    <?xml version="1.0"?>
    <robot name="colored_robot">
      <link name="base_link">
        <visual>
          <geometry>
            <box size="0.1 0.1 0.1"/>
          </geometry>
          <material name="red">
            <color rgba="1.0 0.0 0.0 1.0"/>
          </material>
        </visual>
      </link>
    </robot>
  )";

  const auto bytes = std::as_bytes(std::span(urdfContent));
  const auto character = loadUrdfCharacter(bytes);

  ASSERT_NE(character.mesh, nullptr);
  EXPECT_EQ(character.mesh->vertices.size(), 8); // box

  // Should have per-vertex colors matching the red material
  EXPECT_EQ(character.mesh->colors.size(), 8);
  for (const auto& c : character.mesh->colors) {
    EXPECT_EQ(c.x(), 255); // red
    EXPECT_EQ(c.y(), 0); // green
    EXPECT_EQ(c.z(), 0); // blue
  }
}

TEST(IoUrdfTest, NoMaterialColorsEmpty) {
  // When no visuals have material colors, colors vector should be empty
  const std::string urdfContent = R"(
    <?xml version="1.0"?>
    <robot name="no_color_robot">
      <link name="base_link">
        <visual>
          <geometry>
            <box size="0.1 0.1 0.1"/>
          </geometry>
        </visual>
      </link>
    </robot>
  )";

  const auto bytes = std::as_bytes(std::span(urdfContent));
  const auto character = loadUrdfCharacter(bytes);

  ASSERT_NE(character.mesh, nullptr);
  EXPECT_TRUE(character.mesh->colors.empty());
}

} // namespace
