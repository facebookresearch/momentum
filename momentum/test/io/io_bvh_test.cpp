/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/io/bvh/bvh_io.h>
#include <momentum/test/io/io_helpers.h>

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>

using namespace momentum;

namespace {

std::optional<filesystem::path> getTestResourcePath(const std::string& filename) {
  auto envVar = GetEnvVar("TEST_RESOURCES_PATH");
  if (!envVar.has_value()) {
    return std::nullopt;
  }
  return filesystem::path(envVar.value()) / filename;
}

TEST(IoBvhTest, LoadCharacter) {
  auto bvhPath = getTestResourcePath("bvh/simple.bvh");
  if (!bvhPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_RESOURCES_PATH' is not set.";
  }

  const auto character = loadBvhCharacter(*bvhPath);
  const auto& skeleton = character.skeleton;

  // Verify joint count: Hips, Chest, LeftShoulder, RightShoulder (End Sites are excluded)
  EXPECT_EQ(skeleton.joints.size(), 4);

  // Verify joint names
  EXPECT_EQ(skeleton.joints[0].name, "Hips");
  EXPECT_EQ(skeleton.joints[1].name, "Chest");
  EXPECT_EQ(skeleton.joints[2].name, "LeftShoulder");
  EXPECT_EQ(skeleton.joints[3].name, "RightShoulder");

  // Verify hierarchy
  EXPECT_EQ(skeleton.joints[0].parent, kInvalidIndex); // Hips is root
  EXPECT_EQ(skeleton.joints[1].parent, 0u); // Chest -> Hips
  EXPECT_EQ(skeleton.joints[2].parent, 1u); // LeftShoulder -> Chest
  EXPECT_EQ(skeleton.joints[3].parent, 1u); // RightShoulder -> Chest

  // Verify offsets
  EXPECT_TRUE(skeleton.joints[0].translationOffset.isApprox(Vector3f(0.0f, 0.0f, 0.0f)));
  EXPECT_TRUE(skeleton.joints[1].translationOffset.isApprox(Vector3f(0.0f, 5.21f, 0.0f)));
  EXPECT_TRUE(skeleton.joints[2].translationOffset.isApprox(Vector3f(3.0f, 5.0f, 0.0f)));
  EXPECT_TRUE(skeleton.joints[3].translationOffset.isApprox(Vector3f(-3.0f, 5.0f, 0.0f)));

  // Verify identity preRotation
  for (const auto& joint : skeleton.joints) {
    EXPECT_TRUE(joint.preRotation.isApprox(Quaternionf::Identity()));
  }

  // Verify identity ParameterTransform
  const auto& pt = character.parameterTransform;
  const size_t numJoints = skeleton.joints.size();
  const size_t numJointParams = numJoints * kParametersPerJoint;
  EXPECT_EQ(pt.name.size(), numJointParams);
  EXPECT_EQ(pt.transform.rows(), static_cast<Eigen::Index>(numJointParams));
  EXPECT_EQ(pt.transform.cols(), static_cast<Eigen::Index>(numJointParams));

  // Verify parameter names follow the identity pattern
  EXPECT_EQ(pt.name[0], "Hips_tx");
  EXPECT_EQ(pt.name[1], "Hips_ty");
  EXPECT_EQ(pt.name[2], "Hips_tz");
  EXPECT_EQ(pt.name[3], "Hips_rx");
  EXPECT_EQ(pt.name[4], "Hips_ry");
  EXPECT_EQ(pt.name[5], "Hips_rz");
  EXPECT_EQ(pt.name[6], "Hips_sc");
}

TEST(IoBvhTest, LoadCharacterWithMotion) {
  auto bvhPath = getTestResourcePath("bvh/simple.bvh");
  if (!bvhPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_RESOURCES_PATH' is not set.";
  }

  const auto [character, motionParams, fps] = loadBvhCharacterWithMotion(*bvhPath);
  const auto& [paramNames, motion] = motionParams;

  // Verify FPS (1/0.033333 ~ 30)
  EXPECT_NEAR(fps, 30.0f, 0.1f);

  // Verify motion dimensions: 4 joints * 7 params = 28 rows, 3 frames
  EXPECT_EQ(motion.rows(), 28);
  EXPECT_EQ(motion.cols(), 3);

  // Verify parameter names match the character
  EXPECT_EQ(paramNames.size(), character.parameterTransform.name.size());
  for (size_t i = 0; i < paramNames.size(); ++i) {
    EXPECT_EQ(paramNames[i], character.parameterTransform.name[i]);
  }

  // Frame 0: all zeros (rest pose)
  for (Eigen::Index i = 0; i < motion.rows(); ++i) {
    EXPECT_NEAR(motion(i, 0), 0.0f, 1e-6f);
  }

  // Frame 1: verify root translation (Hips)
  // Hips: Xposition=1.0, Yposition=2.0, Zposition=3.0
  EXPECT_NEAR(motion(0, 1), 1.0f, 1e-5f); // Hips_tx
  EXPECT_NEAR(motion(1, 1), 2.0f, 1e-5f); // Hips_ty
  EXPECT_NEAR(motion(2, 1), 3.0f, 1e-5f); // Hips_tz

  // Verify that rotation values are in radians (not degrees)
  // For Frame 1, Hips has BVH channels: Zrot=10, Xrot=20, Yrot=30 degrees
  // The rotation values should be non-zero and in a reasonable radian range
  const float hipsRx = motion(3, 1); // Hips_rx
  const float hipsRy = motion(4, 1); // Hips_ry
  const float hipsRz = motion(5, 1); // Hips_rz
  EXPECT_NE(hipsRx, 0.0f);
  EXPECT_NE(hipsRy, 0.0f);
  EXPECT_NE(hipsRz, 0.0f);
  // All rotation values should be less than pi (reasonable for 10-30 degree inputs)
  EXPECT_LT(std::abs(hipsRx), pi<float>());
  EXPECT_LT(std::abs(hipsRy), pi<float>());
  EXPECT_LT(std::abs(hipsRz), pi<float>());

  // Verify that the BVH rotation can be reconstructed
  // BVH channels for Hips: Zrotation=10deg, Xrotation=20deg, Yrotation=30deg
  // This means intrinsic ZXY: R = Rz(10deg) * Rx(20deg) * Ry(30deg)
  const float deg2rad = pi<float>() / 180.0f;
  Quaternionf expectedQ = Quaternionf(Eigen::AngleAxisf(10.0f * deg2rad, Vector3f::UnitZ())) *
      Quaternionf(Eigen::AngleAxisf(20.0f * deg2rad, Vector3f::UnitX())) *
      Quaternionf(Eigen::AngleAxisf(30.0f * deg2rad, Vector3f::UnitY()));

  // Reconstruct rotation from Momentum's ZYX convention
  Quaternionf momentumQ = Quaternionf(Eigen::AngleAxisf(hipsRz, Vector3f::UnitZ())) *
      Quaternionf(Eigen::AngleAxisf(hipsRy, Vector3f::UnitY())) *
      Quaternionf(Eigen::AngleAxisf(hipsRx, Vector3f::UnitX()));

  // The two rotations should be equivalent
  EXPECT_TRUE(expectedQ.toRotationMatrix().isApprox(momentumQ.toRotationMatrix(), 1e-4f));

  // Verify scale parameters are zero
  EXPECT_NEAR(motion(6, 1), 0.0f, 1e-6f); // Hips_sc = 0
  EXPECT_NEAR(motion(13, 1), 0.0f, 1e-6f); // Chest_sc = 0
}

TEST(IoBvhTest, LoadFromBuffer) {
  auto bvhPath = getTestResourcePath("bvh/simple.bvh");
  if (!bvhPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_RESOURCES_PATH' is not set.";
  }

  // Read the file into a buffer
  std::ifstream file(*bvhPath, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(file.is_open());
  const auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<std::byte> buffer(fileSize);
  file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

  // Load from buffer
  const auto character = loadBvhCharacter(std::span<const std::byte>(buffer));
  EXPECT_EQ(character.skeleton.joints.size(), 4);

  // Load with motion from buffer
  const auto [charWithMotion, motionParams, fps] =
      loadBvhCharacterWithMotion(std::span<const std::byte>(buffer));
  const auto& [paramNames, motion] = motionParams;
  EXPECT_EQ(charWithMotion.skeleton.joints.size(), 4);
  EXPECT_EQ(motion.cols(), 3);
  EXPECT_NEAR(fps, 30.0f, 0.1f);
}

TEST(IoBvhTest, SaveAndReloadCharacter) {
  auto bvhPath = getTestResourcePath("bvh/simple.bvh");
  if (!bvhPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_RESOURCES_PATH' is not set.";
  }

  // Load the original character
  const auto original = loadBvhCharacter(*bvhPath);

  // Save to a temporary file and reload
  const auto tmpFile = temporaryFile("test_save_character", "bvh");
  saveBvhCharacter(tmpFile.path(), original);
  const auto reloaded = loadBvhCharacter(tmpFile.path());

  // Verify joint count and names survive round-trip
  ASSERT_EQ(reloaded.skeleton.joints.size(), original.skeleton.joints.size());
  for (size_t i = 0; i < original.skeleton.joints.size(); ++i) {
    EXPECT_EQ(reloaded.skeleton.joints[i].name, original.skeleton.joints[i].name);
    EXPECT_EQ(reloaded.skeleton.joints[i].parent, original.skeleton.joints[i].parent);
  }

  // Verify offsets survive round-trip
  for (size_t i = 0; i < original.skeleton.joints.size(); ++i) {
    EXPECT_TRUE(reloaded.skeleton.joints[i].translationOffset.isApprox(
        original.skeleton.joints[i].translationOffset, 1e-3f));
  }
}

TEST(IoBvhTest, SaveAndReloadCharacterWithMotion) {
  auto bvhPath = getTestResourcePath("bvh/simple.bvh");
  if (!bvhPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_RESOURCES_PATH' is not set.";
  }

  // Load the original character with motion
  const auto [origCharacter, origMotionParams, origFps] = loadBvhCharacterWithMotion(*bvhPath);
  const auto& [origParamNames, origMotion] = origMotionParams;

  // Save to a temporary file and reload
  const auto tmpFile = temporaryFile("test_save_motion", "bvh");
  saveBvhCharacterWithMotion(tmpFile.path(), origCharacter, origMotionParams, origFps);
  const auto [reCharacter, reMotionParams, reFps] = loadBvhCharacterWithMotion(tmpFile.path());
  const auto& [reParamNames, reMotion] = reMotionParams;

  // Verify FPS round-trip
  EXPECT_NEAR(reFps, origFps, 0.5f);

  // Verify motion dimensions
  EXPECT_EQ(reMotion.rows(), origMotion.rows());
  EXPECT_EQ(reMotion.cols(), origMotion.cols());

  // Verify motion data round-trip: translations should be very close
  for (size_t jointIdx = 0; jointIdx < origCharacter.skeleton.joints.size(); ++jointIdx) {
    const auto base = static_cast<Eigen::Index>(jointIdx * kParametersPerJoint);
    for (Eigen::Index frameIdx = 0; frameIdx < origMotion.cols(); ++frameIdx) {
      // Translation params
      EXPECT_NEAR(reMotion(base + TX, frameIdx), origMotion(base + TX, frameIdx), 1e-3f);
      EXPECT_NEAR(reMotion(base + TY, frameIdx), origMotion(base + TY, frameIdx), 1e-3f);
      EXPECT_NEAR(reMotion(base + TZ, frameIdx), origMotion(base + TZ, frameIdx), 1e-3f);

      // Rotation: compare as rotation matrices since Euler angles can differ
      // but represent the same rotation
      const float origRx = origMotion(base + RX, frameIdx);
      const float origRy = origMotion(base + RY, frameIdx);
      const float origRz = origMotion(base + RZ, frameIdx);
      const Quaternionf origQ = Quaternionf(Eigen::AngleAxisf(origRz, Vector3f::UnitZ())) *
          Quaternionf(Eigen::AngleAxisf(origRy, Vector3f::UnitY())) *
          Quaternionf(Eigen::AngleAxisf(origRx, Vector3f::UnitX()));

      const float reRx = reMotion(base + RX, frameIdx);
      const float reRy = reMotion(base + RY, frameIdx);
      const float reRz = reMotion(base + RZ, frameIdx);
      const Quaternionf reQ = Quaternionf(Eigen::AngleAxisf(reRz, Vector3f::UnitZ())) *
          Quaternionf(Eigen::AngleAxisf(reRy, Vector3f::UnitY())) *
          Quaternionf(Eigen::AngleAxisf(reRx, Vector3f::UnitX()));

      EXPECT_TRUE(origQ.toRotationMatrix().isApprox(reQ.toRotationMatrix(), 1e-3f))
          << "Rotation mismatch at joint " << jointIdx << " frame " << frameIdx;

      // Scale should remain zero
      EXPECT_NEAR(reMotion(base + SC, frameIdx), 0.0f, 1e-6f);
    }
  }
}

TEST(IoBvhTest, SaveAndReloadFromBuffer) {
  auto bvhPath = getTestResourcePath("bvh/simple.bvh");
  if (!bvhPath.has_value()) {
    GTEST_SKIP() << "Environment variable 'TEST_RESOURCES_PATH' is not set.";
  }

  // Load the original character with motion
  const auto [origCharacter, origMotionParams, origFps] = loadBvhCharacterWithMotion(*bvhPath);
  const auto& [origParamNames, origMotion] = origMotionParams;

  // Save to byte buffer and reload
  const auto bytes = saveBvhCharacterWithMotion(origCharacter, origMotionParams, origFps);
  EXPECT_GT(bytes.size(), 0u);

  const auto [reCharacter, reMotionParams, reFps] =
      loadBvhCharacterWithMotion(std::span<const std::byte>(bytes));
  const auto& [reParamNames, reMotion] = reMotionParams;

  // Verify basic round-trip properties
  EXPECT_EQ(reCharacter.skeleton.joints.size(), origCharacter.skeleton.joints.size());
  EXPECT_EQ(reMotion.rows(), origMotion.rows());
  EXPECT_EQ(reMotion.cols(), origMotion.cols());
  EXPECT_NEAR(reFps, origFps, 0.5f);

  // Verify joint names
  for (size_t i = 0; i < origCharacter.skeleton.joints.size(); ++i) {
    EXPECT_EQ(reCharacter.skeleton.joints[i].name, origCharacter.skeleton.joints[i].name);
  }
}

} // namespace
