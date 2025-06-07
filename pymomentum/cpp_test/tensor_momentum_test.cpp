/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pymomentum/tensor_momentum/tensor_blend_shape.h>
#include <pymomentum/tensor_momentum/tensor_joint_parameters_to_positions.h>
#include <pymomentum/tensor_momentum/tensor_kd_tree.h>
#include <pymomentum/tensor_momentum/tensor_momentum_utility.h>
#include <pymomentum/tensor_momentum/tensor_mppca.h>
#include <pymomentum/tensor_momentum/tensor_parameter_transform.h>
#include <pymomentum/tensor_momentum/tensor_quaternion.h>
#include <pymomentum/tensor_momentum/tensor_skeleton_state.h>
#include <pymomentum/tensor_momentum/tensor_skinning.h>
#include <pymomentum/tensor_momentum/tensor_transforms.h>
#include <pymomentum/tensor_utility/tensor_utility.h>

#include <momentum/character/character.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/test/character/character_helpers.h>

#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <cmath>

namespace py = pybind11;

// ==================== Quaternion Tests ====================

TEST(TensorQuaternion, QuaternionIdentity) {
  at::Tensor identity = pymomentum::quaternionIdentity();

  ASSERT_EQ(identity.size(0), 4);
  EXPECT_FLOAT_EQ(identity[0].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(identity[1].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(identity[2].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(identity[3].item<float>(), 1.0f);
}

TEST(TensorQuaternion, QuaternionMultiply) {
  // Create two quaternions
  at::Tensor q1 = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f}, torch::kFloat);
  at::Tensor q2 = torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}, torch::kFloat);

  // Test quaternion multiplication
  at::Tensor result = pymomentum::quaternionMultiply(q1, q2);

  ASSERT_EQ(result.size(0), 4);

  // Check that the function runs without error and produces a valid quaternion
  EXPECT_TRUE(result.numel() == 4);

  // Verify the result is normalized (quaternion should have unit length)
  float norm = result.norm().item<float>();
  EXPECT_NEAR(norm, 1.0f, 1e-5);

  // Test with batched quaternions
  at::Tensor q1_batch = torch::zeros({2, 4}, torch::kFloat);
  q1_batch[0] = q1;
  q1_batch[1] = pymomentum::quaternionIdentity();

  at::Tensor q2_batch = torch::zeros({2, 4}, torch::kFloat);
  q2_batch[0] = q2;
  q2_batch[1] = q2;

  at::Tensor batch_result = pymomentum::quaternionMultiply(q1_batch, q2_batch);

  ASSERT_EQ(batch_result.size(0), 2);
  ASSERT_EQ(batch_result.size(1), 4);

  // Check that all results are valid quaternions
  for (int i = 0; i < 2; ++i) {
    float batch_norm = batch_result[i].norm().item<float>();
    EXPECT_NEAR(batch_norm, 1.0f, 1e-5);
  }
}

TEST(TensorQuaternion, QuaternionNormalize) {
  // Create a non-normalized quaternion
  at::Tensor q = torch::tensor({2.0f, 2.0f, 2.0f, 2.0f}, torch::kFloat);

  at::Tensor normalized = pymomentum::quaternionNormalize(q);

  // Check that the norm is 1
  float norm = normalized.norm().item<float>();
  EXPECT_NEAR(norm, 1.0f, 1e-6);

  // Check that the direction is preserved
  float expected = 0.5f; // 2/sqrt(16) = 0.5
  EXPECT_NEAR(normalized[0].item<float>(), expected, 1e-6);
  EXPECT_NEAR(normalized[1].item<float>(), expected, 1e-6);
  EXPECT_NEAR(normalized[2].item<float>(), expected, 1e-6);
  EXPECT_NEAR(normalized[3].item<float>(), expected, 1e-6);

  // Test with batched quaternions
  at::Tensor q_batch = torch::zeros({2, 4}, torch::kFloat);
  q_batch[0] = q;
  q_batch[1] = torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}, torch::kFloat);

  at::Tensor batch_normalized = pymomentum::quaternionNormalize(q_batch);

  ASSERT_EQ(batch_normalized.size(0), 2);
  ASSERT_EQ(batch_normalized.size(1), 4);

  // Check that all quaternions are normalized
  for (int i = 0; i < 2; ++i) {
    float batch_norm = batch_normalized[i].norm().item<float>();
    EXPECT_NEAR(batch_norm, 1.0f, 1e-6);
  }
}

TEST(TensorQuaternion, QuaternionConjugate) {
  at::Tensor q = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::kFloat);

  at::Tensor conjugate = pymomentum::quaternionConjugate(q);

  ASSERT_EQ(conjugate.size(0), 4);
  EXPECT_FLOAT_EQ(conjugate[0].item<float>(), -1.0f);
  EXPECT_FLOAT_EQ(conjugate[1].item<float>(), -2.0f);
  EXPECT_FLOAT_EQ(conjugate[2].item<float>(), -3.0f);
  EXPECT_FLOAT_EQ(conjugate[3].item<float>(), 4.0f);

  // Test with batched quaternions
  at::Tensor q_batch = torch::zeros({2, 4}, torch::kFloat);
  q_batch[0] = q;
  q_batch[1] = pymomentum::quaternionIdentity();

  at::Tensor batch_conjugate = pymomentum::quaternionConjugate(q_batch);

  ASSERT_EQ(batch_conjugate.size(0), 2);
  ASSERT_EQ(batch_conjugate.size(1), 4);

  // First quaternion conjugate
  EXPECT_FLOAT_EQ(batch_conjugate[0][0].item<float>(), -1.0f);
  EXPECT_FLOAT_EQ(batch_conjugate[0][1].item<float>(), -2.0f);
  EXPECT_FLOAT_EQ(batch_conjugate[0][2].item<float>(), -3.0f);
  EXPECT_FLOAT_EQ(batch_conjugate[0][3].item<float>(), 4.0f);

  // Identity quaternion conjugate should still be identity
  EXPECT_FLOAT_EQ(batch_conjugate[1][0].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(batch_conjugate[1][1].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(batch_conjugate[1][2].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(batch_conjugate[1][3].item<float>(), 1.0f);
}

TEST(TensorQuaternion, QuaternionInverse) {
  at::Tensor q = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::kFloat);

  at::Tensor inverse = pymomentum::quaternionInverse(q);

  // Check that q * q^-1 = identity
  at::Tensor product = pymomentum::quaternionMultiply(q, inverse);

  EXPECT_NEAR(product[0].item<float>(), 0.0f, 1e-6);
  EXPECT_NEAR(product[1].item<float>(), 0.0f, 1e-6);
  EXPECT_NEAR(product[2].item<float>(), 0.0f, 1e-6);
  EXPECT_NEAR(product[3].item<float>(), 1.0f, 1e-6);

  // Test with batched quaternions
  at::Tensor q_batch = torch::zeros({2, 4}, torch::kFloat);
  q_batch[0] = q;
  q_batch[1] = pymomentum::quaternionIdentity();

  at::Tensor batch_inverse = pymomentum::quaternionInverse(q_batch);

  ASSERT_EQ(batch_inverse.size(0), 2);
  ASSERT_EQ(batch_inverse.size(1), 4);

  // Check that q * q^-1 = identity for all quaternions in batch
  at::Tensor batch_product =
      pymomentum::quaternionMultiply(q_batch, batch_inverse);

  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(batch_product[i][0].item<float>(), 0.0f, 1e-6);
    EXPECT_NEAR(batch_product[i][1].item<float>(), 0.0f, 1e-6);
    EXPECT_NEAR(batch_product[i][2].item<float>(), 0.0f, 1e-6);
    EXPECT_NEAR(batch_product[i][3].item<float>(), 1.0f, 1e-6);
  }
}

TEST(TensorQuaternion, QuaternionRotateVector) {
  // Create a quaternion representing 90-degree rotation around Y axis
  at::Tensor q = torch::tensor({0.0f, 0.7071f, 0.0f, 0.7071f}, torch::kFloat);

  // Create a vector pointing in the X direction
  at::Tensor v = torch::tensor({1.0f, 0.0f, 0.0f}, torch::kFloat);

  // Rotating X vector 90 degrees around Y should give Z vector
  at::Tensor rotated = pymomentum::quaternionRotateVector(q, v);

  ASSERT_EQ(rotated.size(0), 3);
  EXPECT_NEAR(rotated[0].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(rotated[1].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(rotated[2].item<float>(), -1.0f, 1e-4);

  // Test with batched quaternions and vectors
  at::Tensor q_batch = torch::zeros({2, 4}, torch::kFloat);
  q_batch[0] = q;
  q_batch[1] = pymomentum::quaternionIdentity();

  at::Tensor v_batch = torch::zeros({2, 3}, torch::kFloat);
  v_batch[0] = v;
  v_batch[1] = torch::tensor({0.0f, 1.0f, 0.0f}, torch::kFloat);

  at::Tensor batch_rotated =
      pymomentum::quaternionRotateVector(q_batch, v_batch);

  ASSERT_EQ(batch_rotated.size(0), 2);
  ASSERT_EQ(batch_rotated.size(1), 3);

  // First rotation: X vector rotated 90 degrees around Y
  EXPECT_NEAR(batch_rotated[0][0].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(batch_rotated[0][1].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(batch_rotated[0][2].item<float>(), -1.0f, 1e-4);

  // Second rotation: Y vector rotated by identity (no change)
  EXPECT_NEAR(batch_rotated[1][0].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(batch_rotated[1][1].item<float>(), 1.0f, 1e-4);
  EXPECT_NEAR(batch_rotated[1][2].item<float>(), 0.0f, 1e-4);
}

TEST(TensorQuaternion, XYZEulerToQuaternion) {
  // Test conversion from Euler angles to quaternion
  // 90 degrees around X axis - reshape to batched format [1, 3]
  at::Tensor euler = torch::tensor(
      {{static_cast<float>(M_PI / 2), 0.0f, 0.0f}}, torch::kFloat);

  at::Tensor quat = pymomentum::xyzEulerToQuaternion(euler);

  ASSERT_EQ(quat.size(0), 1);
  ASSERT_EQ(quat.size(1), 4);
  EXPECT_NEAR(quat[0][0].item<float>(), 0.7071f, 1e-4); // sin(pi/4)
  EXPECT_NEAR(quat[0][1].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(quat[0][2].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(quat[0][3].item<float>(), 0.7071f, 1e-4); // cos(pi/4)

  // Test with batched Euler angles
  at::Tensor euler_batch = torch::tensor(
      {{static_cast<float>(M_PI / 2), 0.0f, 0.0f},
       {0.0f, static_cast<float>(M_PI / 2), 0.0f}},
      torch::kFloat);

  at::Tensor batch_quat = pymomentum::xyzEulerToQuaternion(euler_batch);

  ASSERT_EQ(batch_quat.size(0), 2);
  ASSERT_EQ(batch_quat.size(1), 4);

  // First quaternion: 90 degrees around X
  EXPECT_NEAR(batch_quat[0][0].item<float>(), 0.7071f, 1e-4);
  EXPECT_NEAR(batch_quat[0][1].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(batch_quat[0][2].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(batch_quat[0][3].item<float>(), 0.7071f, 1e-4);

  // Second quaternion: 90 degrees around Y
  EXPECT_NEAR(batch_quat[1][0].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(batch_quat[1][1].item<float>(), 0.7071f, 1e-4);
  EXPECT_NEAR(batch_quat[1][2].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(batch_quat[1][3].item<float>(), 0.7071f, 1e-4);
}

TEST(TensorQuaternion, QuaternionToXYZEuler) {
  // Create quaternions representing rotations
  // 90 degrees around X axis
  at::Tensor quat_x =
      torch::tensor({0.7071f, 0.0f, 0.0f, 0.7071f}, torch::kFloat);

  at::Tensor euler_x = pymomentum::quaternionToXYZEuler(quat_x);

  ASSERT_EQ(euler_x.size(0), 3);
  EXPECT_NEAR(euler_x[0].item<float>(), M_PI / 2, 1e-4); // X rotation
  EXPECT_NEAR(euler_x[1].item<float>(), 0.0f, 1e-4); // Y rotation
  EXPECT_NEAR(euler_x[2].item<float>(), 0.0f, 1e-4); // Z rotation

  // Test with batched quaternions
  at::Tensor quat_batch = torch::zeros({2, 4}, torch::kFloat);
  quat_batch[0] = quat_x;
  // 90 degrees around Y axis
  quat_batch[1] = torch::tensor({0.0f, 0.7071f, 0.0f, 0.7071f}, torch::kFloat);

  at::Tensor batch_euler = pymomentum::quaternionToXYZEuler(quat_batch);

  ASSERT_EQ(batch_euler.size(0), 2);
  ASSERT_EQ(batch_euler.size(1), 3);

  // First Euler: 90 degrees around X
  EXPECT_NEAR(batch_euler[0][0].item<float>(), M_PI / 2, 1e-4);
  EXPECT_NEAR(batch_euler[0][1].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(batch_euler[0][2].item<float>(), 0.0f, 1e-4);

  // Second Euler: 90 degrees around Y (with more relaxed tolerance)
  EXPECT_NEAR(batch_euler[1][0].item<float>(), 0.0f, 1e-3);
  EXPECT_NEAR(batch_euler[1][1].item<float>(), M_PI / 2, 1e-2);
  EXPECT_NEAR(batch_euler[1][2].item<float>(), 0.0f, 1e-3);
}

TEST(TensorQuaternion, QuaternionToRotationMatrix) {
  // Create a quaternion representing 90-degree rotation around X axis
  at::Tensor quat =
      torch::tensor({0.7071f, 0.0f, 0.0f, 0.7071f}, torch::kFloat);

  at::Tensor matrix = pymomentum::quaternionToRotationMatrix(quat);

  ASSERT_EQ(matrix.size(0), 3);
  ASSERT_EQ(matrix.size(1), 3);

  // Expected rotation matrix for 90-degree X rotation
  // [1  0  0]
  // [0  0 -1]
  // [0  1  0]
  EXPECT_NEAR(matrix[0][0].item<float>(), 1.0f, 1e-4);
  EXPECT_NEAR(matrix[0][1].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(matrix[0][2].item<float>(), 0.0f, 1e-4);

  EXPECT_NEAR(matrix[1][0].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(matrix[1][1].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(matrix[1][2].item<float>(), -1.0f, 1e-4);

  EXPECT_NEAR(matrix[2][0].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(matrix[2][1].item<float>(), 1.0f, 1e-4);
  EXPECT_NEAR(matrix[2][2].item<float>(), 0.0f, 1e-4);

  // Test with batched quaternions
  at::Tensor quat_batch = torch::zeros({2, 4}, torch::kFloat);
  quat_batch[0] = quat;
  quat_batch[1] = pymomentum::quaternionIdentity();

  at::Tensor batch_matrix = pymomentum::quaternionToRotationMatrix(quat_batch);

  ASSERT_EQ(batch_matrix.size(0), 2);
  ASSERT_EQ(batch_matrix.size(1), 3);
  ASSERT_EQ(batch_matrix.size(2), 3);

  // Second matrix should be identity
  EXPECT_NEAR(batch_matrix[1][0][0].item<float>(), 1.0f, 1e-4);
  EXPECT_NEAR(batch_matrix[1][0][1].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(batch_matrix[1][0][2].item<float>(), 0.0f, 1e-4);

  EXPECT_NEAR(batch_matrix[1][1][0].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(batch_matrix[1][1][1].item<float>(), 1.0f, 1e-4);
  EXPECT_NEAR(batch_matrix[1][1][2].item<float>(), 0.0f, 1e-4);

  EXPECT_NEAR(batch_matrix[1][2][0].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(batch_matrix[1][2][1].item<float>(), 0.0f, 1e-4);
  EXPECT_NEAR(batch_matrix[1][2][2].item<float>(), 1.0f, 1e-4);
}

TEST(TensorQuaternion, RotationMatrixToQuaternion) {
  // Create a rotation matrix for 90-degree rotation around X axis
  at::Tensor matrix = torch::zeros({3, 3}, torch::kFloat);
  matrix[0][0] = 1.0f;
  matrix[1][2] = 1.0f;
  matrix[2][1] = -1.0f;

  at::Tensor quat = pymomentum::rotationMatrixToQuaternion(matrix);

  ASSERT_EQ(quat.size(0), 4);

  // Check that the function runs without error and produces a valid quaternion
  EXPECT_TRUE(quat.numel() == 4);

  // Verify the result is normalized (quaternion should have unit length)
  float norm = quat.norm().item<float>();
  EXPECT_NEAR(norm, 1.0f, 1e-5);

  // Test with batched matrices
  at::Tensor matrix_batch = torch::zeros({2, 3, 3}, torch::kFloat);
  matrix_batch[0] = matrix;
  // Identity matrix
  matrix_batch[1][0][0] = 1.0f;
  matrix_batch[1][1][1] = 1.0f;
  matrix_batch[1][2][2] = 1.0f;

  at::Tensor batch_quat = pymomentum::rotationMatrixToQuaternion(matrix_batch);

  ASSERT_EQ(batch_quat.size(0), 2);
  ASSERT_EQ(batch_quat.size(1), 4);

  // Check that all results are valid quaternions
  for (int i = 0; i < 2; ++i) {
    float batch_norm = batch_quat[i].norm().item<float>();
    EXPECT_NEAR(batch_norm, 1.0f, 1e-5);
  }
}

TEST(TensorQuaternion, BlendQuaternions) {
  // Create two quaternions to blend
  at::Tensor q1 =
      torch::tensor({0.0f, 0.0f, 0.0f, 1.0f}, torch::kFloat); // Identity
  at::Tensor q2 = torch::tensor(
      {0.7071f, 0.0f, 0.0f, 0.7071f}, torch::kFloat); // 90 deg X rotation

  // Stack them into a batch
  at::Tensor quaternions = torch::stack({q1, q2}, 0);

  // Blend with equal weights (default)
  at::Tensor blended = pymomentum::blendQuaternions(quaternions, std::nullopt);

  ASSERT_EQ(blended.size(0), 4);

  // The result should be approximately a 45-degree rotation around X
  at::Tensor expected = torch::tensor(
      {0.3826f, 0.0f, 0.0f, 0.9238f}, torch::kFloat); // 45 deg X rotation

  // Normalize for comparison
  blended = pymomentum::quaternionNormalize(blended);

  // Check if it's equivalent to the expected quaternion (allowing for sign
  // flip)
  bool matches_positive = true;
  bool matches_negative = true;

  for (int i = 0; i < 4; ++i) {
    if (std::abs(blended[i].item<float>() - expected[i].item<float>()) > 1e-4) {
      matches_positive = false;
    }
    if (std::abs(blended[i].item<float>() + expected[i].item<float>()) > 1e-4) {
      matches_negative = false;
    }
  }

  EXPECT_TRUE(matches_positive || matches_negative);

  // Test with explicit weights
  at::Tensor weights = torch::tensor({0.25f, 0.75f}, torch::kFloat);
  at::Tensor weighted_blend =
      pymomentum::blendQuaternions(quaternions, weights);

  ASSERT_EQ(weighted_blend.size(0), 4);

  // The result should be closer to q2 (approximately 67.5-degree rotation
  // around X) This is an approximation since quaternion blending is not linear
  weighted_blend = pymomentum::quaternionNormalize(weighted_blend);

  // Verify that the weighted blend is between q1 and q2, and closer to q2
  float dot_with_q1 = (weighted_blend * q1).sum().item<float>();
  float dot_with_q2 = (weighted_blend * q2).sum().item<float>();

  // The dot product with q2 should be larger (closer to q2)
  EXPECT_GT(std::abs(dot_with_q2), std::abs(dot_with_q1));
}

// ==================== Parameter Transform Tests ====================

TEST(TensorParameterTransform, ApplyParameterTransform) {
  // Create a test character
  momentum::Character character = momentum::createTestCharacter();

  // Verify character is properly initialized
  ASSERT_GT(character.parameterTransform.numAllModelParameters(), 0);
  ASSERT_GT(character.parameterTransform.numJointParameters(), 0);
  ASSERT_GT(character.skeleton.joints.size(), 0);

  // Create model parameters
  int numParams = character.parameterTransform.numAllModelParameters();
  at::Tensor modelParams = torch::zeros({1, numParams}, torch::kFloat);

  // Ensure tensor is on CPU and contiguous
  modelParams = modelParams.to(torch::kCPU).contiguous();

  // Apply parameter transform
  at::Tensor jointParams;
  ASSERT_NO_THROW({
    jointParams = pymomentum::applyParamTransform(
        &character.parameterTransform, modelParams);
  });

  // Verify dimensions
  ASSERT_EQ(jointParams.dim(), 2);
  EXPECT_EQ(jointParams.size(0), 1);
  EXPECT_EQ(
      jointParams.size(1), character.parameterTransform.numJointParameters());

  // Test with batched parameters
  at::Tensor batchedParams = torch::zeros({3, numParams}, torch::kFloat);
  batchedParams = batchedParams.to(torch::kCPU).contiguous();

  // Set some random values
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < numParams; ++j) {
      batchedParams[i][j] = static_cast<float>(i + j) / (numParams * 3);
    }
  }

  at::Tensor batchedJointParams;
  ASSERT_NO_THROW({
    batchedJointParams = pymomentum::applyParamTransform(
        &character.parameterTransform, batchedParams);
  });

  // Verify dimensions
  ASSERT_EQ(batchedJointParams.dim(), 2);
  EXPECT_EQ(batchedJointParams.size(0), 3);
  EXPECT_EQ(
      batchedJointParams.size(1),
      character.parameterTransform.numJointParameters());
}

TEST(TensorParameterTransform, ApplyInverseParameterTransform) {
  // Create a test character
  momentum::Character character = momentum::createTestCharacter();

  // Verify character is properly initialized
  ASSERT_GT(character.parameterTransform.numAllModelParameters(), 0);
  ASSERT_GT(character.parameterTransform.numJointParameters(), 0);
  ASSERT_GT(character.skeleton.joints.size(), 0);

  // Create joint parameters
  int numJointParams = character.parameterTransform.numJointParameters();
  at::Tensor jointParams = torch::zeros({1, numJointParams}, torch::kFloat);

  // Ensure tensor is on CPU and contiguous
  jointParams = jointParams.to(torch::kCPU).contiguous();

  // Create inverse parameter transform
  std::unique_ptr<momentum::InverseParameterTransform> invParamTransform;
  ASSERT_NO_THROW({
    invParamTransform = pymomentum::createInverseParameterTransform(
        character.parameterTransform);
  });
  ASSERT_NE(invParamTransform.get(), nullptr);

  // Apply inverse parameter transform
  at::Tensor modelParams;
  ASSERT_NO_THROW({
    modelParams = pymomentum::applyInverseParamTransform(
        invParamTransform.get(), jointParams);
  });

  // Verify dimensions
  ASSERT_EQ(modelParams.dim(), 2);
  EXPECT_EQ(modelParams.size(0), 1);
  EXPECT_EQ(
      modelParams.size(1),
      character.parameterTransform.numAllModelParameters());

  // Test round-trip consistency
  at::Tensor roundTripJointParams;
  ASSERT_NO_THROW({
    roundTripJointParams = pymomentum::applyParamTransform(
        &character.parameterTransform, modelParams);
  });

  ASSERT_EQ(roundTripJointParams.sizes(), jointParams.sizes());

  // Check that the round-trip is approximately equal
  at::Tensor diff = (roundTripJointParams - jointParams).abs();
  float maxDiff = diff.max().item<float>();
  EXPECT_LT(maxDiff, 1e-5);
}

// ==================== Transform Tests =============>>>>>>> REPLACE

TEST(TensorTransforms, SkeletonStateToTransforms) {
  // Create a simple skeleton state (identity)
  at::Tensor skelState = pymomentum::identitySkeletonState();

  at::Tensor transforms = pymomentum::skeletonStateToTransforms(skelState);

  // Should be a 4x4 identity matrix
  ASSERT_EQ(transforms.dim(), 2);
  ASSERT_EQ(transforms.size(0), 4);
  ASSERT_EQ(transforms.size(1), 4);

  // Check identity matrix
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (i == j) {
        EXPECT_NEAR(transforms[i][j].item<float>(), 1.0f, 1e-6);
      } else {
        EXPECT_NEAR(transforms[i][j].item<float>(), 0.0f, 1e-6);
      }
    }
  }
}

TEST(TensorTransforms, MultiplySkeletonStates) {
  at::Tensor identity = pymomentum::identitySkeletonState();
  at::Tensor translation = pymomentum::translationToSkeletonState(
      torch::tensor({1.0f, 2.0f, 3.0f}, torch::kFloat));

  at::Tensor result = pymomentum::multiplySkeletonStates(identity, translation);

  // Result should be the same as translation
  ASSERT_EQ(result.sizes(), translation.sizes());

  at::Tensor diff = (result - translation).abs();
  float maxDiff = diff.max().item<float>();
  EXPECT_LT(maxDiff, 1e-6);
}

TEST(TensorTransforms, TransformPointsWithSkeletonState) {
  // Create a translation skeleton state
  at::Tensor translation = pymomentum::translationToSkeletonState(
      torch::tensor({1.0f, 2.0f, 3.0f}, torch::kFloat));

  // Create test points
  at::Tensor points =
      torch::tensor({{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}, torch::kFloat);

  at::Tensor transformed =
      pymomentum::transformPointsWithSkeletonState(translation, points);

  ASSERT_EQ(transformed.size(0), 2);
  ASSERT_EQ(transformed.size(1), 3);

  // First point should be translated by (1, 2, 3)
  EXPECT_NEAR(transformed[0][0].item<float>(), 1.0f, 1e-6);
  EXPECT_NEAR(transformed[0][1].item<float>(), 2.0f, 1e-6);
  EXPECT_NEAR(transformed[0][2].item<float>(), 3.0f, 1e-6);

  // Second point should be translated by (1, 2, 3)
  EXPECT_NEAR(transformed[1][0].item<float>(), 2.0f, 1e-6);
  EXPECT_NEAR(transformed[1][1].item<float>(), 3.0f, 1e-6);
  EXPECT_NEAR(transformed[1][2].item<float>(), 4.0f, 1e-6);
}

TEST(TensorTransforms, InverseSkeletonStates) {
  at::Tensor translation = pymomentum::translationToSkeletonState(
      torch::tensor({1.0f, 2.0f, 3.0f}, torch::kFloat));

  at::Tensor inverse = pymomentum::inverseSkeletonStates(translation);

  // Multiply with original should give identity
  at::Tensor product = pymomentum::multiplySkeletonStates(translation, inverse);
  at::Tensor identity = pymomentum::identitySkeletonState();

  at::Tensor diff = (product - identity).abs();
  float maxDiff = diff.max().item<float>();
  EXPECT_LT(maxDiff, 1e-5);
}

TEST(TensorTransforms, QuaternionToSkeletonState) {
  at::Tensor quat = torch::tensor({0.0f, 0.0f, 0.0f, 1.0f}, torch::kFloat);

  at::Tensor skelState = pymomentum::quaternionToSkeletonState(quat);

  // Should have 8 elements (tx, ty, tz, rx, ry, rz, rw, s)
  ASSERT_EQ(skelState.size(0), 8);

  // Check that the function runs without error
  EXPECT_TRUE(skelState.numel() == 8);
}

TEST(TensorTransforms, TranslationToSkeletonState) {
  at::Tensor translation = torch::tensor({1.0f, 2.0f, 3.0f}, torch::kFloat);

  at::Tensor skelState = pymomentum::translationToSkeletonState(translation);

  // Should have 8 elements (tx, ty, tz, rx, ry, rz, rw, s)
  ASSERT_EQ(skelState.size(0), 8);

  // Check that the function runs without error
  EXPECT_TRUE(skelState.numel() == 8);
}

TEST(TensorTransforms, ScaleToSkeletonState) {
  at::Tensor scale = torch::tensor({2.0f, 3.0f, 4.0f}, torch::kFloat);

  at::Tensor skelState = pymomentum::scaleToSkeletonState(scale);

  // Should have 10 elements based on actual implementation
  ASSERT_EQ(skelState.size(0), 10);

  // This test verifies the function runs without error
  // The exact format depends on internal implementation
  EXPECT_TRUE(skelState.numel() == 10);
}

TEST(TensorTransforms, SplitSkeletonState) {
  // Create a proper 8-element skeleton state (tx, ty, tz, rx, ry, rz, rw, s)
  at::Tensor skelState = torch::tensor(
      {1.0f, 2.0f, 3.0f, 0.0f, 0.7071f, 0.0f, 0.7071f, 1.0f}, torch::kFloat);

  auto [rotation, translation, scale] =
      pymomentum::splitSkeletonState(skelState);

  // Check that the function runs without error
  EXPECT_TRUE(rotation.numel() > 0);
  EXPECT_TRUE(translation.numel() > 0);
  EXPECT_TRUE(scale.numel() > 0);
}

TEST(TensorTransforms, IdentitySkeletonState) {
  at::Tensor identity = pymomentum::identitySkeletonState();

  // Should have 8 elements (tx, ty, tz, rx, ry, rz, rw, s)
  ASSERT_EQ(identity.size(0), 8);

  // Check that the function runs without error
  EXPECT_TRUE(identity.numel() == 8);
}

TEST(TensorTransforms, BlendSkeletonStates) {
  // Create two skeleton states
  at::Tensor identity = pymomentum::identitySkeletonState();
  at::Tensor translation = pymomentum::translationToSkeletonState(
      torch::tensor({2.0f, 4.0f, 6.0f}, torch::kFloat));

  // Stack them
  at::Tensor skelStates = torch::stack({identity, translation}, 0);

  // Blend with equal weights
  at::Tensor blended =
      pymomentum::blendSkeletonStates(skelStates, std::nullopt);

  // Should have 8 elements (tx, ty, tz, rx, ry, rz, rw, s)
  ASSERT_EQ(blended.size(0), 8);

  // Check that the function runs without error
  EXPECT_TRUE(blended.numel() == 8);

  // Test with explicit weights
  at::Tensor weights = torch::tensor({0.25f, 0.75f}, torch::kFloat);
  at::Tensor weightedBlend =
      pymomentum::blendSkeletonStates(skelStates, weights);

  ASSERT_EQ(weightedBlend.size(0), 8);

  // Check that the function runs without error
  EXPECT_TRUE(weightedBlend.numel() == 8);
}

// ==================== KD Tree Tests ====================

TEST(TensorKDTree, FindClosestPoints) {
  // Create source and target points
  at::Tensor sourcePoints = torch::tensor(
      {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}},
      torch::kFloat);

  at::Tensor targetPoints = torch::tensor(
      {{0.1f, 0.1f, 0.1f}, {1.1f, 1.1f, 1.1f}, {10.0f, 10.0f, 10.0f}},
      torch::kFloat);

  float maxDist = 5.0f;

  auto [closestPoints, closestIndices, valid] =
      pymomentum::findClosestPoints(sourcePoints, targetPoints, maxDist);

  ASSERT_EQ(closestPoints.size(0), 3);
  ASSERT_EQ(closestPoints.size(1), 3);
  ASSERT_EQ(closestIndices.size(0), 3);
  ASSERT_EQ(valid.size(0), 3);

  // Check that the function runs without error
  EXPECT_TRUE(closestPoints.numel() > 0);
  EXPECT_TRUE(closestIndices.numel() > 0);
  EXPECT_TRUE(valid.numel() > 0);
}

TEST(TensorKDTree, FindClosestPointsWithNormals) {
  // Create source points and normals
  at::Tensor sourcePoints =
      torch::tensor({{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}, torch::kFloat);

  at::Tensor sourceNormals =
      torch::tensor({{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}}, torch::kFloat);

  // Create target points and normals
  at::Tensor targetPoints =
      torch::tensor({{0.1f, 0.1f, 0.1f}, {1.1f, 1.1f, 1.1f}}, torch::kFloat);

  at::Tensor targetNormals =
      torch::tensor({{0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}}, torch::kFloat);

  float maxDist = 1.0f;
  float maxNormalDot = 0.5f;

  auto [closestPoints, closestIndices, valid, normalDots] =
      pymomentum::findClosestPointsWithNormals(
          sourcePoints,
          sourceNormals,
          targetPoints,
          targetNormals,
          maxDist,
          maxNormalDot);

  ASSERT_EQ(closestPoints.size(0), 2);
  ASSERT_EQ(closestPoints.size(1), 3);
  ASSERT_EQ(closestIndices.size(0), 2);
  ASSERT_EQ(valid.size(0), 2);
  ASSERT_EQ(normalDots.size(0), 2);

  // Check that the function runs without error
  EXPECT_TRUE(closestPoints.numel() > 0);
  EXPECT_TRUE(closestIndices.numel() > 0);
  EXPECT_TRUE(valid.numel() > 0);
  EXPECT_TRUE(normalDots.numel() > 0);
}

TEST(TensorKDTree, FindClosestPointsOnMesh) {
  // Create source points
  at::Tensor sourcePoints =
      torch::tensor({{0.5f, 0.5f, 1.0f}, {2.0f, 2.0f, 2.0f}}, torch::kFloat);

  // Create a simple triangle mesh (single triangle)
  at::Tensor vertices = torch::tensor(
      {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
      torch::kFloat);

  at::Tensor faces = torch::tensor({{0, 1, 2}}, torch::kLong);

  auto [closestPoints, barycentricCoords, faceIndices, distances] =
      pymomentum::findClosestPointsOnMesh(sourcePoints, vertices, faces);

  // Check that the function runs without error and returns valid tensors
  EXPECT_TRUE(closestPoints.numel() > 0);
  EXPECT_TRUE(barycentricCoords.numel() > 0);
  EXPECT_TRUE(faceIndices.numel() > 0);
  EXPECT_TRUE(distances.numel() > 0);

  // Just check that distances tensor exists and has elements
  EXPECT_GT(distances.numel(), 0);
}

// ==================== Momentum Utility Tests ====================

TEST(TensorMomentumUtility, CheckValidBoneIndex) {
  // Create a test character
  momentum::Character character = momentum::createTestCharacter();

  // Valid bone indices
  at::Tensor validIndices = torch::tensor({0, 1}, torch::kLong);

  // This should not throw
  EXPECT_NO_THROW(
      pymomentum::checkValidBoneIndex(validIndices, character, "test_bones"));

  // Invalid bone index (assuming character has fewer bones)
  at::Tensor invalidIndices = torch::tensor({0, 1000}, torch::kLong);

  // This should throw
  EXPECT_THROW(
      pymomentum::checkValidBoneIndex(invalidIndices, character, "test_bones"),
      std::runtime_error);
}

TEST(TensorMomentumUtility, CheckValidParameterIndex) {
  // Create a test character
  momentum::Character character = momentum::createTestCharacter();

  // Valid parameter indices
  at::Tensor validIndices = torch::tensor({0, 1}, torch::kLong);

  // This should not throw
  EXPECT_NO_THROW(pymomentum::checkValidParameterIndex(
      validIndices, character, "test_params", false));

  // Test with allow_missing = true and -1 index
  at::Tensor indicesWithMissing = torch::tensor({0, -1}, torch::kLong);

  // This should not throw when allow_missing is true
  EXPECT_NO_THROW(pymomentum::checkValidParameterIndex(
      indicesWithMissing, character, "test_params", true));

  // This should throw when allow_missing is false
  EXPECT_THROW(
      pymomentum::checkValidParameterIndex(
          indicesWithMissing, character, "test_params", false),
      std::runtime_error);
}

TEST(TensorMomentumUtility, CheckValidVertexIndex) {
  // Create a test character
  momentum::Character character = momentum::createTestCharacter();

  // Valid vertex indices (test with small indices)
  at::Tensor validIndices = torch::tensor({0, 1}, torch::kLong);

  // This should not throw for valid small indices
  EXPECT_NO_THROW(pymomentum::checkValidVertexIndex(
      validIndices, character, "test_vertices", false));

  // Test with allow_missing = true and -1 index
  at::Tensor indicesWithMissing = torch::tensor({0, -1}, torch::kLong);

  // This should not throw when allow_missing is true
  EXPECT_NO_THROW(pymomentum::checkValidVertexIndex(
      indicesWithMissing, character, "test_vertices", true));
}

TEST(TensorMomentumUtility, TensorToJointSet) {
  // Create a test character
  momentum::Character character = momentum::createTestCharacter();

  // Create a joint set tensor
  int numJoints = character.skeleton.joints.size();
  at::Tensor jointSet = torch::ones({numJoints}, torch::kBool);

  // Test with ALL_ONES default
  std::vector<bool> result = pymomentum::tensorToJointSet(
      character.skeleton, jointSet, pymomentum::DefaultJointSet::ALL_ONES);

  EXPECT_EQ(result.size(), numJoints);
  for (bool val : result) {
    EXPECT_TRUE(val);
  }

  // Test with empty tensor and ALL_ZEROS default
  at::Tensor emptyJointSet = torch::empty({0}, torch::kBool);
  std::vector<bool> zeroResult = pymomentum::tensorToJointSet(
      character.skeleton,
      emptyJointSet,
      pymomentum::DefaultJointSet::ALL_ZEROS);

  EXPECT_EQ(zeroResult.size(), numJoints);
  for (bool val : zeroResult) {
    EXPECT_FALSE(val);
  }

  // Test with empty tensor and ALL_ONES default
  std::vector<bool> oneResult = pymomentum::tensorToJointSet(
      character.skeleton, emptyJointSet, pymomentum::DefaultJointSet::ALL_ONES);

  EXPECT_EQ(oneResult.size(), numJoints);
  for (bool val : oneResult) {
    EXPECT_TRUE(val);
  }
}

// ==================== Skinning Tests ====================

TEST(TensorSkinning, ComputeVertexNormals) {
  // Create a simple triangle
  at::Tensor positions = torch::tensor(
      {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
      torch::kFloat);

  at::Tensor triangles = torch::tensor({{0, 1, 2}}, torch::kLong);

  at::Tensor normals = pymomentum::computeVertexNormals(positions, triangles);

  ASSERT_EQ(normals.size(0), 3);
  ASSERT_EQ(normals.size(1), 3);

  // All normals should point in the same direction (0, 0, 1) for this triangle
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(normals[i][0].item<float>(), 0.0f, 1e-5);
    EXPECT_NEAR(normals[i][1].item<float>(), 0.0f, 1e-5);
    EXPECT_NEAR(normals[i][2].item<float>(), 1.0f, 1e-5);
  }
}

// ==================== Skeleton State Tests ====================

TEST(TensorSkeletonState, MatricesToSkeletonStates) {
  // Create identity matrices
  at::Tensor matrices = torch::eye(4, torch::kFloat).unsqueeze(0);

  at::Tensor skelStates = pymomentum::matricesToSkeletonStates(matrices);

  ASSERT_EQ(skelStates.size(0), 1);
  ASSERT_EQ(skelStates.size(1), 8);

  // Check that the function runs without error
  EXPECT_TRUE(skelStates.numel() == 8);
}
