/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <pymomentum/tensor_momentum/tensor_kd_tree.h>
#include <pymomentum/tensor_momentum/tensor_quaternion.h>
#include <pymomentum/tensor_momentum/tensor_skinning.h>
#include <pymomentum/tensor_momentum/tensor_transforms.h>
#include <pymomentum/tensor_utility/autograd_utility.h>

#include <cmath>

// =============================================================================
// tensor_quaternion.h tests
// =============================================================================

TEST(TensorQuaternion, QuaternionIdentity) {
  // Eigen quaternion identity coeffs are (x=0, y=0, z=0, w=1)
  at::Tensor identity = pymomentum::quaternionIdentity();
  EXPECT_EQ(1, identity.ndimension());
  EXPECT_EQ(4, identity.size(0));
  EXPECT_FLOAT_EQ(0.0f, identity[0].item<float>());
  EXPECT_FLOAT_EQ(0.0f, identity[1].item<float>());
  EXPECT_FLOAT_EQ(0.0f, identity[2].item<float>());
  EXPECT_FLOAT_EQ(1.0f, identity[3].item<float>());
}

TEST(TensorQuaternion, CheckQuaternion) {
  // Valid quaternion (last dim = 4)
  EXPECT_NO_THROW(pymomentum::checkQuaternion(at::zeros({4})));
  EXPECT_NO_THROW(pymomentum::checkQuaternion(at::zeros({3, 4})));

  // Invalid quaternion (last dim != 4)
  EXPECT_THROW(pymomentum::checkQuaternion(at::zeros({3})), std::runtime_error);
  EXPECT_THROW(pymomentum::checkQuaternion(at::zeros({2, 5})), std::runtime_error);
}

TEST(TensorQuaternion, SplitQuaternion) {
  // q = [x, y, z, w] -> scalar = [w], vec = [x, y, z]
  at::Tensor q = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
  auto [scalar, vec] = pymomentum::splitQuaternion(q);

  // Scalar part is the w component (index 3)
  EXPECT_FLOAT_EQ(4.0f, scalar[0].item<float>());

  // Vector part is [x, y, z] (indices 0-2)
  EXPECT_FLOAT_EQ(1.0f, vec[0].item<float>());
  EXPECT_FLOAT_EQ(2.0f, vec[1].item<float>());
  EXPECT_FLOAT_EQ(3.0f, vec[2].item<float>());
}

TEST(TensorQuaternion, QuaternionNormalize) {
  at::Tensor q = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
  at::Tensor normalized = pymomentum::quaternionNormalize(q);

  // Check the result has unit norm
  auto norm = normalized.norm().item<float>();
  EXPECT_NEAR(1.0f, norm, 1e-6f);

  // Check direction is preserved
  auto origNorm = q.norm().item<float>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(q[i].item<float>() / origNorm, normalized[i].item<float>(), 1e-6f);
  }
}

TEST(TensorQuaternion, QuaternionConjugate) {
  at::Tensor q = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});
  at::Tensor conj = pymomentum::quaternionConjugate(q);

  // Conjugate negates the vector part, keeps scalar
  EXPECT_FLOAT_EQ(-1.0f, conj[0].item<float>());
  EXPECT_FLOAT_EQ(-2.0f, conj[1].item<float>());
  EXPECT_FLOAT_EQ(-3.0f, conj[2].item<float>());
  EXPECT_FLOAT_EQ(4.0f, conj[3].item<float>());
}

TEST(TensorQuaternion, QuaternionInverse) {
  // For a unit quaternion, inverse == conjugate
  at::Tensor q = pymomentum::quaternionNormalize(torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}));
  at::Tensor inv = pymomentum::quaternionInverse(q);
  at::Tensor conj = pymomentum::quaternionConjugate(q);

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(conj[i].item<float>(), inv[i].item<float>(), 1e-6f);
  }

  // q * q^-1 should be identity
  at::Tensor product = pymomentum::quaternionMultiply(q, inv);
  at::Tensor identity = pymomentum::quaternionIdentity();
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(identity[i].item<float>(), product[i].item<float>(), 1e-5f);
  }
}

TEST(TensorQuaternion, QuaternionMultiplyIdentity) {
  at::Tensor identity = pymomentum::quaternionIdentity();
  at::Tensor q = pymomentum::quaternionNormalize(torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}));

  // q * identity = q
  at::Tensor result = pymomentum::quaternionMultiply(q, identity);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(q[i].item<float>(), result[i].item<float>(), 1e-6f);
  }

  // identity * q = q
  result = pymomentum::quaternionMultiply(identity, q);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(q[i].item<float>(), result[i].item<float>(), 1e-6f);
  }
}

TEST(TensorQuaternion, QuaternionMultiplyBatch) {
  // Test batched quaternion multiplication
  at::Tensor q1 = pymomentum::quaternionNormalize(torch::tensor({{1.0f, 0.0f, 0.0f, 1.0f}}));
  at::Tensor q2 = pymomentum::quaternionNormalize(torch::tensor({{0.0f, 1.0f, 0.0f, 1.0f}}));

  at::Tensor result = pymomentum::quaternionMultiply(q1, q2);
  EXPECT_EQ(2, result.ndimension());
  EXPECT_EQ(1, result.size(0));
  EXPECT_EQ(4, result.size(1));

  // Result should be a unit quaternion
  auto norm = result[0].norm().item<float>();
  EXPECT_NEAR(1.0f, norm, 1e-5f);
}

TEST(TensorQuaternion, QuaternionRotateVector) {
  // 90 degree rotation around Z axis: q = (0, 0, sin(pi/4), cos(pi/4))
  float s = std::sin(M_PI / 4.0f);
  float c = std::cos(M_PI / 4.0f);
  at::Tensor q = torch::tensor({0.0f, 0.0f, s, c});
  at::Tensor v = torch::tensor({1.0f, 0.0f, 0.0f});

  at::Tensor rotated = pymomentum::quaternionRotateVector(q, v);

  // (1,0,0) rotated 90 degrees around Z should give approximately (0,1,0)
  EXPECT_NEAR(0.0f, rotated[0].item<float>(), 1e-5f);
  EXPECT_NEAR(1.0f, rotated[1].item<float>(), 1e-5f);
  EXPECT_NEAR(0.0f, rotated[2].item<float>(), 1e-5f);
}

TEST(TensorQuaternion, QuaternionRotateVectorIdentity) {
  at::Tensor identity = pymomentum::quaternionIdentity();
  at::Tensor v = torch::tensor({1.0f, 2.0f, 3.0f});

  at::Tensor rotated = pymomentum::quaternionRotateVector(identity, v);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(v[i].item<float>(), rotated[i].item<float>(), 1e-6f);
  }
}

TEST(TensorQuaternion, EulerToQuaternionAndBack) {
  // Test round-trip: euler -> quaternion -> euler
  // xyzEulerToQuaternion expects shape [nEuler, 3]
  at::Tensor euler = torch::tensor({{0.1f, 0.2f, 0.3f}});

  at::Tensor quat = pymomentum::xyzEulerToQuaternion(euler);
  EXPECT_EQ(4, quat.size(-1));

  // The quaternion should be unit length
  auto norm = quat[0].norm().item<float>();
  EXPECT_NEAR(1.0f, norm, 1e-5f);

  at::Tensor eulerBack = pymomentum::quaternionToXYZEuler(quat);
  EXPECT_EQ(3, eulerBack.size(-1));

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(euler[0][i].item<float>(), eulerBack[0][i].item<float>(), 1e-5f);
  }
}

TEST(TensorQuaternion, EulerToQuaternionZero) {
  // xyzEulerToQuaternion expects shape [nEuler, 3]
  at::Tensor zeroEuler = torch::tensor({{0.0f, 0.0f, 0.0f}});
  at::Tensor quat = pymomentum::xyzEulerToQuaternion(zeroEuler);
  at::Tensor identity = pymomentum::quaternionIdentity();

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(identity[i].item<float>(), quat[0][i].item<float>(), 1e-6f);
  }
}

TEST(TensorQuaternion, QuaternionToRotationMatrix) {
  // Identity quaternion should produce identity rotation matrix
  at::Tensor identity = pymomentum::quaternionIdentity();
  at::Tensor rotMat = pymomentum::quaternionToRotationMatrix(identity);

  EXPECT_EQ(2, rotMat.ndimension());
  EXPECT_EQ(3, rotMat.size(0));
  EXPECT_EQ(3, rotMat.size(1));

  at::Tensor expected = torch::eye(3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(expected[i][j].item<float>(), rotMat[i][j].item<float>(), 1e-5f);
    }
  }
}

TEST(TensorQuaternion, QuaternionToRotationMatrixRoundTrip) {
  // Construct a unit quaternion and check round-trip
  at::Tensor q = pymomentum::quaternionNormalize(torch::tensor({0.1f, 0.2f, 0.3f, 0.9f}));
  at::Tensor rotMat = pymomentum::quaternionToRotationMatrix(q);

  // Verify it's a proper rotation matrix: R^T * R = I
  at::Tensor product = at::matmul(rotMat.transpose(0, 1), rotMat);
  at::Tensor eye = torch::eye(3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(eye[i][j].item<float>(), product[i][j].item<float>(), 1e-5f);
    }
  }

  // Verify determinant is 1
  auto det = at::det(rotMat).item<float>();
  EXPECT_NEAR(1.0f, det, 1e-5f);
}

TEST(TensorQuaternion, RotationMatrixToQuaternionRoundTrip) {
  // rotationMatrixToQuaternion uses at::linalg_eig internally, which calls
  // LAPACK routines that are incompatible with AddressSanitizer.
#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
  GTEST_SKIP() << "linalg_eig is incompatible with ASAN";
#endif

  // rotationMatrixToQuaternion expects batched input (..., 3, 3)
  at::Tensor q = pymomentum::quaternionNormalize(torch::tensor({{0.1f, 0.2f, 0.3f, 0.9f}}));
  at::Tensor rotMat = pymomentum::quaternionToRotationMatrix(q);
  at::Tensor qBack = pymomentum::rotationMatrixToQuaternion(rotMat);

  // Quaternions q and -q represent the same rotation, so check |dot product| ~= 1
  auto dot = (q * qBack).sum().item<float>();
  EXPECT_NEAR(1.0f, std::abs(dot), 1e-4f);
}

TEST(TensorQuaternion, BlendQuaternionsIdentical) {
  // Blending identical quaternions should return the same quaternion
  at::Tensor q = pymomentum::quaternionNormalize(torch::tensor({0.1f, 0.2f, 0.3f, 0.9f}));
  at::Tensor quaternions = at::stack({q, q, q}, 0);

  at::Tensor blended = pymomentum::blendQuaternions(quaternions, std::nullopt);

  auto dot = (q * blended).sum().item<float>();
  EXPECT_NEAR(1.0f, std::abs(dot), 1e-5f);
}

TEST(TensorQuaternion, CheckAndNormalizeWeightsDefault) {
  // When no weights provided, should get uniform weights
  at::Tensor quaternions = at::zeros({3, 4});
  at::Tensor weights = pymomentum::checkAndNormalizeWeights(quaternions, std::nullopt);

  EXPECT_EQ(1, weights.ndimension());
  EXPECT_EQ(3, weights.size(0));
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(1.0f / 3.0f, weights[i].item<float>(), 1e-6f);
  }
}

TEST(TensorQuaternion, CheckAndNormalizeWeightsCustom) {
  at::Tensor quaternions = at::zeros({3, 4});
  at::Tensor customWeights = torch::tensor({1.0f, 2.0f, 3.0f});
  at::Tensor weights = pymomentum::checkAndNormalizeWeights(quaternions, customWeights);

  // Sum should be 1
  auto sum = weights.sum().item<float>();
  EXPECT_NEAR(1.0f, sum, 1e-6f);

  // Verify proportions are maintained
  EXPECT_NEAR(1.0f / 6.0f, weights[0].item<float>(), 1e-6f);
  EXPECT_NEAR(2.0f / 6.0f, weights[1].item<float>(), 1e-6f);
  EXPECT_NEAR(3.0f / 6.0f, weights[2].item<float>(), 1e-6f);
}

// =============================================================================
// tensor_transforms.h tests
// =============================================================================

TEST(TensorTransforms, IdentitySkeletonState) {
  at::Tensor identity = pymomentum::identitySkeletonState();

  EXPECT_EQ(1, identity.ndimension());
  EXPECT_EQ(8, identity.size(0));

  // Translation [0,0,0]
  EXPECT_FLOAT_EQ(0.0f, identity[0].item<float>());
  EXPECT_FLOAT_EQ(0.0f, identity[1].item<float>());
  EXPECT_FLOAT_EQ(0.0f, identity[2].item<float>());

  // Quaternion identity [0,0,0,1]
  EXPECT_FLOAT_EQ(0.0f, identity[3].item<float>());
  EXPECT_FLOAT_EQ(0.0f, identity[4].item<float>());
  EXPECT_FLOAT_EQ(0.0f, identity[5].item<float>());
  EXPECT_FLOAT_EQ(1.0f, identity[6].item<float>());

  // Scale [1]
  EXPECT_FLOAT_EQ(1.0f, identity[7].item<float>());
}

TEST(TensorTransforms, SplitSkeletonState) {
  // [tx, ty, tz, rx, ry, rz, rw, s]
  at::Tensor skelState = torch::tensor({1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.5f});

  auto [t, q, s] = pymomentum::splitSkeletonState(skelState);

  EXPECT_EQ(3, t.size(-1));
  EXPECT_FLOAT_EQ(1.0f, t[0].item<float>());
  EXPECT_FLOAT_EQ(2.0f, t[1].item<float>());
  EXPECT_FLOAT_EQ(3.0f, t[2].item<float>());

  EXPECT_EQ(4, q.size(-1));
  EXPECT_FLOAT_EQ(0.0f, q[0].item<float>());
  EXPECT_FLOAT_EQ(0.0f, q[1].item<float>());
  EXPECT_FLOAT_EQ(0.0f, q[2].item<float>());
  EXPECT_FLOAT_EQ(1.0f, q[3].item<float>());

  EXPECT_EQ(1, s.size(-1));
  EXPECT_FLOAT_EQ(2.5f, s[0].item<float>());
}

TEST(TensorTransforms, SplitSkeletonStateInvalidDim) {
  // Last dim != 8 should throw
  EXPECT_THROW(pymomentum::splitSkeletonState(at::zeros({7})), std::runtime_error);
  EXPECT_THROW(pymomentum::splitSkeletonState(at::zeros({9})), std::runtime_error);
}

TEST(TensorTransforms, QuaternionToSkeletonState) {
  at::Tensor q = pymomentum::quaternionIdentity();
  at::Tensor skelState = pymomentum::quaternionToSkeletonState(q);

  EXPECT_EQ(8, skelState.size(-1));

  // Translation should be zero
  EXPECT_FLOAT_EQ(0.0f, skelState[0].item<float>());
  EXPECT_FLOAT_EQ(0.0f, skelState[1].item<float>());
  EXPECT_FLOAT_EQ(0.0f, skelState[2].item<float>());

  // Quaternion should match input
  EXPECT_FLOAT_EQ(0.0f, skelState[3].item<float>());
  EXPECT_FLOAT_EQ(0.0f, skelState[4].item<float>());
  EXPECT_FLOAT_EQ(0.0f, skelState[5].item<float>());
  EXPECT_FLOAT_EQ(1.0f, skelState[6].item<float>());

  // Scale should be 1
  EXPECT_FLOAT_EQ(1.0f, skelState[7].item<float>());
}

TEST(TensorTransforms, TranslationToSkeletonState) {
  at::Tensor t = torch::tensor({5.0f, 6.0f, 7.0f});
  at::Tensor skelState = pymomentum::translationToSkeletonState(t);

  EXPECT_EQ(8, skelState.size(-1));

  // Translation
  EXPECT_FLOAT_EQ(5.0f, skelState[0].item<float>());
  EXPECT_FLOAT_EQ(6.0f, skelState[1].item<float>());
  EXPECT_FLOAT_EQ(7.0f, skelState[2].item<float>());

  // Identity quaternion
  EXPECT_FLOAT_EQ(0.0f, skelState[3].item<float>());
  EXPECT_FLOAT_EQ(0.0f, skelState[4].item<float>());
  EXPECT_FLOAT_EQ(0.0f, skelState[5].item<float>());
  EXPECT_FLOAT_EQ(1.0f, skelState[6].item<float>());

  // Scale = 1
  EXPECT_FLOAT_EQ(1.0f, skelState[7].item<float>());
}

TEST(TensorTransforms, TranslationToSkeletonStateInvalidDim) {
  EXPECT_THROW(pymomentum::translationToSkeletonState(at::zeros({4})), std::runtime_error);
}

TEST(TensorTransforms, ScaleToSkeletonState) {
  at::Tensor s = torch::tensor({2.0f});
  at::Tensor skelState = pymomentum::scaleToSkeletonState(s);

  EXPECT_EQ(8, skelState.size(-1));

  // Zero translation
  EXPECT_FLOAT_EQ(0.0f, skelState[0].item<float>());
  EXPECT_FLOAT_EQ(0.0f, skelState[1].item<float>());
  EXPECT_FLOAT_EQ(0.0f, skelState[2].item<float>());

  // Identity quaternion
  EXPECT_FLOAT_EQ(0.0f, skelState[3].item<float>());
  EXPECT_FLOAT_EQ(0.0f, skelState[4].item<float>());
  EXPECT_FLOAT_EQ(0.0f, skelState[5].item<float>());
  EXPECT_FLOAT_EQ(1.0f, skelState[6].item<float>());

  // Scale
  EXPECT_FLOAT_EQ(2.0f, skelState[7].item<float>());
}

TEST(TensorTransforms, SkeletonStateToTransformsIdentity) {
  at::Tensor identity = pymomentum::identitySkeletonState();
  at::Tensor transform = pymomentum::skeletonStateToTransforms(identity);

  EXPECT_EQ(2, transform.ndimension());
  EXPECT_EQ(4, transform.size(0));
  EXPECT_EQ(4, transform.size(1));

  // Should produce 4x4 identity matrix
  at::Tensor expected = torch::eye(4);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NEAR(expected[i][j].item<float>(), transform[i][j].item<float>(), 1e-5f);
    }
  }
}

TEST(TensorTransforms, SkeletonStateToTransformsTranslation) {
  at::Tensor skelState = pymomentum::translationToSkeletonState(torch::tensor({1.0f, 2.0f, 3.0f}));
  at::Tensor transform = pymomentum::skeletonStateToTransforms(skelState);

  // 3x3 rotation block should be identity
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      float expected = (i == j) ? 1.0f : 0.0f;
      EXPECT_NEAR(expected, transform[i][j].item<float>(), 1e-5f);
    }
  }

  // Translation column
  EXPECT_NEAR(1.0f, transform[0][3].item<float>(), 1e-5f);
  EXPECT_NEAR(2.0f, transform[1][3].item<float>(), 1e-5f);
  EXPECT_NEAR(3.0f, transform[2][3].item<float>(), 1e-5f);

  // Last row
  EXPECT_NEAR(0.0f, transform[3][0].item<float>(), 1e-5f);
  EXPECT_NEAR(0.0f, transform[3][1].item<float>(), 1e-5f);
  EXPECT_NEAR(0.0f, transform[3][2].item<float>(), 1e-5f);
  EXPECT_NEAR(1.0f, transform[3][3].item<float>(), 1e-5f);
}

TEST(TensorTransforms, MultiplySkeletonStatesIdentity) {
  at::Tensor identity = pymomentum::identitySkeletonState();
  at::Tensor skelState = torch::tensor({1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.5f});

  at::Tensor result = pymomentum::multiplySkeletonStates(identity, skelState);
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(skelState[i].item<float>(), result[i].item<float>(), 1e-5f);
  }
}

TEST(TensorTransforms, TransformPointsWithSkeletonStateIdentity) {
  at::Tensor identity = pymomentum::identitySkeletonState();
  at::Tensor point = torch::tensor({1.0f, 2.0f, 3.0f});

  at::Tensor result = pymomentum::transformPointsWithSkeletonState(identity, point);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(point[i].item<float>(), result[i].item<float>(), 1e-5f);
  }
}

TEST(TensorTransforms, TransformPointsWithTranslation) {
  at::Tensor skelState =
      pymomentum::translationToSkeletonState(torch::tensor({10.0f, 20.0f, 30.0f}));
  at::Tensor point = torch::tensor({1.0f, 2.0f, 3.0f});

  at::Tensor result = pymomentum::transformPointsWithSkeletonState(skelState, point);
  EXPECT_NEAR(11.0f, result[0].item<float>(), 1e-5f);
  EXPECT_NEAR(22.0f, result[1].item<float>(), 1e-5f);
  EXPECT_NEAR(33.0f, result[2].item<float>(), 1e-5f);
}

TEST(TensorTransforms, TransformPointsWithScale) {
  at::Tensor skelState = pymomentum::scaleToSkeletonState(torch::tensor({2.0f}));
  at::Tensor point = torch::tensor({1.0f, 2.0f, 3.0f});

  at::Tensor result = pymomentum::transformPointsWithSkeletonState(skelState, point);
  EXPECT_NEAR(2.0f, result[0].item<float>(), 1e-5f);
  EXPECT_NEAR(4.0f, result[1].item<float>(), 1e-5f);
  EXPECT_NEAR(6.0f, result[2].item<float>(), 1e-5f);
}

TEST(TensorTransforms, TransformPointsInvalidDim) {
  at::Tensor identity = pymomentum::identitySkeletonState();
  EXPECT_THROW(
      pymomentum::transformPointsWithSkeletonState(identity, at::zeros({4})), std::runtime_error);
}

TEST(TensorTransforms, InverseSkeletonStates) {
  // Create a skeleton state with translation and scale
  at::Tensor skelState = torch::tensor({1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f});

  at::Tensor inverse = pymomentum::inverseSkeletonStates(skelState);

  // Multiplying state by its inverse should give identity
  at::Tensor product = pymomentum::multiplySkeletonStates(skelState, inverse);
  at::Tensor identity = pymomentum::identitySkeletonState();

  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(identity[i].item<float>(), product[i].item<float>(), 1e-5f);
  }
}

TEST(TensorTransforms, InverseSkeletonStatesWithRotation) {
  // Create a skeleton state with a rotation
  float s = std::sin(M_PI / 4.0f);
  float c = std::cos(M_PI / 4.0f);
  at::Tensor skelState = torch::tensor({1.0f, 2.0f, 3.0f, 0.0f, 0.0f, s, c, 1.0f});

  at::Tensor inverse = pymomentum::inverseSkeletonStates(skelState);
  at::Tensor product = pymomentum::multiplySkeletonStates(skelState, inverse);
  at::Tensor identity = pymomentum::identitySkeletonState();

  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(identity[i].item<float>(), product[i].item<float>(), 1e-4f);
  }
}

TEST(TensorTransforms, BlendSkeletonStatesIdentical) {
  at::Tensor skelState = torch::tensor({1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.5f});
  at::Tensor states = at::stack({skelState, skelState, skelState}, 0);

  at::Tensor blended = pymomentum::blendSkeletonStates(states, std::nullopt);

  // Translation and scale should be averaged (but all equal so same value)
  EXPECT_NEAR(1.0f, blended[0].item<float>(), 1e-5f);
  EXPECT_NEAR(2.0f, blended[1].item<float>(), 1e-5f);
  EXPECT_NEAR(3.0f, blended[2].item<float>(), 1e-5f);

  // Scale
  EXPECT_NEAR(1.5f, blended[7].item<float>(), 1e-5f);
}

// =============================================================================
// tensor_kd_tree.h tests
// =============================================================================

TEST(TensorKdTree, FindClosestPointsExact) {
  // Source and target are the same points
  at::Tensor source = torch::tensor({{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}});
  at::Tensor target = torch::tensor({{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}});

  auto [closest_points, closest_indices, valid] =
      pymomentum::findClosestPoints(source, target, FLT_MAX);

  // All points should be valid
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(valid[i].item<bool>());
  }

  // Each source point should match itself exactly
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(source[i][j].item<float>(), closest_points[i][j].item<float>(), 1e-5f);
    }
  }
}

TEST(TensorKdTree, FindClosestPointsNearest) {
  at::Tensor source = torch::tensor({{0.0f, 0.0f, 0.0f}});
  at::Tensor target = torch::tensor({{10.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {5.0f, 0.0f, 0.0f}});

  auto [closest_points, closest_indices, valid] =
      pymomentum::findClosestPoints(source, target, FLT_MAX);

  EXPECT_TRUE(valid[0].item<bool>());

  // Nearest point to origin should be (1,0,0) at index 1
  EXPECT_EQ(1, closest_indices[0].item<int>());
  EXPECT_NEAR(1.0f, closest_points[0][0].item<float>(), 1e-5f);
}

TEST(TensorKdTree, FindClosestPointsMaxDist) {
  at::Tensor source = torch::tensor({{0.0f, 0.0f, 0.0f}});
  at::Tensor target = torch::tensor({{10.0f, 0.0f, 0.0f}});

  // maxDist = 5, so the point at distance 10 should not be found
  auto [closest_points, closest_indices, valid] =
      pymomentum::findClosestPoints(source, target, 5.0f);

  EXPECT_FALSE(valid[0].item<bool>());
  EXPECT_EQ(-1, closest_indices[0].item<int>());
}

TEST(TensorKdTree, FindClosestPointsOnMeshTriangle) {
  // Simple triangle in XY plane
  at::Tensor vertices = torch::tensor({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}});
  at::Tensor faces = torch::tensor({{0, 1, 2}}, at::kInt);

  // Query point directly above the triangle centroid
  at::Tensor source = torch::tensor({{0.25f, 0.25f, 1.0f}});

  auto [face_valid, closest_points, face_indices, barycoords] =
      pymomentum::findClosestPointsOnMesh(source, vertices, faces);

  EXPECT_TRUE(face_valid[0].item<bool>());
  EXPECT_EQ(0, face_indices[0].item<int>());

  // Closest point should be the projection onto the triangle plane (z=0)
  EXPECT_NEAR(0.25f, closest_points[0][0].item<float>(), 1e-4f);
  EXPECT_NEAR(0.25f, closest_points[0][1].item<float>(), 1e-4f);
  EXPECT_NEAR(0.0f, closest_points[0][2].item<float>(), 1e-4f);
}

// =============================================================================
// tensor_skinning.h tests (computeVertexNormals)
// =============================================================================

TEST(TensorSkinning, ComputeVertexNormalsSimpleTriangle) {
  // A triangle in the XY plane: vertices at (0,0,0), (1,0,0), (0,1,0)
  at::Tensor positions =
      torch::tensor({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}});
  at::Tensor triangles = torch::tensor({{0, 1, 2}}, at::kLong);

  at::Tensor normals = pymomentum::computeVertexNormals(positions, triangles);

  EXPECT_EQ(2, normals.ndimension());
  EXPECT_EQ(3, normals.size(0));
  EXPECT_EQ(3, normals.size(1));

  // All vertex normals should point in the +Z direction for this triangle
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(0.0f, normals[i][0].item<float>(), 1e-5f);
    EXPECT_NEAR(0.0f, normals[i][1].item<float>(), 1e-5f);
    EXPECT_NEAR(1.0f, normals[i][2].item<float>(), 1e-5f);
  }
}

TEST(TensorSkinning, ComputeVertexNormalsInvalidInput) {
  // vertex_positions must have at least 2 dimensions
  EXPECT_THROW(
      pymomentum::computeVertexNormals(at::zeros({3}), at::zeros({1, 3}, at::kLong)),
      std::runtime_error);

  // Last dim of vertex_positions must be 3
  EXPECT_THROW(
      pymomentum::computeVertexNormals(at::zeros({3, 4}), at::zeros({1, 3}, at::kLong)),
      std::runtime_error);

  // triangles must have shape [n_triangles, 3]
  EXPECT_THROW(
      pymomentum::computeVertexNormals(at::zeros({3, 3}), at::zeros({1, 4}, at::kLong)),
      std::runtime_error);
}

TEST(TensorSkinning, ComputeVertexNormalsNormalized) {
  // Quad split into two triangles
  at::Tensor positions = torch::tensor({
      {0.0f, 0.0f, 0.0f},
      {1.0f, 0.0f, 0.0f},
      {1.0f, 1.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
  });
  at::Tensor triangles = torch::tensor({{0, 1, 2}, {0, 2, 3}}, at::kLong);

  at::Tensor normals = pymomentum::computeVertexNormals(positions, triangles);

  // All normals should have unit length
  for (int i = 0; i < 4; ++i) {
    auto norm = normals[i].norm().item<float>();
    EXPECT_NEAR(1.0f, norm, 1e-5f);
  }
}
