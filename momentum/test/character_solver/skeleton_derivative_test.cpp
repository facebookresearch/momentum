/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character_solver/skeleton_derivative.h>

#include <momentum/character/character.h>
#include <momentum/character/linear_skinning.h>
#include <momentum/character/mesh_state.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/math/random.h>
#include <momentum/test/character/character_helpers.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>

namespace momentum {

namespace {

// Helper to create enabled parameters set (all enabled)
ParameterSet createEnabledParameters() {
  ParameterSet result;
  result.set();
  return result;
}

} // namespace

template <typename T>
class SkeletonDerivativeTest : public testing::Test {
 protected:
  void SetUp() override {
    character_ = createTestCharacter();
    enabledParams_ = createEnabledParameters();

    // Create random joint parameters
    const auto numJointParams = character_.skeleton.joints.size() * kParametersPerJoint;
    JointParametersT<T> jointParams = JointParametersT<T>::Zero(numJointParams);

    // Set some non-zero values
    for (Eigen::Index i = 0; i < jointParams.size(); ++i) {
      jointParams.v(i) = static_cast<T>(i % 7) * T(0.1);
    }

    // Create skeleton state from joint parameters
    state_ = SkeletonStateT<T>(jointParams, character_.skeleton, true);
  }

  // Helper: compute numerical gradient of f(x) = 0.5 * ||v(x) - target||^2
  // where v(x) is a world-space vector attached to jointIdx via localOffset.
  template <size_t FuncDim>
  void verifyGradientVsFiniteDifference(
      size_t jointIdx,
      const Eigen::Vector3<T>& localOffset,
      const Eigen::Matrix<T, FuncDim, 3>& dfdv,
      bool isPoint) {
    const auto& skeleton = character_.skeleton;
    const auto& floatTransform = character_.parameterTransform;
    const ParameterTransformT<T> transform = floatTransform.template cast<T>();
    const auto numParams = transform.numAllModelParameters();

    SkeletonDerivativeT<T> deriv(
        skeleton, floatTransform, floatTransform.activeJointParams, enabledParams_);

    // Reference model parameters (small non-zero values)
    ModelParametersT<T> refParams = ModelParametersT<T>::Zero(numParams);
    for (Eigen::Index i = 0; i < refParams.size(); ++i) {
      refParams(i) = static_cast<T>(i % 5) * T(0.05);
    }

    SkeletonStateT<T> refState(transform.apply(refParams), skeleton, true);
    const auto& refJointState = refState.jointState[jointIdx];

    // Compute world-space vector
    Eigen::Vector3<T> worldVec;
    if (isPoint) {
      worldVec = refJointState.transform * localOffset;
    } else {
      worldVec = refJointState.rotation() * localOffset;
    }

    // Target (shifted so residual is non-zero)
    const Eigen::Matrix<T, FuncDim, 1> fVal = dfdv * worldVec;
    const Eigen::Matrix<T, FuncDim, 1> target =
        fVal + Eigen::Matrix<T, FuncDim, 1>::Constant(T(0.1));
    const Eigen::Matrix<T, FuncDim, 1> residual = fVal - target;

    // Analytical gradient
    std::array<Eigen::Vector3<T>, 1> worldVecs = {worldVec};
    std::array<Eigen::Matrix<T, FuncDim, 3>, 1> dfdvs = {dfdv};
    std::array<uint8_t, 1> isPoints = {isPoint ? uint8_t(1) : uint8_t(0)};

    Eigen::VectorX<T> anaGradient = Eigen::VectorX<T>::Zero(numParams);
    deriv.template accumulateJointGradient<FuncDim>(
        jointIdx,
        std::span<const Eigen::Vector3<T>>(worldVecs),
        std::span<const Eigen::Matrix<T, FuncDim, 3>>(dfdvs),
        std::span<const uint8_t>(isPoints),
        residual,
        refState.jointState,
        anaGradient);

    // Numerical gradient via central differences
    const T kStepSize = Eps<T>(1e-4f, 1e-7);
    Eigen::VectorX<T> numGradient = Eigen::VectorX<T>::Zero(numParams);
    for (Eigen::Index p = 0; p < numParams; ++p) {
      auto evalError = [&](T delta) -> T {
        ModelParametersT<T> params = refParams;
        params(p) += delta;
        SkeletonStateT<T> perturbedState(transform.apply(params), skeleton, true);
        const auto& pertJoint = perturbedState.jointState[jointIdx];
        Eigen::Vector3<T> pertVec;
        if (isPoint) {
          pertVec = pertJoint.transform * localOffset;
        } else {
          pertVec = pertJoint.rotation() * localOffset;
        }
        const Eigen::Matrix<T, FuncDim, 1> f = dfdv * pertVec;
        return T(0.5) * (f - target).squaredNorm();
      };
      numGradient(p) = (evalError(kStepSize) - evalError(-kStepSize)) / (T(2) * kStepSize);
    }

    // Compare with relative error
    const T kTolerance = Eps<T>(5e-3f, 1e-4);
    EXPECT_GT(anaGradient.squaredNorm(), T(0)) << "Analytical gradient is zero";
    if ((numGradient + anaGradient).norm() > T(0)) {
      const T relError = (numGradient - anaGradient).norm() /
          ((numGradient + anaGradient).norm() + std::numeric_limits<T>::epsilon());
      EXPECT_LE(relError, kTolerance)
          << "Numerical gradient norm: " << numGradient.norm()
          << ", Analytical gradient norm: " << anaGradient.norm()
          << ", Difference norm: " << (numGradient - anaGradient).norm();
    }
  }

  // Helper: verify Jacobian against finite differences
  template <size_t FuncDim>
  void verifyJacobianVsFiniteDifference(
      size_t jointIdx,
      const Eigen::Vector3<T>& localOffset,
      const Eigen::Matrix<T, FuncDim, 3>& dfdv,
      bool isPoint) {
    const auto& skeleton = character_.skeleton;
    const auto& floatTransform = character_.parameterTransform;
    const ParameterTransformT<T> transform = floatTransform.template cast<T>();
    const auto numParams = transform.numAllModelParameters();

    SkeletonDerivativeT<T> deriv(
        skeleton, floatTransform, floatTransform.activeJointParams, enabledParams_);

    ModelParametersT<T> refParams = ModelParametersT<T>::Zero(numParams);
    for (Eigen::Index i = 0; i < refParams.size(); ++i) {
      refParams(i) = static_cast<T>(i % 5) * T(0.05);
    }

    SkeletonStateT<T> refState(transform.apply(refParams), skeleton, true);
    const auto& refJointState = refState.jointState[jointIdx];

    Eigen::Vector3<T> worldVec;
    if (isPoint) {
      worldVec = refJointState.transform * localOffset;
    } else {
      worldVec = refJointState.rotation() * localOffset;
    }

    // Analytical Jacobian
    std::array<Eigen::Vector3<T>, 1> worldVecs = {worldVec};
    std::array<Eigen::Matrix<T, FuncDim, 3>, 1> dfdvs = {dfdv};
    std::array<uint8_t, 1> isPoints = {isPoint ? uint8_t(1) : uint8_t(0)};

    Eigen::MatrixX<T> anaJacobian = Eigen::MatrixX<T>::Zero(FuncDim, numParams);
    deriv.template accumulateJointJacobian<FuncDim>(
        jointIdx,
        std::span<const Eigen::Vector3<T>>(worldVecs),
        std::span<const Eigen::Matrix<T, FuncDim, 3>>(dfdvs),
        std::span<const uint8_t>(isPoints),
        T(1),
        refState.jointState,
        anaJacobian,
        0);

    // Compute residual at reference point: f(x) = dfdv * v(x)
    const Eigen::Matrix<T, FuncDim, 1> refResidual = dfdv * worldVec;

    // Numerical Jacobian via forward differences
    const T kStepSize = Eps<T>(1e-4f, 1e-7);
    Eigen::MatrixX<T> numJacobian = Eigen::MatrixX<T>::Zero(FuncDim, numParams);
    for (Eigen::Index p = 0; p < numParams; ++p) {
      ModelParametersT<T> params = refParams;
      params(p) += kStepSize;
      SkeletonStateT<T> perturbedState(transform.apply(params), skeleton, true);
      const auto& pertJoint = perturbedState.jointState[jointIdx];
      Eigen::Vector3<T> pertVec;
      if (isPoint) {
        pertVec = pertJoint.transform * localOffset;
      } else {
        pertVec = pertJoint.rotation() * localOffset;
      }
      numJacobian.col(p) = (dfdv * pertVec - refResidual) / kStepSize;
    }

    // Compare with relative error
    const T kTolerance = Eps<T>(5e-3f, 1e-4);
    EXPECT_GT(anaJacobian.squaredNorm(), T(0)) << "Analytical Jacobian is zero";
    if ((numJacobian + anaJacobian).norm() > T(0)) {
      const T relError = (numJacobian - anaJacobian).norm() /
          ((numJacobian + anaJacobian).norm() + std::numeric_limits<T>::epsilon());
      EXPECT_LE(relError, kTolerance)
          << "Numerical Jacobian norm: " << numJacobian.norm()
          << ", Analytical Jacobian norm: " << anaJacobian.norm()
          << ", Difference norm: " << (numJacobian - anaJacobian).norm();
    }
  }

  static constexpr T kEps = Eps<T>(1e-5f, 1e-9);

  Character character_;
  SkeletonStateT<T> state_;
  ParameterSet enabledParams_;
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SkeletonDerivativeTest, ScalarTypes);

// Test that the utility can be constructed without crashing
TYPED_TEST(SkeletonDerivativeTest, Construction) {
  using T = TypeParam;

  SkeletonDerivativeT<T> deriv(
      this->character_.skeleton,
      this->character_.parameterTransform,
      this->character_.parameterTransform.activeJointParams,
      this->enabledParams_);

  SUCCEED();
}

// Test that zero derivatives result in zero gradient contribution
TYPED_TEST(SkeletonDerivativeTest, ZeroDerivativeSkipped) {
  using T = TypeParam;

  constexpr size_t FuncDim = 3;
  const auto numParams = this->character_.parameterTransform.numAllModelParameters();

  SkeletonDerivativeT<T> deriv(
      this->character_.skeleton,
      this->character_.parameterTransform,
      this->character_.parameterTransform.activeJointParams,
      this->enabledParams_);

  const size_t jointIdx = 0;
  const auto& jointState = this->state_.jointState[jointIdx];

  const Eigen::Vector3<T> worldVec = jointState.transform * Eigen::Vector3<T>::UnitX();

  // Zero derivative
  const Eigen::Matrix<T, FuncDim, 3> dfdv = Eigen::Matrix<T, FuncDim, 3>::Zero();

  Eigen::Matrix<T, FuncDim, 1> weightedResidual;
  weightedResidual.setOnes();

  std::array<Eigen::Vector3<T>, 1> worldVecs = {worldVec};
  std::array<Eigen::Matrix<T, FuncDim, 3>, 1> dfdvs = {dfdv};
  std::array<uint8_t, 1> isPoints = {1};

  Eigen::VectorX<T> gradient = Eigen::VectorX<T>::Zero(numParams);

  deriv.template accumulateJointGradient<FuncDim>(
      jointIdx,
      std::span<const Eigen::Vector3<T>>(worldVecs),
      std::span<const Eigen::Matrix<T, FuncDim, 3>>(dfdvs),
      std::span<const uint8_t>(isPoints),
      weightedResidual,
      this->state_.jointState,
      gradient);

  // Gradient should remain zero
  EXPECT_NEAR(gradient.squaredNorm(), T(0), this->kEps);
}

// Test empty span (no vectors) doesn't crash
TYPED_TEST(SkeletonDerivativeTest, EmptySpan) {
  using T = TypeParam;

  constexpr size_t FuncDim = 3;
  const auto numParams = this->character_.parameterTransform.numAllModelParameters();

  SkeletonDerivativeT<T> deriv(
      this->character_.skeleton,
      this->character_.parameterTransform,
      this->character_.parameterTransform.activeJointParams,
      this->enabledParams_);

  const size_t jointIdx = 0;

  Eigen::Matrix<T, FuncDim, 1> weightedResidual;
  weightedResidual.setOnes();

  Eigen::VectorX<T> gradient = Eigen::VectorX<T>::Zero(numParams);

  // Empty spans
  deriv.template accumulateJointGradient<FuncDim>(
      jointIdx,
      std::span<const Eigen::Vector3<T>>(),
      std::span<const Eigen::Matrix<T, FuncDim, 3>>(),
      std::span<const uint8_t>(),
      weightedResidual,
      this->state_.jointState,
      gradient);

  // Should not crash, gradient remains zero
  EXPECT_NEAR(gradient.squaredNorm(), T(0), this->kEps);
}

// Gradient finite-difference tests for each FuncDim
TYPED_TEST(SkeletonDerivativeTest, GradientFiniteDifference_Point_FuncDim3) {
  using T = TypeParam;
  this->template verifyGradientVsFiniteDifference<3>(
      0, Eigen::Vector3<T>::UnitX(), Eigen::Matrix<T, 3, 3>::Identity(), true);
}

TYPED_TEST(SkeletonDerivativeTest, GradientFiniteDifference_Point_FuncDim1) {
  using T = TypeParam;
  const Eigen::Matrix<T, 1, 3> dfdv = Eigen::RowVector3<T>(0, 1, 0);
  this->template verifyGradientVsFiniteDifference<1>(0, Eigen::Vector3<T>::UnitZ(), dfdv, true);
}

TYPED_TEST(SkeletonDerivativeTest, GradientFiniteDifference_Point_FuncDim2) {
  using T = TypeParam;
  Eigen::Matrix<T, 2, 3> dfdv;
  dfdv << T(1), T(0), T(0), T(0), T(0), T(1);
  this->template verifyGradientVsFiniteDifference<2>(
      0, Eigen::Vector3<T>(1, 2, 3).normalized(), dfdv, true);
}

TYPED_TEST(SkeletonDerivativeTest, GradientFiniteDifference_Direction_FuncDim9) {
  using T = TypeParam;
  Eigen::Matrix<T, 9, 3> dfdv;
  dfdv.setIdentity(); // top 3x3 = I, rest = 0 but enough for testing
  this->template verifyGradientVsFiniteDifference<9>(0, Eigen::Vector3<T>::UnitX(), dfdv, false);
}

// Jacobian finite-difference tests
TYPED_TEST(SkeletonDerivativeTest, JacobianFiniteDifference_Point_FuncDim3) {
  using T = TypeParam;
  this->template verifyJacobianVsFiniteDifference<3>(
      0, Eigen::Vector3<T>::UnitX(), Eigen::Matrix<T, 3, 3>::Identity(), true);
}

// Test batched vectors (combined fast path): verify gradient matches sum of individual gradients
TYPED_TEST(SkeletonDerivativeTest, BatchedVectorsMatchIndividual) {
  using T = TypeParam;

  constexpr size_t FuncDim = 3;
  const auto& skeleton = this->character_.skeleton;
  const auto& floatTransform = this->character_.parameterTransform;
  const ParameterTransformT<T> transform = floatTransform.template cast<T>();
  const auto numParams = transform.numAllModelParameters();

  SkeletonDerivativeT<T> deriv(
      skeleton, floatTransform, floatTransform.activeJointParams, this->enabledParams_);

  ModelParametersT<T> refParams = ModelParametersT<T>::Zero(numParams);
  for (Eigen::Index i = 0; i < refParams.size(); ++i) {
    refParams(i) = static_cast<T>(i % 5) * T(0.05);
  }
  SkeletonStateT<T> refState(transform.apply(refParams), skeleton, true);

  const size_t jointIdx = 0;
  const auto& jointState = refState.jointState[jointIdx];

  // Two vectors: one point and one direction
  const Eigen::Vector3<T> worldPoint = jointState.transform * Eigen::Vector3<T>::UnitX();
  const Eigen::Vector3<T> worldDir = jointState.rotation() * Eigen::Vector3<T>::UnitY();

  const Eigen::Matrix<T, FuncDim, 3> dfdvI = Eigen::Matrix<T, FuncDim, 3>::Identity();
  Eigen::Matrix<T, FuncDim, 1> weightedResidual;
  weightedResidual.setOnes();

  // Batched call
  std::array<Eigen::Vector3<T>, 2> batchVecs = {worldPoint, worldDir};
  std::array<Eigen::Matrix<T, FuncDim, 3>, 2> batchDfdvs = {dfdvI, dfdvI};
  std::array<uint8_t, 2> batchIsPoints = {1, 0};

  Eigen::VectorX<T> batchedGradient = Eigen::VectorX<T>::Zero(numParams);
  deriv.template accumulateJointGradient<FuncDim>(
      jointIdx,
      std::span<const Eigen::Vector3<T>>(batchVecs),
      std::span<const Eigen::Matrix<T, FuncDim, 3>>(batchDfdvs),
      std::span<const uint8_t>(batchIsPoints),
      weightedResidual,
      refState.jointState,
      batchedGradient);

  // Individual calls
  Eigen::VectorX<T> sumGradient = Eigen::VectorX<T>::Zero(numParams);

  std::array<Eigen::Vector3<T>, 1> vec1 = {worldPoint};
  std::array<Eigen::Matrix<T, FuncDim, 3>, 1> dfdv1 = {dfdvI};
  std::array<uint8_t, 1> isP1 = {1};
  deriv.template accumulateJointGradient<FuncDim>(
      jointIdx,
      std::span<const Eigen::Vector3<T>>(vec1),
      std::span<const Eigen::Matrix<T, FuncDim, 3>>(dfdv1),
      std::span<const uint8_t>(isP1),
      weightedResidual,
      refState.jointState,
      sumGradient);

  std::array<Eigen::Vector3<T>, 1> vec2 = {worldDir};
  std::array<Eigen::Matrix<T, FuncDim, 3>, 1> dfdv2 = {dfdvI};
  std::array<uint8_t, 1> isP2 = {0};
  deriv.template accumulateJointGradient<FuncDim>(
      jointIdx,
      std::span<const Eigen::Vector3<T>>(vec2),
      std::span<const Eigen::Matrix<T, FuncDim, 3>>(dfdv2),
      std::span<const uint8_t>(isP2),
      weightedResidual,
      refState.jointState,
      sumGradient);

  // Batched should match sum of individual
  EXPECT_GT(batchedGradient.squaredNorm(), T(0));
  EXPECT_NEAR(
      (batchedGradient - sumGradient).norm(), T(0), Eps<T>(1e-5f, 1e-12) * sumGradient.norm());
}

// Vertex gradient: verify against finite differences
TEST(SkeletonDerivativeVertexTest, VertexGradientFiniteDifference) {
  constexpr size_t FuncDim = 3;
  const Character character = createTestCharacter();

  if (!character.mesh || character.mesh->vertices.empty()) {
    GTEST_SKIP() << "Test character has no mesh data";
  }

  const auto numParams = character.parameterTransform.numAllModelParameters();
  ParameterSet enabledParams;
  enabledParams.set();

  SkeletonDerivativeT<float> deriv(
      character.skeleton,
      character.parameterTransform,
      character.parameterTransform.activeJointParams,
      enabledParams);

  const size_t vertexIndex = 0;

  // Reference parameters
  ModelParameters refParams = ModelParameters::Zero(numParams);
  for (Eigen::Index i = 0; i < refParams.size(); ++i) {
    refParams(i) = static_cast<float>(i % 5) * 0.05f;
  }
  const SkeletonState refSkelState(
      character.parameterTransform.apply(refParams), character.skeleton, true);
  const MeshState refMeshState(refParams, refSkelState, character);

  const Eigen::Vector3f worldPos = refMeshState.posedMesh_->vertices[vertexIndex];
  const Eigen::Vector3f target = worldPos + Eigen::Vector3f(0.1f, -0.2f, 0.05f);
  const Eigen::Vector3f residual = worldPos - target;

  Eigen::Matrix<float, FuncDim, 3> dfdv;
  dfdv.setIdentity();

  // Analytical gradient
  Eigen::VectorXf anaGradient = Eigen::VectorXf::Zero(numParams);
  deriv.accumulateVertexGradient<FuncDim>(
      vertexIndex, worldPos, dfdv, residual, refSkelState, refMeshState, character, anaGradient);

  // Numerical gradient
  constexpr float kStepSize = 1e-4f;
  Eigen::VectorXf numGradient = Eigen::VectorXf::Zero(numParams);
  for (Eigen::Index p = 0; p < numParams; ++p) {
    auto evalError = [&](float delta) -> float {
      ModelParameters params = refParams;
      params(p) += delta;
      SkeletonState pertSkelState(
          character.parameterTransform.apply(params), character.skeleton, true);
      MeshState pertMeshState(params, pertSkelState, character);
      const Eigen::Vector3f pertPos = pertMeshState.posedMesh_->vertices[vertexIndex];
      return 0.5f * (pertPos - target).squaredNorm();
    };
    numGradient(p) = (evalError(kStepSize) - evalError(-kStepSize)) / (2.0f * kStepSize);
  }

  EXPECT_GT(anaGradient.squaredNorm(), 0.0f) << "Analytical gradient is zero";
  if ((numGradient + anaGradient).norm() > 0.0f) {
    const float relError = (numGradient - anaGradient).norm() /
        ((numGradient + anaGradient).norm() + std::numeric_limits<float>::epsilon());
    EXPECT_LE(relError, 5e-3f) << "Numerical gradient norm: " << numGradient.norm()
                               << ", Analytical gradient norm: " << anaGradient.norm()
                               << ", Difference norm: " << (numGradient - anaGradient).norm();
  }
}

// Vertex Jacobian: verify against finite differences
TEST(SkeletonDerivativeVertexTest, VertexJacobianFiniteDifference) {
  constexpr size_t FuncDim = 3;
  const Character character = createTestCharacter();

  if (!character.mesh || character.mesh->vertices.empty()) {
    GTEST_SKIP() << "Test character has no mesh data";
  }

  const auto numParams = character.parameterTransform.numAllModelParameters();
  ParameterSet enabledParams;
  enabledParams.set();

  SkeletonDerivativeT<float> deriv(
      character.skeleton,
      character.parameterTransform,
      character.parameterTransform.activeJointParams,
      enabledParams);

  const size_t vertexIndex = 0;

  ModelParameters refParams = ModelParameters::Zero(numParams);
  for (Eigen::Index i = 0; i < refParams.size(); ++i) {
    refParams(i) = static_cast<float>(i % 5) * 0.05f;
  }
  const SkeletonState refSkelState(
      character.parameterTransform.apply(refParams), character.skeleton, true);
  const MeshState refMeshState(refParams, refSkelState, character);

  const Eigen::Vector3f worldPos = refMeshState.posedMesh_->vertices[vertexIndex];

  Eigen::Matrix<float, FuncDim, 3> dfdv;
  dfdv.setIdentity();

  // Analytical Jacobian
  Eigen::MatrixXf anaJacobian = Eigen::MatrixXf::Zero(FuncDim, numParams);
  deriv.accumulateVertexJacobian<FuncDim>(
      vertexIndex, worldPos, dfdv, 1.0f, refSkelState, refMeshState, character, anaJacobian, 0);

  // Reference residual
  const Eigen::Vector3f refResidual = dfdv * worldPos;

  // Numerical Jacobian
  constexpr float kStepSize = 1e-4f;
  Eigen::MatrixXf numJacobian = Eigen::MatrixXf::Zero(FuncDim, numParams);
  for (Eigen::Index p = 0; p < numParams; ++p) {
    ModelParameters params = refParams;
    params(p) += kStepSize;
    SkeletonState pertSkelState(
        character.parameterTransform.apply(params), character.skeleton, true);
    MeshState pertMeshState(params, pertSkelState, character);
    const Eigen::Vector3f pertPos = pertMeshState.posedMesh_->vertices[vertexIndex];
    numJacobian.col(p) = (dfdv * pertPos - refResidual) / kStepSize;
  }

  EXPECT_GT(anaJacobian.squaredNorm(), 0.0f) << "Analytical Jacobian is zero";
  if ((numJacobian + anaJacobian).norm() > 0.0f) {
    const float relError = (numJacobian - anaJacobian).norm() /
        ((numJacobian + anaJacobian).norm() + std::numeric_limits<float>::epsilon());
    EXPECT_LE(relError, 5e-3f) << "Numerical Jacobian norm: " << numJacobian.norm()
                               << ", Analytical Jacobian norm: " << anaJacobian.norm()
                               << ", Difference norm: " << (numJacobian - anaJacobian).norm();
  }
}

} // namespace momentum
