/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character_solver/skeleton_only_solver_function.h>

#include <momentum/character/character.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/limit_error_function.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/skeleton_solver_function.h>
#include <momentum/character_solver/state_error_function.h>
#include <momentum/math/random.h>
#include <momentum/solver/gauss_newton_solver.h>
#include <momentum/test/character/character_helpers.h>

#include <gtest/gtest.h>

#include <limits>

using namespace momentum;

namespace {

template <typename T>
struct SkeletonOnlySolverFunctionTest : testing::Test {
  using Type = T;
  Random<> rng{12345};
};

using Types = testing::Types<float, double>;
TYPED_TEST_SUITE(SkeletonOnlySolverFunctionTest, Types);

// A state term on the last joint and a MinMax limit on every model parameter.
template <typename T>
std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>> makeTerms(
    const Character& character,
    const ParameterTransformT<T>& transform,
    const SkeletonStateT<T>& targetState) {
  auto state =
      std::make_shared<StateErrorFunctionT<T>>(character.skeleton, character.parameterTransform);
  Eigen::VectorX<T> weights = Eigen::VectorX<T>::Zero(character.skeleton.joints.size());
  weights(weights.size() - 1) = 1;
  state->setTargetWeights(weights, weights);
  state->setTargetState(targetState);
  ParameterLimits limits;
  for (Eigen::Index i = 0; i < transform.numAllModelParameters(); ++i) {
    ParameterLimit limit;
    limit.type = MinMax;
    limit.data.minMax.parameterIndex = static_cast<size_t>(i);
    limit.data.minMax.limits = Vector2f(-0.3f, 0.3f);
    limits.push_back(limit);
  }
  auto limit = std::make_shared<LimitErrorFunctionT<T>>(
      character.skeleton, character.parameterTransform, limits);
  return {state, limit};
}

} // namespace

// The skeleton-only function evaluates and solves as SkeletonSolverFunction with the same terms.
TYPED_TEST(SkeletonOnlySolverFunctionTest, MatchesSkeletonSolverFunction) {
  using T = typename TestFixture::Type;
  const Character character = createTestCharacter(5);
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const auto n = transform.numAllModelParameters();
  const ModelParametersT<T> target = this->rng.template uniform<VectorX<T>>(n, -0.5, 0.5);
  const SkeletonStateT<T> targetState(transform.apply(target), character.skeleton);

  SkeletonSolverFunctionT<T> reference(
      character, transform, makeTerms(character, transform, targetState));
  SkeletonOnlySolverFunctionT<T> skeletonOnly(character.skeleton, transform);
  for (auto& term : makeTerms(character, transform, targetState)) {
    skeletonOnly.addErrorFunction(term);
  }

  const auto parameters = this->rng.template uniform<VectorX<T>>(n, -0.5, 0.5);
  // SkeletonSolverFunction rounds its error to float.
  EXPECT_FLOAT_EQ(reference.getError(parameters), skeletonOnly.getError(parameters));
  Eigen::VectorX<T> referenceGradient;
  Eigen::VectorX<T> gradient;
  EXPECT_EQ(
      reference.getGradient(parameters, referenceGradient),
      skeletonOnly.getGradient(parameters, gradient));
  EXPECT_EQ(referenceGradient, gradient);

  GaussNewtonSolverOptions options;
  options.minIterations = 2;
  options.maxIterations = 10;
  GaussNewtonSolverT<T> referenceSolver(options, &reference);
  GaussNewtonSolverT<T> solver(options, &skeletonOnly);
  Eigen::VectorX<T> referenceSolved = parameters;
  Eigen::VectorX<T> solved = parameters;
  referenceSolver.solve(referenceSolved);
  solver.solve(solved);
  EXPECT_EQ(referenceSolved, solved);
}

// With a subset of the parameters enabled, it solves as SkeletonSolverFunction and leaves the rest.
TYPED_TEST(SkeletonOnlySolverFunctionTest, MatchesSkeletonSolverFunctionOnEnabledParameters) {
  using T = typename TestFixture::Type;
  const Character character = createTestCharacter(5);
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const auto n = transform.numAllModelParameters();
  const ModelParametersT<T> target = this->rng.template uniform<VectorX<T>>(n, -0.5, 0.5);
  const SkeletonStateT<T> targetState(transform.apply(target), character.skeleton);

  SkeletonSolverFunctionT<T> reference(
      character, transform, makeTerms(character, transform, targetState));
  SkeletonOnlySolverFunctionT<T> skeletonOnly(character.skeleton, transform);
  for (auto& term : makeTerms(character, transform, targetState)) {
    skeletonOnly.addErrorFunction(term);
  }
  ParameterSet enabled;
  for (Eigen::Index i = 0; i < n; ++i) {
    if (i == 0 || this->rng.template uniform<float>(0.0f, 1.0f) < 0.5f) {
      enabled.set(static_cast<size_t>(i));
    }
  }

  GaussNewtonSolverOptions options;
  options.minIterations = 2;
  options.maxIterations = 10;
  GaussNewtonSolverT<T> referenceSolver(options, &reference);
  GaussNewtonSolverT<T> solver(options, &skeletonOnly);
  referenceSolver.setEnabledParameters(enabled);
  solver.setEnabledParameters(enabled);
  const auto parameters = this->rng.template uniform<VectorX<T>>(n, -0.5, 0.5);
  Eigen::VectorX<T> referenceSolved = parameters;
  Eigen::VectorX<T> solved = parameters;
  referenceSolver.solve(referenceSolved);
  solver.solve(solved);
  EXPECT_EQ(referenceSolved, solved);
  for (Eigen::Index i = 0; i < n; ++i) {
    if (!enabled.test(static_cast<size_t>(i))) {
      EXPECT_EQ(parameters(i), solved(i));
    }
  }
  EXPECT_NE(parameters, solved);
}

// With bounds, every update ends on [lower, upper].
TYPED_TEST(SkeletonOnlySolverFunctionTest, UpdatesEndOnTheBounds) {
  using T = typename TestFixture::Type;
  const Character character = createTestCharacter(5);
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const auto n = transform.numAllModelParameters();
  SkeletonOnlySolverFunctionT<T> function(character.skeleton, transform);
  Eigen::VectorX<T> lower = Eigen::VectorX<T>::Constant(n, -0.1);
  Eigen::VectorX<T> upper = Eigen::VectorX<T>::Constant(n, 0.2);
  function.setParameterBounds(lower, upper);

  Eigen::VectorX<T> parameters = Eigen::VectorX<T>::Zero(n);
  const auto delta = this->rng.template uniform<VectorX<T>>(n, -1, 1);
  function.updateParameters(parameters, delta);
  EXPECT_EQ((-delta).cwiseMax(lower).cwiseMin(upper), parameters);
  EXPECT_THROW(function.setParameterBounds(lower.head(n - 1), upper.head(n - 1)), std::exception);
}

// Inverted or NaN bounds are refused; infinite bounds leave a parameter unbounded.
TYPED_TEST(SkeletonOnlySolverFunctionTest, RefusesInvalidBounds) {
  using T = typename TestFixture::Type;
  const Character character = createTestCharacter(5);
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const auto n = transform.numAllModelParameters();
  SkeletonOnlySolverFunctionT<T> function(character.skeleton, transform);
  const Eigen::VectorX<T> lower = Eigen::VectorX<T>::Constant(n, -0.1);
  const Eigen::VectorX<T> upper = Eigen::VectorX<T>::Constant(n, 0.2);

  EXPECT_THROW(function.setParameterBounds(upper, lower), std::exception);
  Eigen::VectorX<T> withNaN = upper;
  withNaN(0) = std::numeric_limits<T>::quiet_NaN();
  EXPECT_THROW(function.setParameterBounds(lower, withNaN), std::exception);

  Eigen::VectorX<T> unboundedLower = lower;
  Eigen::VectorX<T> unboundedUpper = upper;
  unboundedLower(0) = -std::numeric_limits<T>::infinity();
  unboundedUpper(0) = std::numeric_limits<T>::infinity();
  function.setParameterBounds(unboundedLower, unboundedUpper);
  Eigen::VectorX<T> parameters = Eigen::VectorX<T>::Zero(n);
  const Eigen::VectorX<T> delta = Eigen::VectorX<T>::Constant(n, 5);
  function.updateParameters(parameters, delta);
  Eigen::VectorX<T> expected = Eigen::VectorX<T>::Constant(n, -0.1);
  expected(0) = -5;
  EXPECT_EQ(expected, parameters);
}

// An error function that reads the mesh.
template <typename T>
struct MeshErrorFunction : SkeletonErrorFunctionT<T> {
  using SkeletonErrorFunctionT<T>::SkeletonErrorFunctionT;
  [[nodiscard]] bool needsMesh() const override {
    return true;
  }
};

// Error functions that read the mesh are refused.
TYPED_TEST(SkeletonOnlySolverFunctionTest, RefusesErrorFunctionsThatNeedAMesh) {
  using T = typename TestFixture::Type;
  const Character character = createTestCharacter(5);
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  SkeletonOnlySolverFunctionT<T> function(character.skeleton, transform);
  EXPECT_THROW(
      function.addErrorFunction(
          std::make_shared<MeshErrorFunction<T>>(character.skeleton, character.parameterTransform)),
      std::exception);
}
