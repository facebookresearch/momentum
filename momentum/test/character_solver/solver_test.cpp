/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/character.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/types.h"
#include "momentum/character_solver/gauss_newton_solver_qr.h"
#include "momentum/character_solver/limit_error_function.h"
#include "momentum/character_solver/model_parameters_error_function.h"
#include "momentum/character_solver/orientation_error_function.h"
#include "momentum/character_solver/position_error_function.h"
#include "momentum/character_solver/transform_pose.h"
#include "momentum/character_solver/trust_region_qr.h"
#include "momentum/math/fmt_eigen.h"
#include "momentum/math/random.h"
#include "momentum/math/types.h"
#include "momentum/solver/gauss_newton_solver.h"
#include "momentum/solver/subset_gauss_newton_solver.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/solver/solver_test_helpers.h"

#include <gtest/gtest.h>

#include <chrono>
#include <limits>

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct GaussNewtonQRTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(GaussNewtonQRTest, Types);

TYPED_TEST(GaussNewtonQRTest, CompareGaussNewton) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& parameterTransform = character.parameterTransform;

  SkeletonState initialBodySkeletonState(parameterTransform.zero(), skeleton);

  const Eigen::VectorX<T> parametersInit =
      Eigen::VectorX<T>::Zero(parameterTransform.numAllModelParameters());

  const size_t nFrames = 10;

  std::vector<Eigen::VectorX<T>> targetParams;

  SolverOptions solverOptions;
  solverOptions.minIterations = 2;
  solverOptions.maxIterations = 10;

  auto subsetGaussNewtonSolverOptions = SubsetGaussNewtonSolverOptions(solverOptions);
  subsetGaussNewtonSolverOptions.doLineSearch = true;
  subsetGaussNewtonSolverOptions.regularization = 0.05;

  auto gaussNewtonSolverOptions = GaussNewtonSolverOptions(solverOptions);
  gaussNewtonSolverOptions.doLineSearch = true;
  gaussNewtonSolverOptions.regularization = 0.05;

  auto gaussNewtonSolverQROptions = GaussNewtonSolverQROptions(solverOptions);
  gaussNewtonSolverQROptions.doLineSearch = true;
  gaussNewtonSolverQROptions.regularization = 0.05;

  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    SkeletonSolverFunctionT<T> solverFunction(character, castedCharacterParameterTransform);

    ModelParametersT<T> randomParams_cur =
        VectorX<T>::Random(parameterTransform.numAllModelParameters());

    targetParams.push_back(randomParams_cur.v);

    auto positionErrorFunction =
        std::make_shared<PositionErrorFunctionT<T>>(skeleton, parameterTransform);
    auto orientErrorFunction =
        std::make_shared<OrientationErrorFunctionT<T>>(skeleton, parameterTransform);
    SkeletonStateT<T> skelState_cur(
        castedCharacterParameterTransform.apply(randomParams_cur), character.skeleton);
    for (size_t iJoint = 0; iJoint < skelState_cur.jointState.size(); ++iJoint) {
      positionErrorFunction->addConstraint(PositionDataT<T>(
          Eigen::Vector3<T>::Zero(),
          skelState_cur.jointState[iJoint].translation().template cast<T>(),
          iJoint,
          1.0));
      orientErrorFunction->addConstraint(OrientationDataT<T>(
          Eigen::Quaternion<T>::Identity(),
          skelState_cur.jointState[iJoint].rotation().template cast<T>(),
          iJoint,
          1.0));
    }

    solverFunction.addErrorFunction(positionErrorFunction);
    solverFunction.addErrorFunction(orientErrorFunction);

    SubsetGaussNewtonSolverT<T> solver_sub(subsetGaussNewtonSolverOptions, &solverFunction);
    const T err_sub = test::checkAndTimeSolver<T>(solverFunction, solver_sub, parametersInit);
    GaussNewtonSolverOptions options(test::defaultSolverOptions());
    options.useBlockJtJ = true;
    GaussNewtonSolverT<T> solver_gn(gaussNewtonSolverOptions, &solverFunction);
    const T err_gn = test::checkAndTimeSolver<T>(solverFunction, solver_gn, parametersInit);
    GaussNewtonSolverQRT<T> solver_qr(gaussNewtonSolverQROptions, &solverFunction);
    const T err_qr = test::checkAndTimeSolver<T>(solverFunction, solver_qr, parametersInit);

    // QR solver should do at least as well as Gauss-Newton.
    EXPECT_LE(err_qr, (T(1.001) * err_gn + T(0.001)));
    EXPECT_LE(err_sub, (T(1.001) * err_gn + T(0.001)));
  }
}

template <typename T>
struct TrustRegionTest : public testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(TrustRegionTest, Types);

TYPED_TEST(TrustRegionTest, PerfectQuadratic) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransform& parameterTransform = character.parameterTransform;

  const Eigen::VectorX<T> parametersInit =
      Eigen::VectorX<T>::Zero(parameterTransform.numAllModelParameters());

  const size_t nFrames = 10;

  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    SkeletonSolverFunctionT<T> solverFunction(character, castedCharacterParameterTransform);

    ModelParametersT<T> randomParams_cur =
        VectorX<T>::Random(parameterTransform.numAllModelParameters());

    auto mpErrorFunction =
        std::make_shared<ModelParametersErrorFunctionT<T>>(skeleton, parameterTransform);
    Eigen::VectorX<T> weights =
        VectorX<T>::Random(parameterTransform.numAllModelParameters()).array().abs();
    mpErrorFunction->setTargetParameters(randomParams_cur, weights);
    solverFunction.addErrorFunction(mpErrorFunction);

    TrustRegionQRT<T> solver_tr(test::defaultSolverOptions(), &solverFunction);
    const T err_tr = test::checkAndTimeSolver<T>(solverFunction, solver_tr, parametersInit);
    GaussNewtonSolverOptions options(test::defaultSolverOptions());
    options.useBlockJtJ = true;
    GaussNewtonSolverT<T> solver_gn(test::defaultSolverOptions(), &solverFunction);
    const T err_gn = test::checkAndTimeSolver<T>(solverFunction, solver_gn, parametersInit);

    // Trust Region solver should do at least as well as Gauss-Newton.
    EXPECT_LE(err_tr, (1.001 * err_gn + 0.001));
  }
}

TYPED_TEST(TrustRegionTest, SanityCheck) {
  using T = typename TestFixture::Type;

  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();
  const Skeleton& skeleton = character.skeleton;

  SkeletonState initialBodySkeletonState(character.parameterTransform.zero(), skeleton);

  const Eigen::VectorX<T> parametersInit =
      Eigen::VectorX<T>::Zero(character.parameterTransform.numAllModelParameters());

  const size_t nFrames = 10;

  std::vector<Eigen::VectorX<T>> targetParams;

  for (size_t iFrame = 0; iFrame < nFrames; ++iFrame) {
    SkeletonSolverFunctionT<T> solverFunction(character, castedCharacterParameterTransform);

    ModelParametersT<T> randomParams_cur =
        VectorX<T>::Random(character.parameterTransform.numAllModelParameters());

    targetParams.push_back(randomParams_cur.v);

    auto positionErrorFunction =
        std::make_shared<PositionErrorFunctionT<T>>(skeleton, character.parameterTransform);
    auto orientErrorFunction =
        std::make_shared<OrientationErrorFunctionT<T>>(skeleton, character.parameterTransform);
    SkeletonStateT<T> skelState_cur(
        castedCharacterParameterTransform.apply(randomParams_cur), character.skeleton);
    for (size_t iJoint = 0; iJoint < skelState_cur.jointState.size(); ++iJoint) {
      positionErrorFunction->addConstraint(PositionDataT<T>(
          Eigen::Vector3<T>::Zero(),
          skelState_cur.jointState[iJoint].translation().template cast<T>(),
          iJoint,
          1.0));
      orientErrorFunction->addConstraint(OrientationDataT<T>(
          Eigen::Quaternion<T>::Identity(),
          skelState_cur.jointState[iJoint].rotation().template cast<T>(),
          iJoint,
          1.0));
    }

    solverFunction.addErrorFunction(positionErrorFunction);
    solverFunction.addErrorFunction(orientErrorFunction);

    TrustRegionQRT<T> solver_tr(test::defaultSolverOptions(), &solverFunction);
    const T err_tr = test::checkAndTimeSolver<T>(solverFunction, solver_tr, parametersInit);
    GaussNewtonSolverQRT<T> solver_qr(test::defaultSolverOptions(), &solverFunction);
    const T err_qr = test::checkAndTimeSolver<T>(solverFunction, solver_qr, parametersInit);
    GaussNewtonSolverOptions options(test::defaultSolverOptions());
    options.useBlockJtJ = true;
    GaussNewtonSolverT<T> solver_gn(options, &solverFunction);
    const T err_gn = test::checkAndTimeSolver<T>(solverFunction, solver_gn, parametersInit);

    // Trust Region solver and QR solver should do at least as well as Gauss-Newton.
    EXPECT_LE(err_tr, (1.001 * err_gn + 0.001));
    EXPECT_LE(err_qr, (1.001 * err_gn + 0.001));
  }
}

template <typename T>
struct TransformPoseTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(TransformPoseTest, Types);

// Create a character where the transform parameters are divided between two joints:
[[nodiscard]] Character createDividedRootCharacter() {
  Skeleton skel;

  auto createJoint = [&](const char* name, Eigen::Quaternionf preRot, Eigen::Vector3f trans) {
    Joint joint;
    joint.name = name;
    joint.parent = (skel.joints.empty() ? kInvalidIndex : skel.joints.size() - 1);
    joint.preRotation = preRot;
    joint.translationOffset = trans;
    const size_t jointIndex = skel.joints.size();
    skel.joints.push_back(joint);
    return jointIndex;
  };

  const auto worldJoint =
      createJoint("root", Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero());
  const auto rootJoint =
      createJoint("world", Eigen::Quaternionf::Identity(), Eigen::Vector3f::UnitY());
  const auto childJoint =
      createJoint("child", Eigen::Quaternionf::Identity(), Eigen::Vector3f::UnitX());

  ParameterTransform paramTransform;
  const auto numJointParameters = skel.joints.size() * kParametersPerJoint;
  paramTransform.name = {
      {"root_tx",
       "root_ty",
       "root_tz",
       "root_rx",
       "root_ry",
       "root_rz",
       "scale_global",
       "joint1_rx"}};

  paramTransform.offsets = Eigen::VectorXf::Zero(numJointParameters);
  paramTransform.transform.resize(numJointParameters, static_cast<int>(paramTransform.name.size()));

  std::vector<Eigen::Triplet<float>> triplets;
  triplets.push_back(
      Eigen::Triplet<float>(worldJoint * kParametersPerJoint + 0, 0, 1.0f)); // root_tx
  triplets.push_back(
      Eigen::Triplet<float>(rootJoint * kParametersPerJoint + 1, 1, 1.0f)); // root_ty
  triplets.push_back(
      Eigen::Triplet<float>(rootJoint * kParametersPerJoint + 2, 2, 1.0f)); // root_tz
  triplets.push_back(
      Eigen::Triplet<float>(rootJoint * kParametersPerJoint + 3, 3, 1.0f)); // root_rx
  triplets.push_back(
      Eigen::Triplet<float>(worldJoint * kParametersPerJoint + 4, 4, 1.0f)); // root_ry
  triplets.push_back(
      Eigen::Triplet<float>(rootJoint * kParametersPerJoint + 5, 5, 1.0f)); // root_rz
  triplets.push_back(
      Eigen::Triplet<float>(worldJoint * kParametersPerJoint + 6, 6, 1.0f)); // scale_global
  triplets.push_back(
      Eigen::Triplet<float>(childJoint * kParametersPerJoint + 3, 7, 1.0f)); // joint1_rx

  paramTransform.transform.setFromTriplets(triplets.begin(), triplets.end());
  paramTransform.activeJointParams = paramTransform.computeActiveJointParams();

  return {skel, paramTransform};
}

TYPED_TEST(TransformPoseTest, ValidateTransformSimple) {
  using T = typename TestFixture::Type;
  Random<> rng{12345};

  const Character character = createTestCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();
  const Skeleton& skeleton = character.skeleton;

  SkeletonState initialBodySkeletonState(character.parameterTransform.zero(), skeleton);

  ModelParametersT<T> randomParams_init =
      rng.uniform<VectorX<T>>(character.parameterTransform.numAllModelParameters(), -1.0, 1.0);

  std::vector<TransformT<T>> transforms;
  transforms.emplace_back(
      Vector3<T>::Zero(), Quaternion<T>(AngleAxis<T>(pi<T>(), Vector3<T>::UnitY())));
  transforms.emplace_back(
      Vector3<T>(5, 15, 3), Quaternion<T>(AngleAxis<T>(pi<T>(), Vector3<T>::UnitX())));
  transforms.emplace_back(
      Vector3<T>(205, -15, -304),
      Quaternion<T>(
          AngleAxis<T>(pi<T>() / 2, Vector3<T>::UnitZ()) *
          AngleAxis<T>(pi<T>() / 4, Vector3<T>::UnitX())));

  const std::vector<ModelParametersT<T>> modelParams_final = transformPose(
      character,
      std::vector<ModelParametersT<T>>(transforms.size(), randomParams_init),
      transforms);

  const SkeletonStateT<T> skelState_init(
      castedCharacterParameterTransform.apply(randomParams_init), character.skeleton);
  for (size_t iTransform = 0; iTransform < transforms.size(); ++iTransform) {
    const auto transform = transforms[iTransform];
    const SkeletonStateT<T> skelState_final(
        castedCharacterParameterTransform.apply(modelParams_final[iTransform]), character.skeleton);

    for (size_t iJoint = 0; iJoint < skelState_init.jointState.size(); ++iJoint) {
      const TransformT<T> targetTransform = transform * skelState_init.jointState[iJoint].transform;
      const TransformT<T> actualTransform = skelState_final.jointState[iJoint].transform;

      EXPECT_LE((targetTransform.toMatrix() - actualTransform.toMatrix()).norm(), 5e-4);
    }
  }
}

TYPED_TEST(TransformPoseTest, ValidateTransformDividedRoot) {
  using T = typename TestFixture::Type;
  Random<> rng{12345};

  const Character character = createDividedRootCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();
  const Skeleton& skeleton = character.skeleton;

  SkeletonState initialBodySkeletonState(character.parameterTransform.zero(), skeleton);

  ModelParametersT<T> randomParams_init =
      rng.uniform<VectorX<T>>(character.parameterTransform.numAllModelParameters(), -1.0, 1.0);

  std::vector<TransformT<T>> transforms;
  transforms.emplace_back(
      Vector3<T>::Zero(), Quaternion<T>(AngleAxis<T>(pi<T>(), Vector3<T>::UnitY())));
  transforms.emplace_back(
      Vector3<T>(5, 15, 3), Quaternion<T>(AngleAxis<T>(pi<T>(), Vector3<T>::UnitX())));
  transforms.emplace_back(
      Vector3<T>(205, -15, -304),
      Quaternion<T>(
          AngleAxis<T>(pi<T>() / 2, Vector3<T>::UnitZ()) *
          AngleAxis<T>(pi<T>() / 4, Vector3<T>::UnitX())));

  const std::vector<ModelParametersT<T>> modelParams_final = transformPose(
      character,
      std::vector<ModelParametersT<T>>(transforms.size(), randomParams_init),
      transforms);

  const SkeletonStateT<T> skelState_init(
      castedCharacterParameterTransform.apply(randomParams_init), character.skeleton);
  for (size_t iTransform = 0; iTransform < transforms.size(); ++iTransform) {
    const auto transform = transforms[iTransform];
    const SkeletonStateT<T> skelState_final(
        castedCharacterParameterTransform.apply(modelParams_final[iTransform]), character.skeleton);

    // Start at the root joint, the world joint doesn't have all the parameters it needs to match
    // the transform:
    const auto expectedError = T(50) * std::sqrt(std::numeric_limits<T>::epsilon());
    for (size_t iJoint = 1; iJoint < skelState_init.jointState.size(); ++iJoint) {
      const TransformT<T> targetTransform = transform * skelState_init.jointState[iJoint].transform;
      const TransformT<T> actualTransform = skelState_final.jointState[iJoint].transform;

      EXPECT_LE((targetTransform.toMatrix() - actualTransform.toMatrix()).norm(), expectedError);
    }
  }
}

TYPED_TEST(TransformPoseTest, ValidateTransformsContinuous) {
  using T = typename TestFixture::Type;
  Random<> rng{12345};

  const Character character = createDividedRootCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();
  const Skeleton& skeleton = character.skeleton;

  SkeletonState initialBodySkeletonState(character.parameterTransform.zero(), skeleton);

  const auto root_rx_param = character.parameterTransform.getParameterIdByName("root_rx");
  const auto root_ry_param = character.parameterTransform.getParameterIdByName("root_ry");
  const auto root_rz_param = character.parameterTransform.getParameterIdByName("root_rz");
  ASSERT_LT(root_rx_param, character.parameterTransform.numAllModelParameters());
  ASSERT_LT(root_ry_param, character.parameterTransform.numAllModelParameters());
  ASSERT_LT(root_rz_param, character.parameterTransform.numAllModelParameters());

  std::vector<ModelParametersT<T>> modelParams_init;
  {
    ModelParametersT<T> randomParams_init =
        rng.uniform<VectorX<T>>(character.parameterTransform.numAllModelParameters(), -1.0, 1.0);
    // 100 frames ensures we go around several times.
    for (size_t iFrame = 0; iFrame < 100; ++iFrame) {
      modelParams_init.push_back(randomParams_init);
      randomParams_init(root_rx_param) += 0.1;
      randomParams_init(root_ry_param) -= 0.05;
    }
  }

  const TransformT<T> transform(
      Vector3<T>(0, 0, 0),
      Quaternion<T>(AngleAxis<T>(pi<T>(), Vector3<T>::UnitY())) *
          Quaternion<T>(AngleAxis<T>(pi<T>(), Vector3<T>::UnitX())));

  const std::vector<ModelParametersT<T>> modelParams_final =
      transformPose(character, modelParams_init, transform);

  ASSERT_EQ(modelParams_init.size(), modelParams_final.size());

  for (size_t iPose = 0; iPose < modelParams_init.size(); ++iPose) {
    const SkeletonStateT<T> skelState_init(
        castedCharacterParameterTransform.apply(modelParams_init[iPose]), character.skeleton);
    const SkeletonStateT<T> skelState_final(
        castedCharacterParameterTransform.apply(modelParams_final[iPose]), character.skeleton);

    // Start at the root joint, the world joint doesn't have all the parameters it needs to match
    // the transform:
    const auto expectedError = T(10) * std::sqrt(std::numeric_limits<T>::epsilon());
    for (size_t iJoint = 1; iJoint < skelState_init.jointState.size(); ++iJoint) {
      const TransformT<T> targetTransform = transform * skelState_init.jointState[iJoint].transform;
      const TransformT<T> actualTransform = skelState_final.jointState[iJoint].transform;

      EXPECT_LE((targetTransform.toMatrix() - actualTransform.toMatrix()).norm(), expectedError);
    }
  }

  // Check for Euler flips:
  for (size_t iPose = 1; iPose < modelParams_init.size(); ++iPose) {
    ASSERT_NEAR(
        modelParams_final[iPose][root_rx_param], modelParams_final[iPose - 1][root_rx_param], 0.2);
    ASSERT_NEAR(
        modelParams_final[iPose][root_ry_param], modelParams_final[iPose - 1][root_ry_param], 0.2);
    ASSERT_NEAR(
        modelParams_final[iPose][root_rz_param], modelParams_final[iPose - 1][root_rz_param], 0.2);
  }
}

TYPED_TEST(TransformPoseTest, ValidateTransformsContinuousWithSkips) {
  using T = typename TestFixture::Type;

  const Character character = createDividedRootCharacter();
  ParameterTransformT<T> castedCharacterParameterTransform = character.parameterTransform.cast<T>();
  const Skeleton& skeleton = character.skeleton;

  SkeletonState initialBodySkeletonState(character.parameterTransform.zero(), skeleton);

  const auto root_rx_param = character.parameterTransform.getParameterIdByName("root_rx");
  const auto root_ry_param = character.parameterTransform.getParameterIdByName("root_ry");
  const auto root_rz_param = character.parameterTransform.getParameterIdByName("root_rz");
  ASSERT_LT(root_rx_param, character.parameterTransform.numAllModelParameters());
  ASSERT_LT(root_ry_param, character.parameterTransform.numAllModelParameters());
  ASSERT_LT(root_rz_param, character.parameterTransform.numAllModelParameters());

  std::vector<ModelParametersT<T>> modelParams_init;
  {
    ModelParametersT<T> randomParams_init =
        VectorX<T>::Random(character.parameterTransform.numAllModelParameters());
    // 100 frames ensures we go around several times.
    for (size_t iFrame = 0; iFrame < 400; ++iFrame) {
      if (iFrame % 2 == 0) {
        modelParams_init.push_back(ModelParametersT<T>());
        continue;
      }

      modelParams_init.push_back(randomParams_init);
      randomParams_init(root_rx_param) -= 0.05;
      randomParams_init(root_ry_param) += 0.03;
      if (iFrame % 25 == 0) {
        randomParams_init(root_ry_param) += 2 * pi<T>();
      }
    }
  }

  const TransformT<T> transform(
      Vector3<T>(0, 0, 0),
      Quaternion<T>(AngleAxis<T>(pi<T>(), Vector3<T>::UnitY())) *
          Quaternion<T>(AngleAxis<T>(pi<T>(), Vector3<T>::UnitX())));

  const std::vector<ModelParametersT<T>> modelParams_final =
      transformPose(character, modelParams_init, transform);

  ASSERT_EQ(modelParams_init.size(), modelParams_final.size());

  for (size_t iPose = 0; iPose < modelParams_init.size(); ++iPose) {
    ASSERT_EQ(modelParams_final[iPose].size(), modelParams_init[iPose].size());

    if (modelParams_init[iPose].size() == 0) {
      continue;
    }

    const SkeletonStateT<T> skelState_init(
        castedCharacterParameterTransform.apply(modelParams_init[iPose]), character.skeleton);
    const SkeletonStateT<T> skelState_final(
        castedCharacterParameterTransform.apply(modelParams_final[iPose]), character.skeleton);

    // Start at the root joint, the world joint doesn't have all the parameters it needs to match
    // the transform:
    const auto expectedError = T(10) * std::sqrt(std::numeric_limits<T>::epsilon());
    for (size_t iJoint = 1; iJoint < skelState_init.jointState.size(); ++iJoint) {
      const TransformT<T> targetTransform = transform * skelState_init.jointState[iJoint].transform;
      const TransformT<T> actualTransform = skelState_final.jointState[iJoint].transform;

      EXPECT_LE((targetTransform.toMatrix() - actualTransform.toMatrix()).norm(), expectedError);
    }
  }

  // Check for Euler flips:
  ModelParametersT<T> lastPose;
  for (size_t iPose = 0; iPose < modelParams_init.size(); ++iPose) {
    ASSERT_EQ(modelParams_final[iPose].size(), modelParams_init[iPose].size());

    if (modelParams_init[iPose].size() == 0) {
      continue;
    }

    if (lastPose.size() != 0) {
      ASSERT_NEAR(modelParams_final[iPose][root_rx_param], lastPose[root_rx_param], 0.2);
      ASSERT_NEAR(modelParams_final[iPose][root_ry_param], lastPose[root_ry_param], 0.2);
      ASSERT_NEAR(modelParams_final[iPose][root_rz_param], lastPose[root_rz_param], 0.2);
    }

    lastPose = modelParams_final[iPose];
  }
}
