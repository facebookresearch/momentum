/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/marker_tracking/marker_tracker.h"

#include "momentum/character/character.h"
#include "momentum/character/parameter_limits.h"
#include "momentum/character_sequence_solver/model_parameters_sequence_error_function.h"
#include "momentum/character_sequence_solver/sequence_solver.h"
#include "momentum/character_sequence_solver/sequence_solver_function.h"
#include "momentum/character_solver/collision_error_function.h"
#include "momentum/character_solver/collision_error_function_stateless.h"
#include "momentum/character_solver/limit_error_function.h"
#include "momentum/character_solver/model_parameters_error_function.h"
#include "momentum/character_solver/plane_error_function.h"
#include "momentum/character_solver/position_error_function.h"
#include "momentum/character_solver/skeleton_solver_function.h"
#include "momentum/common/log.h"
#include "momentum/common/progress_bar.h"
#include "momentum/marker_tracking/tracker_utils.h"
#include "momentum/solver/gauss_newton_solver.h"
#include "momentum/solver/solver.h"

using namespace momentum;

namespace momentum {

std::vector<size_t> sampleFrames(
    momentum::Character& character,
    const MatrixXf& initialMotion,
    gsl::span<const std::vector<momentum::Marker>> markerData,
    const ParameterSet& parameters,
    const size_t frameStride,
    const size_t numSamples) {
  // sample frames so that we get the most variance from the initial input
  const auto numFrames = static_cast<size_t>(initialMotion.cols());
  size_t solvedFrames = (numFrames - 1) / frameStride + 1;

  std::vector<size_t> frameIndices;
  if (solvedFrames == 0) {
    return frameIndices;
  }
  const size_t numActualSamples = std::min(numSamples, solvedFrames);

  // get the indices of the parameters to be used
  std::vector<size_t> usedParameters;
  for (size_t i = 0; i < static_cast<size_t>(initialMotion.rows()); ++i) {
    if (parameters.test(i)) {
      usedParameters.push_back(i);
    }
  }

  // calculate per frame error
  const ParameterTransform& pt = character.parameterTransform;

  // map locator name to its index
  std::map<std::string, size_t> locatorLookup;
  for (size_t i = 0; i < character.locators.size(); i++) {
    locatorLookup[character.locators[i].name] = i;
  }

  std::vector<double> frameErrors(solvedFrames, 0.0f);
  SkeletonState state;
  for (size_t iFrame = 0, fi = 0; iFrame < numFrames; iFrame += frameStride, fi++) {
    const auto jointParams = pt.apply(initialMotion.col(iFrame));
    state.set(jointParams, character.skeleton, false);

    const auto& markerList = markerData[iFrame];
    for (const auto& jMarker : markerList) {
      if (jMarker.occluded) {
        continue;
      }
      auto query = locatorLookup.find(jMarker.name);
      if (query == locatorLookup.end()) {
        continue;
      }
      size_t locatorIdx = query->second;

      const auto& locator = character.locators[locatorIdx];
      const Vector3f locatorPos = state.jointState[locator.parent].transform * locator.offset;
      const Vector3f diff = locatorPos - jMarker.pos.cast<float>();
      const float markerError = diff.norm();
      frameErrors[fi] += markerError;
    }
  }
  std::vector<double> sortedErrors = frameErrors;
  std::sort(sortedErrors.begin(), sortedErrors.end());

  // do not use the worst 1/4 of the fitted errors, there's likely to be some outliers and
  // errors due to the initialization
  const double threshold = sortedErrors[(sortedErrors.size() * 3) / 4];

  // get the motion of only the used parameters and then normalize
  // by the mean and variance
  MatrixXf normalized;
  {
    const MatrixXf subMotion =
        initialMotion(usedParameters, Eigen::seq(0, numFrames - 1, frameStride));
    const VectorXf mean = subMotion.rowwise().mean();
    const MatrixXf centered = subMotion.colwise() - mean;
    const VectorXf std = centered.array().square().rowwise().sum() / (numFrames - 1);

    // normalize the motion by the sqrt of the variance, i.e. leave a little bit of scale in
    normalized = centered.array().colwise() / (std.array().sqrt().sqrt().cwiseMax(1e-5f));
  }

  for (size_t i = 0; i < frameErrors.size(); ++i) {
    if (frameErrors[i] > threshold) {
      normalized.col(i) = VectorXf::Ones(normalized.rows()) * 1000.0f;
    }
  }

  // calculate the distances from each frame to the already selected indices
  frameIndices.push_back(0);
  VectorXf distances = VectorXf::Zero(solvedFrames);
  for (size_t fi = 0; fi < solvedFrames; fi++) {
    distances[fi] = (normalized.col(frameIndices[0]) - normalized.col(fi)).norm();
  }

  // finally add additional samples with the largest distance
  for (size_t i = frameIndices.size(); i < numActualSamples; ++i) {
    size_t maxFrame = 0;
    const float maxval = distances.maxCoeff(&maxFrame);
    if (maxval < 1e-5f) {
      break;
    }
    frameIndices.push_back(maxFrame);

    for (size_t fi = 0; fi < solvedFrames; fi++) {
      const float dist = (normalized.col(maxFrame) - normalized.col(fi)).cwiseAbs().maxCoeff();
      distances[fi] = std::min(distances[fi], dist);
    }
  }

  for (auto&& fi : frameIndices) {
    fi *= frameStride;
  }

  return frameIndices;
}

Eigen::MatrixXf trackSequence(
    const gsl::span<const std::vector<Marker>> markerData,
    const Character& character,
    const ParameterSet& globalParams,
    const MatrixXf& initialMotion,
    const TrackingConfig& config,
    float regularizer,
    const size_t frameStride,
    bool enforceFloorInFirstFrame,
    const std::string& firstFramePoseConstraintSet) {
  // sanity checks
  const size_t numFrames = markerData.size();
  std::vector<size_t> frames;
  for (size_t fi = 0; fi < numFrames; fi += frameStride) {
    frames.emplace_back(fi);
  }

  return trackSequence(
      markerData,
      character,
      globalParams,
      initialMotion,
      config,
      frames,
      regularizer,
      enforceFloorInFirstFrame,
      firstFramePoseConstraintSet);
}

Eigen::MatrixXf trackSequence(
    const gsl::span<const std::vector<Marker>> markerData,
    const Character& character,
    const ParameterSet& globalParams,
    const MatrixXf& initialMotion,
    const TrackingConfig& config,
    const std::vector<size_t>& frames,
    float regularizer,
    bool enforceFloorInFirstFrame,
    const std::string& firstFramePoseConstraintSet) {
  // sanity checks
  const size_t numFrames = markerData.size();
  MT_CHECK(numFrames > 0, "Input data is empty.");
  MT_CHECK(
      initialMotion.cols() >= numFrames,
      "Number of frames in data {} doesn't match that of input motion {}",
      numFrames,
      initialMotion.cols());
  MT_CHECK(
      initialMotion.rows() == character.parameterTransform.numAllModelParameters(),
      "Input motion parameters {} do not match character model parameters {}",
      initialMotion.rows(),
      character.parameterTransform.numAllModelParameters());

  const ParameterTransform& pt = character.parameterTransform;
  const size_t numMarkers = markerData[0].size();

  // universal parameters include "scaling" and "locators" (if exists); pose parameters need to
  // exclude "locators". universalParams is to indicate to the solver which parameters are "global"
  // (ie. not time varying). The input globalParams indicate which parameters within universalParams
  // we want to solve for. globalParams is either a subset or all of universalParams.
  ParameterSet poseParams = pt.getPoseParameters();
  ParameterSet universalParams = pt.getScalingParameters();
  const auto locatorSet = pt.parameterSets.find("locators");
  if (locatorSet != pt.parameterSets.end()) {
    poseParams &= ~locatorSet->second;
    universalParams |= locatorSet->second;
  }

  // set up the solver function
  std::vector<size_t> sortedFrames = frames;
  std::sort(sortedFrames.begin(), sortedFrames.end());
  size_t solvedFrames = sortedFrames.size();
  auto solverFunc = SequenceSolverFunction(
      &character.skeleton, &character.parameterTransform, universalParams, solvedFrames);

  // floor penetration constraints; we assume the world is y-up and floor is y=0 for mocap data.
  const auto& floorConstraints = createFloorConstraints<float>(
      "Floor_",
      character.locators,
      Vector3f::UnitY(),
      /* y offset */ 0.0f,
      /* weight */ 5.0f);

  // marker constraints
  const auto constrData = createConstraintData(markerData, character.locators);

  // add per-frame constraint data to the solver
  for (size_t solverFrame = 0; solverFrame < solvedFrames; ++solverFrame) {
    const size_t& iFrame = sortedFrames[solverFrame];
    if (constrData.at(iFrame).size() > numMarkers * config.minVisPercent) {
      // prepare positional constraints
      auto posConstrFunc = std::make_shared<PositionErrorFunction>(character, config.lossAlpha);
      posConstrFunc->setConstraints(constrData.at(iFrame));
      if (solverFrame == 0 && (enforceFloorInFirstFrame || !firstFramePoseConstraintSet.empty())) {
        posConstrFunc->setWeight(PositionErrorFunction::kLegacyWeight * solvedFrames);
      } else {
        posConstrFunc->setWeight(PositionErrorFunction::kLegacyWeight);
      }
      solverFunc.addErrorFunction(solverFrame, posConstrFunc);

      // prepare floor constraints
      if (!floorConstraints.empty()) {
        bool halfPlane = true;
        float weightMultiplier = 1.0f;
        if (enforceFloorInFirstFrame && solverFrame == 0) {
          halfPlane = false;
          weightMultiplier = solvedFrames;
        }
        auto halfPlaneConstrFunc =
            std::make_shared<PlaneErrorFunction>(character, /*half plane*/ halfPlane);
        halfPlaneConstrFunc->setConstraints(floorConstraints);
        halfPlaneConstrFunc->setWeight(PlaneErrorFunction::kLegacyWeight * weightMultiplier);
        solverFunc.addErrorFunction(solverFrame, halfPlaneConstrFunc);
      }

      // add pose constraint set if defined
      if (!firstFramePoseConstraintSet.empty() && solverFrame == 0) {
        // check if the constraint set is defined
        if (character.parameterTransform.poseConstraints.count(firstFramePoseConstraintSet) > 0) {
          // add parameter limits
          const auto poseLimits = getPoseConstraintParameterLimits(
              firstFramePoseConstraintSet, character.parameterTransform);
          auto limitConstrFunc = std::make_shared<LimitErrorFunction>(character, poseLimits);
          limitConstrFunc->setWeight(solvedFrames);
          solverFunc.addErrorFunction(solverFrame, limitConstrFunc);
        }
      }
    }
    // Set per-frame initial value
    solverFunc.setFrameParameters(solverFrame, initialMotion.col(iFrame));
  }

  // add parameter limits
  auto limitConstrFunc = std::make_shared<LimitErrorFunction>(character);
  limitConstrFunc->setWeight(0.1);
  solverFunc.addErrorFunction(kAllFrames, limitConstrFunc);

  // add collision error
  if (config.collisionErrorWeight != 0 && character.collision != nullptr) {
    auto collisionConstrFunc = std::make_shared<CollisionErrorFunctionStateless>(character);
    collisionConstrFunc->setWeight(config.collisionErrorWeight);
    solverFunc.addErrorFunction(kAllFrames, collisionConstrFunc);
  }

  // add a smoothness constraint in parameter space
  if (config.smoothing != 0) {
    auto smoothConstrFunc = std::make_shared<ModelParametersSequenceErrorFunction>(character);
    smoothConstrFunc->setWeight(config.smoothing);
    solverFunc.addSequenceErrorFunction(kAllFrames, smoothConstrFunc);
  }

  // minimize the change to global params
  if (globalParams.count() > 0 && regularizer != 0) {
    auto regularizerFunc = std::make_shared<ModelParametersErrorFunction>(character);
    Eigen::VectorXf universalMask(pt.numAllModelParameters());
    for (size_t i = 0; i < universalMask.size(); ++i) {
      if (globalParams.test(i)) {
        universalMask[i] = regularizer;
      } else {
        universalMask[i] = 0.0;
      }
    }
    regularizerFunc->setTargetParameters(initialMotion.col(0), universalMask);
    // Sufficient to add to the first frame since it won't change.
    solverFunc.addErrorFunction(0, regularizerFunc);
  }

  // solver configration
  SequenceSolverOptions solverOptions;
  solverOptions.maxIterations = config.maxIter;
  solverOptions.minIterations = 2;
  solverOptions.progressBar = config.debug;
  solverOptions.doLineSearch = false;
  solverOptions.multithreaded = true;
  solverOptions.verbose = config.debug;
  solverOptions.threshold = 1.f;
  solverOptions.regularization = 0.05f;

  // solve the problem
  SequenceSolver solver = SequenceSolver(solverOptions, &solverFunc);
  solver.setEnabledParameters(poseParams | globalParams);
  // returns all the dofs with initial values nicely packed into a vector
  VectorXf dofs = solverFunc.getJoinedParameterVector();
  solver.solve(dofs);
  double error = solverFunc.getError(dofs);
  MT_LOGI_IF(config.debug, "Solver residual: {}", error);

  // set results to output
  size_t sortedIndex = 0;
  MatrixXf outMotion(pt.numAllModelParameters(), numFrames);
  for (size_t fi = 0; fi < numFrames; fi++) {
    if (sortedIndex < sortedFrames.size() - 1 && fi == sortedFrames[sortedIndex + 1]) {
      sortedIndex++;
    }
    outMotion.col(fi) = solverFunc.getFrameParameters(sortedIndex).v;
  }
  return outMotion;
}

Eigen::MatrixXf trackPosesPerframe(
    const gsl::span<const std::vector<Marker>> markerData,
    const Character& character,
    const ModelParameters& globalParams,
    const TrackingConfig& config,
    const size_t frameStride) {
  const size_t numFrames = markerData.size();
  MT_CHECK(numFrames > 0, "Input data is empty.");
  MT_CHECK(
      globalParams.v.size() == character.parameterTransform.numAllModelParameters(),
      "Input model parameters {} do not match character model parameters {}",
      globalParams.v.size(),
      character.parameterTransform.numAllModelParameters());

  const ParameterTransform& pt = character.parameterTransform;
  const size_t numMarkers = markerData[0].size();

  // pose parameters need to exclude "locators"
  ParameterSet poseParams = pt.getPoseParameters();
  const auto& locatorSet = pt.parameterSets.find("locators");
  if (locatorSet != pt.parameterSets.end()) {
    poseParams &= ~locatorSet->second;
  }

  // set up the solver
  auto solverFunc = SkeletonSolverFunction(&character.skeleton, &pt);
  GaussNewtonSolverOptions solverOptions;
  solverOptions.maxIterations = config.maxIter;
  solverOptions.minIterations = 2;
  solverOptions.doLineSearch = false;
  solverOptions.verbose = config.debug;
  solverOptions.threshold = 1.f;
  solverOptions.regularization = 0.05f;
  auto solver = GaussNewtonSolver(solverOptions, &solverFunc);
  solver.setEnabledParameters(poseParams);

  // parameter limits constraint
  auto limitConstrFunc = std::make_shared<LimitErrorFunction>(character);
  limitConstrFunc->setWeight(0.1);
  solverFunc.addErrorFunction(limitConstrFunc);

  // positional constraint function for markers
  auto posConstrFunc = std::make_shared<PositionErrorFunction>(character, config.lossAlpha);
  posConstrFunc->setWeight(PositionErrorFunction::kLegacyWeight);
  solverFunc.addErrorFunction(posConstrFunc);

  // floor penetration constraint data; we assume the world is y-up and floor is y=0 for mocap data.
  const auto& floorConstraints = createFloorConstraints<float>(
      "Floor_",
      character.locators,
      Vector3f::UnitY(),
      /* y offset */ 0.0f,
      /* weight */ 5.0f);
  auto halfPlaneConstrFunc = std::make_shared<PlaneErrorFunction>(character, /*half plane*/ true);
  halfPlaneConstrFunc->setConstraints(floorConstraints);
  halfPlaneConstrFunc->setWeight(PlaneErrorFunction::kLegacyWeight);
  solverFunc.addErrorFunction(halfPlaneConstrFunc);

  // marker constraint data
  auto constrData = createConstraintData(markerData, character.locators);

  // smoothness constraint only for the joints and exclude global dofs because the global transform
  // needs to be accurate (may not matter in practice?)
  auto smoothConstrFunc = std::make_shared<ModelParametersErrorFunction>(
      character, poseParams & ~pt.getRigidParameters());
  smoothConstrFunc->setWeight(config.smoothing);
  solverFunc.addErrorFunction(smoothConstrFunc);

  // add collision error
  std::shared_ptr<CollisionErrorFunction> collisionErrorFunction;
  if (config.collisionErrorWeight != 0 && character.collision != nullptr) {
    collisionErrorFunction = std::make_shared<CollisionErrorFunction>(character);
    collisionErrorFunction->setWeight(config.collisionErrorWeight);
    solverFunc.addErrorFunction(collisionErrorFunction);
  }

  MatrixXf motion(pt.numAllModelParameters(), numFrames);
  // initialize parameters to contain identity information
  // the identity fields will be used but untouched during optimization
  // globalParams could also be repurposed to pass in initial pose value
  Eigen::VectorXf dof = globalParams.v;
  size_t solverFrame = 0;
  double error = 0.0;
  // Use the initial global transform is it's not zero
  bool needsInit = dof.head(6).isZero(0); // TODO: assume first six dofs are global dofs

  // When the frames are not continuous, we sometimes run into an issue when the desired joint
  // rotation between two consecutive frames is large (eg. larger than 180). If we initialize from
  // the previous result, the smaller rotation will be wrongly chosen, and we cannot recover from
  // this mistake. To prevent this, we will solve each frame completely independently when they are
  // not continuous.
  bool continuous = (frameStride < 5);
  if (!continuous) {
    needsInit = true;
  }

  { // scope the ProgressBar so it returns
    ProgressBar progress("", numFrames);
    for (size_t iFrame = 0; iFrame < numFrames; iFrame += frameStride) {
      // reinitialize if not continuous
      if (!continuous) {
        dof = globalParams.v;
      }

      if (constrData.at(iFrame).size() > numMarkers * config.minVisPercent) {
        // add positional constraints
        posConstrFunc->clearConstraints(); // clear constraint data from the previous frame
        posConstrFunc->setConstraints(constrData.at(iFrame));

        // initialization
        // TODO: run on first frame or tracking failure
        if (needsInit) { // solve only for the rigid parameters as preprocessing
          MT_LOGI_IF(
              config.debug && continuous, "Solving for an initial rigid pose at frame {}", iFrame);

          // Set up different config for initialization
          solverOptions.maxIterations = 50; // make sure it converges
          solver.setOptions(solverOptions);
          solver.setEnabledParameters(pt.getRigidParameters());
          smoothConstrFunc->setWeight(0.0); // turn off smoothing - it doesn't affect rigid dofs

          solver.solve(dof);

          // Recover solver config
          solverOptions.maxIterations = config.maxIter;
          solver.setOptions(solverOptions);
          solver.setEnabledParameters(poseParams);
          smoothConstrFunc->setWeight(config.smoothing);

          if (continuous) {
            needsInit = false;
          }
        }

        // set smoothness target as the last pose -- dof holds parameter values from last (good)
        // frame it will serve as a small regularization to rest pose for the first frame
        // TODO: API needs improvement
        smoothConstrFunc->setTargetParameters(dof, smoothConstrFunc->getTargetWeights());

        error += solver.solve(dof);
        ++solverFrame;
      }

      // set result to output; fill in frames within a stride
      // note that dof contains complete parameter info with identity
      for (size_t jDelta = 0; jDelta < frameStride && iFrame + jDelta < numFrames; ++jDelta) {
        motion.col(iFrame + jDelta) = dof;
      }
      progress.increment(frameStride);
    }
  }
  if (config.debug) {
    if (solverFrame > 0) {
      MT_LOGI("Average per-frame residual: {}", error / solverFrame);
    } else {
      MT_LOGW("no valid frames to solve");
    }
  }
  return motion;
}

Eigen::MatrixXf trackPosesForFrames(
    const gsl::span<const std::vector<Marker>> markerData,
    const Character& character,
    const MatrixXf& initialMotion,
    const TrackingConfig& config,
    const std::vector<size_t>& frameIndices) {
  const size_t numFrames = markerData.size();
  MT_CHECK(numFrames > 0, "Input data is empty.");
  MT_CHECK(
      initialMotion.rows() == character.parameterTransform.numAllModelParameters(),
      "Input motion parameters {} do not match character model parameters {}",
      initialMotion.rows(),
      character.parameterTransform.numAllModelParameters());

  const ParameterTransform& pt = character.parameterTransform;
  const size_t numMarkers = markerData[0].size();

  std::vector<size_t> sortedFrames = frameIndices;
  std::sort(sortedFrames.begin(), sortedFrames.end());

  // pose parameters need to exclude "locators"
  ParameterSet poseParams = pt.getPoseParameters();
  const auto& locatorSet = pt.parameterSets.find("locators");
  if (locatorSet != pt.parameterSets.end()) {
    poseParams &= ~locatorSet->second;
  }

  // set up the solver
  auto solverFunc = SkeletonSolverFunction(&character.skeleton, &pt);
  GaussNewtonSolverOptions solverOptions;
  solverOptions.maxIterations = config.maxIter;
  solverOptions.minIterations = 2;
  solverOptions.doLineSearch = false;
  solverOptions.verbose = config.debug;
  solverOptions.threshold = 1.f;
  solverOptions.regularization = 0.05f;
  auto solver = GaussNewtonSolver(solverOptions, &solverFunc);
  solver.setEnabledParameters(poseParams);

  // parameter limits constraint
  auto limitConstrFunc = std::make_shared<LimitErrorFunction>(character);
  limitConstrFunc->setWeight(0.1);
  solverFunc.addErrorFunction(limitConstrFunc);

  // positional constraint function for markers
  auto posConstrFunc = std::make_shared<PositionErrorFunction>(character, config.lossAlpha);
  posConstrFunc->setWeight(PositionErrorFunction::kLegacyWeight);
  solverFunc.addErrorFunction(posConstrFunc);

  // floor penetration constraint data; we assume the world is y-up and floor is y=0 for mocap data.
  const auto& floorConstraints = createFloorConstraints<float>(
      "Floor_",
      character.locators,
      Vector3f::UnitY(),
      /* y offset */ 0.0f,
      /* weight */ 5.0f);
  auto halfPlaneConstrFunc = std::make_shared<PlaneErrorFunction>(character, /*half plane*/ true);
  halfPlaneConstrFunc->setConstraints(floorConstraints);
  halfPlaneConstrFunc->setWeight(PlaneErrorFunction::kLegacyWeight);
  solverFunc.addErrorFunction(halfPlaneConstrFunc);

  // marker constraint data
  auto constrData = createConstraintData(markerData, character.locators);

  // add collision error
  std::shared_ptr<CollisionErrorFunction> collisionErrorFunction;
  if (config.collisionErrorWeight != 0 && character.collision != nullptr) {
    collisionErrorFunction = std::make_shared<CollisionErrorFunction>(character);
    collisionErrorFunction->setWeight(config.collisionErrorWeight);
    solverFunc.addErrorFunction(collisionErrorFunction);
  }

  // initialize parameters to contain identity information
  // the identity fields will be used but untouched during optimization
  // globalParams could also be repurposed to pass in initial pose value
  std::vector<Eigen::VectorXf> poses(frameIndices.size());
  Eigen::VectorXf dof = initialMotion.col(sortedFrames[0]);
  size_t solverFrame = 0;
  double priorError = 0.0;
  double error = 0.0;

  MatrixXf outMotion(pt.numAllModelParameters(), numFrames);
  { // scope the ProgressBar so it returns
    ProgressBar progress("", sortedFrames.size());
    for (size_t fi = 0; fi < sortedFrames.size(); fi++) {
      const size_t& iFrame = sortedFrames[fi];
      dof = initialMotion.col(iFrame);

      if (constrData.at(iFrame).size() > numMarkers * config.minVisPercent) {
        // add positional constraints
        posConstrFunc->clearConstraints(); // clear constraint data from the previous frame
        posConstrFunc->setConstraints(constrData.at(iFrame));

        // initialization
        solverOptions.maxIterations = 50; // make sure it converges
        solver.setOptions(solverOptions);
        solver.setEnabledParameters(pt.getRigidParameters());

        solver.solve(dof);

        // Recover solver config
        solverOptions.maxIterations = config.maxIter;
        solver.setOptions(solverOptions);
        solver.setEnabledParameters(poseParams);

        priorError += solverFunc.getError(dof);
        error += solver.solve(dof);
        ++solverFrame;
      }

      // store result
      poses[fi] = dof;
      progress.increment();
    }

    // set results to output
    size_t sortedIndex = 0;
    for (size_t fi = 0; fi < numFrames; fi++) {
      if (sortedIndex < sortedFrames.size() - 1 && fi == sortedFrames[sortedIndex + 1]) {
        sortedIndex++;
      }
      outMotion.col(fi) = poses[sortedIndex];
    }
  }
  if (config.debug) {
    if (solverFrame > 0) {
      MT_LOGI("Pre optimization residual: {}", priorError / solverFrame);
      MT_LOGI("Average per-frame residual: {}", error / solverFrame);
    } else {
      MT_LOGW("no valid frames to solve");
    }
  }
  return outMotion;
}

void calibrateModel(
    const gsl::span<const std::vector<Marker>> markerData,
    const CalibrationConfig& config,
    Character& character,
    ModelParameters& identity) {
  MT_CHECK(
      identity.v.size() == character.parameterTransform.numAllModelParameters(),
      "Input identity parameters {} do not match character parameters {}",
      identity.v.size(),
      character.parameterTransform.numAllModelParameters());

  const size_t numFrames = markerData.size();
  // uniformly sample frames for calibration
  int frameStride = (numFrames - 1) / config.calibFrames;
  frameStride = std::max(1, frameStride);

  // create a solving character with markers as bones
  Character solvingCharacter = createLocatorCharacter(character, "locator_");

  // Extended quantities are for the solvingCharacter, which includes locators as bones
  // w/o Extended quantities are for character with fixed locators
  const ParameterTransform& transformExtended = solvingCharacter.parameterTransform;
  const ParameterTransform& transform = character.parameterTransform;

  ParameterSet locatorSet = transformExtended.parameterSets.find("locators")->second;
  ParameterSet calibBodySetExtended;
  ParameterSet calibBodySet;
  if (config.globalScaleOnly) {
    calibBodySetExtended.set(transformExtended.getParameterIdByName("scale_global"));
    calibBodySet.set(transform.getParameterIdByName("scale_global"));
  } else {
    calibBodySetExtended = transformExtended.getScalingParameters();
    calibBodySet = transform.getScalingParameters();
  }

  // special trackingConfig for initialization: zero out smoothness and collision
  TrackingConfig trackingConfig{
      {config.minVisPercent, config.lossAlpha, config.maxIter, config.debug}, 0.0, 0.0};

  // only keep one motion; no need to duplicate.
  // identity information will be initialized and updated in the motion matrix throughout all the
  // solves.
  MatrixXf motion = MatrixXf::Zero(transformExtended.numAllModelParameters(), numFrames);
  std::vector<size_t> frameIndices;

  { // Initialization
    MT_LOGI_IF(config.debug, "Solving for an initial pose and skeleton");

    // first solve for initial tracking poses with fixed identity and locators to default
    // Because we are solving for poses only, use character to save compute.
    if (config.greedySampling > 0) {
      // first only track the first frame
      MT_LOGI_IF(config.debug, "Pre-solving for the first frame");
      std::vector<size_t> firstFrame;
      firstFrame.push_back(0);
      motion.topRows(transform.numAllModelParameters()) = trackPosesForFrames(
          markerData,
          character,
          motion.topRows(transform.numAllModelParameters()),
          trackingConfig,
          firstFrame);
      motion.topRows(transform.numAllModelParameters()) = trackSequence(
          markerData,
          character,
          calibBodySet,
          motion.topRows(transform.numAllModelParameters()),
          trackingConfig,
          firstFrame,
          0.0,
          config.enforceFloorInFirstFrame,
          config.firstFramePoseConstraintSet); // still solving a subset
      std::tie(identity.v, character.locators) =
          extractIdAndLocatorsFromParams(motion.col(0), solvingCharacter, character);

      // track sequence with selected stride. we need to sample at least config.calibFrames frames
      // so make sure we have a stride that allows that
      size_t sampleStride = std::min(numFrames / config.calibFrames, config.greedySampling);

      motion.topRows(transform.numAllModelParameters()) =
          trackPosesPerframe(markerData, character, identity, trackingConfig, sampleStride);

      const ParameterSet ps = transformExtended.getPoseParameters() &
          ~transformExtended.getRigidParameters() & ~locatorSet;
      frameIndices = sampleFrames(
          character,
          motion.topRows(transform.numAllModelParameters()),
          markerData,
          ps,
          sampleStride,
          config.calibFrames);

      // then solve for identity and poses with fixed locators, initialized with solved poses
      // this works using "character" because additional parameters for the locators are appended at
      // the end, so the indices work out using topRows() without special treatment.
      motion.topRows(transform.numAllModelParameters()) = trackSequence(
          markerData,
          character,
          calibBodySet, // only solve for identity and not markers
          motion.topRows(transform.numAllModelParameters()),
          trackingConfig,
          frameIndices,
          0.0,
          /*regularizer*/ // allow large change at initialization without any regularization
          config.enforceFloorInFirstFrame,
          config.firstFramePoseConstraintSet);
    } else {
      motion.topRows(transform.numAllModelParameters()) =
          trackPosesPerframe(markerData, character, identity, trackingConfig, frameStride);

      // then solve for identity and poses with fixed locators, initialized with solved poses
      // this works using "character" because additional parameters for the locators are appended at
      // the end, so the indices work out using topRows() without special treatment.
      motion.topRows(transform.numAllModelParameters()) = trackSequence(
          markerData,
          character,
          calibBodySet, // only solve for identity and not markers
          motion.topRows(transform.numAllModelParameters()),
          trackingConfig,
          0.0 /*regularizer*/, // allow large change at initialization without any regularization
          frameStride,
          config.enforceFloorInFirstFrame,
          config.firstFramePoseConstraintSet);
    }
  }

  // Solve everything together for a few iterations
  for (size_t iIter = 0; iIter < config.majorIter; ++iIter) {
    MT_LOGI_IF(config.debug, "Iteration {} of calibration", iIter);

    if (config.greedySampling > 0) {
      motion = trackSequence(
          markerData,
          solvingCharacter,
          locatorSet | calibBodySetExtended,
          motion,
          trackingConfig,
          frameIndices,
          0.0, // TODO: use a small regularization to prevent too large a change
          config.enforceFloorInFirstFrame,
          config.firstFramePoseConstraintSet); // still solving a subset
    } else {
      motion = trackSequence(
          markerData,
          solvingCharacter,
          locatorSet | calibBodySetExtended,
          motion,
          trackingConfig,
          0.0, // TODO: use a small regularization to prevent too large a change
          frameStride,
          config.enforceFloorInFirstFrame,
          config.firstFramePoseConstraintSet); // still solving a subset
    }
    // extract solving results to identity and character so we can pass them to trackPosesPerframe
    // below.
    std::tie(identity.v, character.locators) =
        extractIdAndLocatorsFromParams(motion.col(0), solvingCharacter, character);

    // The sequence solve above could get stuck with euler singularity but per-frame solve could get
    // it out. Pass in the first frame from previous solve as a better initial guess than the zero
    // pose.
    if (config.greedySampling > 0) {
      motion.topRows(transform.numAllModelParameters()) = trackPosesForFrames(
          markerData,
          character,
          motion.topRows(transform.numAllModelParameters()),
          trackingConfig,
          frameIndices);
    } else {
      const VectorXf initPose = motion.col(0).head(transform.numAllModelParameters());
      motion.topRows(transform.numAllModelParameters()) =
          trackPosesPerframe(markerData, character, initPose, trackingConfig, frameStride);
    }
  }

  // Finally, fine tune marker offsets with fix identity.
  MT_LOGI_IF(config.debug, "Fine-tune marker offsets");

  // TODO: use a larger regularizer to prevent too large a change.
  if (config.greedySampling > 0) {
    motion = trackSequence(
        markerData,
        solvingCharacter,
        locatorSet,
        motion,
        trackingConfig,
        frameIndices,
        0.0,
        config.enforceFloorInFirstFrame,
        config.firstFramePoseConstraintSet);
  } else {
    motion = trackSequence(
        markerData,
        solvingCharacter,
        locatorSet,
        motion,
        trackingConfig,
        0.0,
        frameStride,
        config.enforceFloorInFirstFrame,
        config.firstFramePoseConstraintSet);
  }
  std::tie(identity.v, character.locators) =
      extractIdAndLocatorsFromParams(motion.col(0), solvingCharacter, character);

  // TODO: A hack to return the solved first frame as initialization for tracking later.
  identity.v = motion.col(0).head(transform.numAllModelParameters());
}

void calibrateLocators(
    const gsl::span<const std::vector<Marker>> markerData,
    const CalibrationConfig& config,
    const ModelParameters& identity,
    Character& character) {
  MT_CHECK(
      identity.v.size() == character.parameterTransform.numAllModelParameters(),
      "Input identity parameters {} do not match character parameters {}",
      identity.v.size(),
      character.parameterTransform.numAllModelParameters());

  const size_t numFrames = markerData.size();
  // uniformly sample frames for calibration
  int frameStride = (numFrames - 1) / config.calibFrames;
  frameStride = std::max(1, frameStride);

  // create a solving character with locators as bones
  Character solvingCharacter = createLocatorCharacter(character, "locator_");

  // Extended quantities are for the solvingCharacter, which includes locators as bones
  // w/o Extended quantities are for character with fixed locators
  const ParameterTransform& transformExtended = solvingCharacter.parameterTransform;
  const ParameterTransform& transform = character.parameterTransform;

  ParameterSet locatorSet = transformExtended.parameterSets.find("locators")->second;

  // special trackingConfig for initialization: zero out smoothness and collision
  TrackingConfig trackingConfig{
      {config.minVisPercent, config.lossAlpha, config.maxIter, config.debug}, 0.0, 0.0};

  // only keep one motion for both character and solvingCharacter; no need to duplicate.
  // identity information will be initialized and updated in the motion matrix throughout all the
  // solves.
  MatrixXf motion = MatrixXf::Zero(transformExtended.numAllModelParameters(), numFrames);
  CharacterParameters fullParams;

  // Iterate for a few times
  for (size_t iIter = 0; iIter < config.majorIter; ++iIter) {
    MT_LOGI_IF(config.debug, "Iteration {} of locator calibration", iIter);

    // Solve only for poses using solved locators; it helps to adjust poses to get out of bad
    // solutions.
    motion.topRows(transform.numAllModelParameters()) =
        trackPosesPerframe(markerData, character, identity, trackingConfig, frameStride);
    // Solve for both markers and poses.
    // TODO: add a small regularization to prevent too large a change
    motion = trackSequence(
        markerData,
        solvingCharacter,
        locatorSet,
        motion,
        trackingConfig,
        0.0,
        frameStride,
        config.enforceFloorInFirstFrame,
        config.firstFramePoseConstraintSet);
    // Extract solved locators
    fullParams.pose = motion.col(0);
    character.locators = extractLocatorsFromCharacter(solvingCharacter, fullParams);
  }
}

MatrixXf refineMotion(
    gsl::span<const std::vector<momentum::Marker>> markerData,
    const MatrixXf& motion,
    const RefineConfig& config,
    momentum::Character& character) {
  MT_CHECK(
      markerData.size() == motion.cols(),
      "markers and motion frames mismatch: {} != {}",
      markerData.size(),
      motion.cols());

  MatrixXf newMotion;
  const ParameterSet idParamSet = character.parameterTransform.getScalingParameters();

  // use sequenceSolve to smooth out the input motion
  if (!config.calibLocators) {
    newMotion = trackSequence(
        markerData,
        character,
        config.calibId ? idParamSet : ParameterSet(),
        motion,
        config,
        config.regularizer);
  } else {
    // create a solving character with markers as bones
    Character solvingCharacter = createLocatorCharacter(character, "locator_");
    const ParameterTransform& transformExtended = solvingCharacter.parameterTransform;
    const ParameterSet locatorSet = transformExtended.parameterSets.find("locators")->second;
    ParameterSet calibrationSet = locatorSet;
    if (config.calibId) {
      calibrationSet |= transformExtended.getScalingParameters();
    }

    const auto numParams = character.parameterTransform.numAllModelParameters();
    const auto numParamsExtended = transformExtended.numAllModelParameters();
    MatrixXf motionExtended(numParamsExtended, markerData.size());
    motionExtended.setZero();
    motionExtended.topRows(numParams) = motion;
    newMotion = trackSequence(
        markerData, solvingCharacter, calibrationSet, motionExtended, config, config.regularizer);

    std::tie(std::ignore, character.locators) =
        extractIdAndLocatorsFromParams(newMotion.col(0), solvingCharacter, character);
    newMotion.conservativeResize(numParams, Eigen::NoChange_t::NoChange);
  }

  return newMotion;
}

std::pair<float, float> getLocatorError(
    gsl::span<const std::vector<momentum::Marker>> markerData,
    const MatrixXf& motion,
    momentum::Character& character) {
  const size_t numFrames = markerData.size();
  MT_CHECK(numFrames > 0, "Input data is empty.");
  MT_CHECK(
      motion.rows() == character.parameterTransform.numAllModelParameters(),
      "Input motion parameters {} do not match character model parameters {}",
      motion.rows(),
      character.parameterTransform.numAllModelParameters());

  const ParameterTransform& pt = character.parameterTransform;

  // map locator name to its index
  std::map<std::string, size_t> locatorLookup;
  for (size_t i = 0; i < character.locators.size(); i++) {
    locatorLookup[character.locators[i].name] = i;
  }

  SkeletonState state;

  // go over all frames and pose the locators and compute the error
  double error = 0.0;
  double maxError = 0.0;
  for (size_t iFrame = 0; iFrame < numFrames; ++iFrame) {
    const auto jointParams = pt.apply(motion.col(iFrame));
    state.set(jointParams, character.skeleton, false);

    double frameError = 0.0;

    const auto& markerList = markerData[iFrame];
    size_t validMarkers = 0;
    for (const auto& jMarker : markerList) {
      if (jMarker.occluded) {
        continue;
      }
      auto query = locatorLookup.find(jMarker.name);
      if (query == locatorLookup.end()) {
        continue;
      }
      size_t locatorIdx = query->second;

      const auto& locator = character.locators[locatorIdx];
      const Vector3f locatorPos = state.jointState[locator.parent].transform * locator.offset;
      const Vector3f diff = locatorPos - jMarker.pos.cast<float>();
      const float markerError = diff.norm();
      frameError += markerError;
      maxError = std::max(maxError, static_cast<double>(markerError));
      validMarkers++;
    }

    if (validMarkers > 0) {
      error += frameError / validMarkers;
    }
  }
  return {error / numFrames, maxError};
}

} // namespace momentum
