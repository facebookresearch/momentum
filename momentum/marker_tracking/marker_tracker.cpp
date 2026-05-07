/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/marker_tracking/marker_tracker.h"

#include "momentum/camera/projection_utils.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/parameter_limits.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_sequence_solver/model_parameters_sequence_error_function.h"
#include "momentum/character_sequence_solver/sequence_solver.h"
#include "momentum/character_sequence_solver/sequence_solver_function.h"
#include "momentum/character_solver/collision_error_function.h"
#include "momentum/character_solver/collision_error_function_stateless.h"
#include "momentum/character_solver/gauss_newton_solver_qr.h"
#include "momentum/character_solver/height_error_function.h"
#include "momentum/character_solver/limit_error_function.h"
#include "momentum/character_solver/model_parameters_error_function.h"
#include "momentum/character_solver/plane_error_function.h"
#include "momentum/character_solver/position_error_function.h"
#include "momentum/character_solver/projection_error_function.h"
#include "momentum/character_solver/skeleton_solver_function.h"
#include "momentum/character_solver/skinned_locator_error_function.h"
#include "momentum/character_solver/skinned_locator_triangle_error_function.h"
#include "momentum/common/log.h"
#include "momentum/common/progress_bar.h"
#include "momentum/marker_tracking/glove_utils.h"
#include "momentum/marker_tracking/marker_gap_fill.h"
#include "momentum/marker_tracking/tracker_utils.h"
#include "momentum/math/fmt_eigen.h"
#include "momentum/math/mesh.h"
#include "momentum/solver/solver.h"

using namespace momentum;

namespace momentum {

namespace {

void validateCameraKeypointFrameCount(
    std::span<const CameraKeypointData> cameraKeypointData,
    size_t numFrames) {
  for (size_t cam = 0; cam < cameraKeypointData.size(); ++cam) {
    MT_THROW_IF(
        cameraKeypointData[cam].frameData.size() != numFrames,
        "Camera keypoint data[{}] has {} frames but marker data has {} frames",
        cam,
        cameraKeypointData[cam].frameData.size(),
        numFrames);
  }
}

/// Compute a frame stride for sampling numFrames down to ~targetFrames.
/// Always returns at least 1 to avoid division-by-zero or infinite loops.
/// When greedyMax > 0, the stride is also capped to greedyMax.
size_t computeSampleStride(size_t numFrames, size_t targetFrames, size_t greedyMax = 0) {
  if (targetFrames == 0 || numFrames == 0) {
    return 1;
  }
  size_t stride = (numFrames - 1) / targetFrames;
  if (greedyMax > 0) {
    stride = std::min(stride, greedyMax);
  }
  return std::max(size_t(1), stride);
}

} // namespace

/// Sample representative frames from motion data to maximize parameter variance.
///
/// Uses a greedy algorithm to select frames that are maximally different from each other
/// in parameter space, while filtering out frames with high marker tracking errors.
/// This is useful for calibration where you want to solve on a diverse set of poses
/// rather than all frames.
///
/// @param character The character model used for computing marker errors
/// @param initialMotion Initial motion parameters matrix (parameters x frames)
/// @param markerData Marker observations for each frame
/// @param parameters Set of parameters to consider for variance calculation
/// @param frameStride Only consider every frameStride-th frame as candidates
/// @param numSamples Maximum number of frames to sample
/// @return Vector of frame indices representing the selected keyframes
std::vector<size_t> sampleFrames(
    momentum::Character& character,
    const MatrixXf& initialMotion,
    std::span<const std::vector<momentum::Marker>> markerData,
    const ParameterSet& parameters,
    const size_t frameStride,
    const size_t numSamples) {
  // sample frames so that we get the most variance from the initial input
  const auto numFrames = static_cast<size_t>(initialMotion.cols());
  if (numFrames == 0) {
    return {};
  }
  MT_CHECK(frameStride > 0, "frameStride must be > 0.");
  size_t solvedFrames = (numFrames - 1) / frameStride + 1;
  std::vector<size_t> frameIndices;
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
      if (locatorIdx >= character.locators.size()) {
        continue;
      }
      const auto& locator = character.locators[locatorIdx];
      if (locator.parent >= state.jointState.size()) {
        continue;
      }
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

/// Track motion across multiple frames simultaneously with temporal constraints.
///
/// This is the main global optimization function that solves for both pose parameters
/// and global parameters (scaling, locators, blend shapes) across multiple frames
/// simultaneously. It enforces temporal smoothness constraints and can handle
/// calibration scenarios where identity parameters need to be solved.
///
/// @param markerData Marker observations for each frame
/// @param character The character model with skeleton, locators, and parameter transform
/// @param globalParams Set of global parameters to solve (scaling, locators, etc.)
/// @param initialMotion Initial parameter values (parameters x frames)
/// @param config Tracking configuration settings
/// @param regularizer Weight for regularizing changes to global parameters
/// @param frameStride Process every frameStride-th frame (1 = all frames)
/// @param enforceFloorInFirstFrame Force floor contact constraints in first frame
/// @param firstFramePoseConstraintSet Name of pose constraint set for first frame
/// @return Solved motion parameters matrix (parameters x frames)
Eigen::MatrixXf trackSequence(
    const std::span<const std::vector<Marker>> markerData,
    const Character& character,
    const ParameterSet& globalParams,
    const MatrixXf& initialMotion,
    const TrackingConfig& config,
    float regularizer,
    const size_t frameStride,
    bool enforceFloorInFirstFrame,
    const std::string& firstFramePoseConstraintSet,
    float targetHeightCm,
    std::span<const GloveFrameData> leftGloveData,
    std::span<const GloveFrameData> rightGloveData,
    const std::optional<GloveConfig>& gloveConfig,
    std::span<const CameraKeypointData> cameraKeypointData) {
  // sanity checks
  const size_t numFrames = markerData.size();
  MT_CHECK(numFrames > 0, "Input marker data is empty.");
  MT_CHECK(frameStride > 0, "frameStride must be > 0.");
  validateCameraKeypointFrameCount(cameraKeypointData, numFrames);
  if (!leftGloveData.empty()) {
    MT_CHECK(
        leftGloveData.size() == numFrames,
        "trackSequence: left glove data has {} frames but marker data has {} frames",
        leftGloveData.size(),
        numFrames);
  }
  if (!rightGloveData.empty()) {
    MT_CHECK(
        rightGloveData.size() == numFrames,
        "trackSequence: right glove data has {} frames but marker data has {} frames",
        rightGloveData.size(),
        numFrames);
  }
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
      firstFramePoseConstraintSet,
      targetHeightCm,
      leftGloveData,
      rightGloveData,
      gloveConfig,
      cameraKeypointData);
}

/// Add a blend shape regularizer that penalizes deviations from zero for all
/// blend shape parameters.  Only added for the first solver frame.
void addBlendShapeRegularizer(
    SequenceSolverFunctionT<float>& solverFunc,
    size_t solverFrame,
    size_t solvedFrames,
    const Character& character) {
  const ParameterTransform& pt = character.parameterTransform;
  if (!pt.getBlendShapeParameters().any() || solverFrame != 0) {
    return;
  }
  auto blendShapeConstrFunc = std::make_shared<ModelParametersErrorFunction>(character);
  blendShapeConstrFunc->setWeight(solvedFrames * 0.1f);
  Eigen::VectorXf weights = Eigen::VectorXf::Zero(pt.numAllModelParameters());
  for (Eigen::Index i = 0; i < pt.blendShapeParameters.size(); ++i) {
    if (pt.blendShapeParameters[i] >= 0) {
      weights[pt.blendShapeParameters[i]] = 1.0f;
    }
  }
  blendShapeConstrFunc->setTargetParameters(
      ModelParameters::Zero(pt.numAllModelParameters()), weights);
  solverFunc.addErrorFunction(solverFrame, blendShapeConstrFunc);
}

/// Add 2D keypoint projection constraints from outside-in cameras for one solver frame.
/// Uses ProjectionErrorFunctionT with projectionMatrixRadians() so the residual is in
/// radians (tangent-of-angle), making the weight pixel-independent and naturally scaled
/// relative to marker constraints (in cm).
void addKeypointProjectionConstraints(
    SequenceSolverFunctionT<float>& solverFunc,
    size_t solverFrame,
    size_t iFrame,
    const Character& character,
    float projectionWeight,
    std::span<const CameraKeypointData> cameraKeypointData) {
  if (projectionWeight <= 0.0f) {
    return;
  }
  size_t totalProjectionConstraints = 0;
  size_t totalInvalidProj = 0;
  for (const auto& camData : cameraKeypointData) {
    if (iFrame >= camData.frameData.size() || camData.frameData[iFrame].empty()) {
      continue;
    }
    auto projFunc = std::make_shared<ProjectionErrorFunctionT<float>>(
        character.skeleton, character.parameterTransform);
    for (const auto& obs : camData.frameData[iFrame]) {
      if (obs.confidence <= 0.0f || obs.locatorIndex >= character.locators.size()) {
        continue;
      }
      const auto& locator = character.locators[obs.locatorIndex];
      if (locator.parent >= character.skeleton.joints.size()) {
        continue;
      }
      const auto [M, valid] = projectionMatrixRadians(camData.camera, obs.target);
      if (!valid) {
        totalInvalidProj++;
        continue;
      }
      ProjectionConstraintDataT<float> constr;
      constr.projection = M;
      constr.target = Eigen::Vector2f::Zero();
      constr.parent = locator.parent;
      constr.offset = locator.offset;
      constr.weight = obs.confidence;
      projFunc->addConstraint(constr);
    }
    if (!projFunc->empty()) {
      totalProjectionConstraints += projFunc->getNumConstraints();
      projFunc->setWeight(projectionWeight);
      solverFunc.addErrorFunction(solverFrame, projFunc);
    }
  }
  if (iFrame == 0) {
    MT_LOGI(
        "Frame 0: added {} projection constraints (radians, weight={}), "
        "{} cameras, {} invalid projections",
        totalProjectionConstraints,
        projectionWeight,
        cameraKeypointData.size(),
        totalInvalidProj);
  }
}

/// Add per-frame marker, floor, and glove constraints for one frame of the
/// sequence solver.  Extracted from trackSequence to reduce cyclomatic complexity.
void addSequenceFrameConstraints(
    SequenceSolverFunctionT<float>& solverFunc,
    size_t solverFrame,
    size_t iFrame,
    size_t solvedFrames,
    const Character& character,
    const TrackingConfig& config,
    const std::vector<std::vector<PositionData>>& constrData,
    const std::vector<std::vector<SkinnedLocatorConstraint>>& skinnedConstData,
    const std::vector<SkinnedLocatorTriangleConstraintT<float>>& skinnedLocatorMeshContraints,
    const std::vector<PlaneDataT<float>>& floorConstraints,
    bool enforceFloorInFirstFrame,
    const std::string& firstFramePoseConstraintSet,
    float targetHeightCm,
    const std::optional<GloveConfig>& gloveConfig,
    const std::vector<std::vector<JointToJointPositionDataT<float>>>& leftGlovePosData,
    const std::vector<std::vector<JointToJointOrientationDataT<float>>>& leftGloveOriData,
    const std::vector<std::vector<JointToJointPositionDataT<float>>>& rightGlovePosData,
    const std::vector<std::vector<JointToJointOrientationDataT<float>>>& rightGloveOriData,
    std::span<const CameraKeypointData> cameraKeypointData = {}) {
  auto posConstrWeight = PositionErrorFunction::kLegacyWeight * config.markerWeight;
  if (solverFrame == 0 && (enforceFloorInFirstFrame || !firstFramePoseConstraintSet.empty())) {
    posConstrWeight *= solvedFrames;
  }

  // prepare positional constraints
  if (!constrData.at(iFrame).empty()) {
    auto posConstrFunc = std::make_shared<PositionErrorFunction>(character, config.lossAlpha);
    posConstrFunc->setConstraints(constrData.at(iFrame));
    posConstrFunc->setWeight(posConstrWeight);
    solverFunc.addErrorFunction(solverFrame, posConstrFunc);
  }

  if (!skinnedConstData.at(iFrame).empty()) {
    auto skinnedConstrFunc = std::make_shared<SkinnedLocatorErrorFunction>(character);
    skinnedConstrFunc->setConstraints(skinnedConstData.at(iFrame));
    skinnedConstrFunc->setWeight(posConstrWeight);
    solverFunc.addErrorFunction(solverFrame, skinnedConstrFunc);
  }

  if (!skinnedLocatorMeshContraints.empty() && solverFrame == 0) {
    auto skinnedTriangleConstrFunc =
        std::make_shared<SkinnedLocatorTriangleErrorFunctionT<float>>(character);
    skinnedTriangleConstrFunc->setConstraints(skinnedLocatorMeshContraints);
    skinnedTriangleConstrFunc->setWeight(
        solvedFrames * posConstrWeight * config.meshConstraintWeight);
    solverFunc.addErrorFunction(solverFrame, skinnedTriangleConstrFunc);
  }

  addBlendShapeRegularizer(solverFunc, solverFrame, solvedFrames, character);

  if (targetHeightCm > 0.0f && solverFrame == 0) {
    auto heightConstrFunc = std::make_shared<HeightErrorFunctionT<float>>(
        character, targetHeightCm, Eigen::Vector3f::UnitY(), 10);
    heightConstrFunc->setWeight(solvedFrames * 1.0f);
    solverFunc.addErrorFunction(solverFrame, heightConstrFunc);
  }

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

  if (!firstFramePoseConstraintSet.empty() && solverFrame == 0) {
    if (character.parameterTransform.poseConstraints.count(firstFramePoseConstraintSet) > 0) {
      const auto poseLimits = getPoseConstraintParameterLimits(
          firstFramePoseConstraintSet, character.parameterTransform);
      auto limitConstrFunc = std::make_shared<LimitErrorFunction>(character, poseLimits);
      limitConstrFunc->setWeight(solvedFrames);
      solverFunc.addErrorFunction(solverFrame, limitConstrFunc);
    }
  }

  // add glove constraints for this frame
  if (gloveConfig) {
    addGloveConstraintsToSequenceSolver(
        solverFunc,
        solverFrame,
        character,
        leftGlovePosData,
        leftGloveOriData,
        iFrame,
        gloveConfig->positionWeight,
        gloveConfig->orientationWeight);
    addGloveConstraintsToSequenceSolver(
        solverFunc,
        solverFrame,
        character,
        rightGlovePosData,
        rightGloveOriData,
        iFrame,
        gloveConfig->positionWeight,
        gloveConfig->orientationWeight);
  }

  addKeypointProjectionConstraints(
      solverFunc, solverFrame, iFrame, character, config.projectionWeight, cameraKeypointData);
}

/// Track motion across multiple frames simultaneously for specific frame indices.
///
/// This is the same as the main trackSequence function above, but instead of using
/// a frameStride to sample frames uniformly, it tracks motion only for the specified
/// frame indices. This is particularly useful during calibration when you want to
/// solve on carefully selected keyframes rather than uniformly sampled frames.
///
/// @param markerData Marker observations for each frame
/// @param character The character model with skeleton, locators, and parameter transform
/// @param globalParams Set of global parameters to solve (scaling, locators, etc.)
/// @param initialMotion Initial parameter values (parameters x frames)
/// @param config Tracking configuration settings
/// @param frames Vector of specific frame indices to solve
/// @param regularizer Weight for regularizing changes to global parameters
/// @param enforceFloorInFirstFrame Force floor contact constraints in first frame
/// @param firstFramePoseConstraintSet Name of pose constraint set for first frame
/// @return Solved motion parameters matrix (parameters x frames)
Eigen::MatrixXf trackSequence(
    const std::span<const std::vector<Marker>> markerData,
    const Character& character,
    const ParameterSet& globalParams,
    const MatrixXf& initialMotion,
    const TrackingConfig& config,
    const std::vector<size_t>& frames,
    float regularizer,
    bool enforceFloorInFirstFrame,
    const std::string& firstFramePoseConstraintSet,
    float targetHeightCm,
    std::span<const GloveFrameData> leftGloveData,
    std::span<const GloveFrameData> rightGloveData,
    const std::optional<GloveConfig>& gloveConfig,
    std::span<const CameraKeypointData> cameraKeypointData) {
  // sanity checks
  const size_t numFrames = markerData.size();
  MT_CHECK(numFrames > 0, "Input marker data is empty.");
  MT_CHECK(
      initialMotion.cols() >= numFrames,
      "Number of frames in data {} exceeds input motion columns {}",
      numFrames,
      initialMotion.cols());
  MT_CHECK(
      initialMotion.rows() == character.parameterTransform.numAllModelParameters(),
      "Input motion parameters {} do not match character model parameters {}",
      initialMotion.rows(),
      character.parameterTransform.numAllModelParameters());
  MT_CHECK(
      leftGloveData.empty() || leftGloveData.size() == numFrames,
      "Left glove data has {} frames but marker data has {} frames",
      leftGloveData.size(),
      numFrames);
  MT_CHECK(
      rightGloveData.empty() || rightGloveData.size() == numFrames,
      "Right glove data has {} frames but marker data has {} frames",
      rightGloveData.size(),
      numFrames);
  validateCameraKeypointFrameCount(cameraKeypointData, numFrames);

  const ParameterTransform& pt = character.parameterTransform;

  // universal parameters include "scaling" and "locators" (if exists); pose parameters need to
  // exclude "locators". universalParams is to indicate to the solver which parameters are "global"
  // (ie. not time varying). The input globalParams indicate which parameters within universalParams
  // we want to solve for. globalParams is either a subset or all of universalParams.
  ParameterSet poseParams = pt.getPoseParameters();
  ParameterSet universalParams = pt.getScalingParameters() | pt.getBlendShapeParameters();
  const auto locatorSet =
      pt.getParameterSet("locators", true) | pt.getParameterSet("skinnedLocators", true);
  poseParams &= ~locatorSet;
  universalParams |= locatorSet;

  // Include glove DOFs as universal (not time-varying) parameters
  if (gloveConfig) {
    const auto gloveSet = pt.getParameterSet("gloves", true);
    poseParams &= ~gloveSet;
    universalParams |= gloveSet;
  }

  // Apply caller-specified parameter mask to pose parameters
  if (config.activeParams) {
    poseParams &= *config.activeParams;
  }

  std::vector<momentum::SkinnedLocatorTriangleConstraintT<float>> skinnedLocatorMeshContraints;
  if (!character.skinnedLocators.empty() && character.mesh) {
    skinnedLocatorMeshContraints = createSkinnedLocatorMeshConstraints(character);
    MT_LOGI(
        "Created {} mesh constraints for {} skinned locators",
        skinnedLocatorMeshContraints.size(),
        character.skinnedLocators.size());
  }

  // set up the solver function
  std::vector<size_t> sortedFrames = frames;
  std::sort(sortedFrames.begin(), sortedFrames.end());
  size_t solvedFrames = sortedFrames.size();
  auto solverFunc = SequenceSolverFunction(
      character, character.parameterTransform, universalParams, solvedFrames);

  // floor penetration constraints; we assume the world is y-up and floor is y=0 for mocap data.
  const auto& floorConstraints = createFloorConstraints<float>(
      "Floor_",
      character.locators,
      Vector3f::UnitY(),
      /* y offset */ 0.0f,
      /* weight */ 5.0f);

  // marker constraints
  const auto constrData = createConstraintData(markerData, character.locators);
  const auto skinnedConstData = createSkinnedConstraintData(markerData, character.skinnedLocators);

  // glove constraint data (one per hand × position/orientation)
  std::vector<std::vector<JointToJointPositionDataT<float>>> leftGlovePosData, rightGlovePosData;
  std::vector<std::vector<JointToJointOrientationDataT<float>>> leftGloveOriData, rightGloveOriData;
  if (gloveConfig) {
    leftGlovePosData = createGlovePositionConstraintData(leftGloveData, character, *gloveConfig, 0);
    leftGloveOriData =
        createGloveOrientationConstraintData(leftGloveData, character, *gloveConfig, 0);
    rightGlovePosData =
        createGlovePositionConstraintData(rightGloveData, character, *gloveConfig, 1);
    rightGloveOriData =
        createGloveOrientationConstraintData(rightGloveData, character, *gloveConfig, 1);
  }

  // add per-frame constraint data to the solver
  for (size_t solverFrame = 0; solverFrame < solvedFrames; ++solverFrame) {
    const size_t& iFrame = sortedFrames[solverFrame];
    if ((constrData.at(iFrame).size() + skinnedConstData.at(iFrame).size()) >
        markerData[iFrame].size() * config.minVisPercent) {
      addSequenceFrameConstraints(
          solverFunc,
          solverFrame,
          iFrame,
          solvedFrames,
          character,
          config,
          constrData,
          skinnedConstData,
          skinnedLocatorMeshContraints,
          floorConstraints,
          enforceFloorInFirstFrame,
          firstFramePoseConstraintSet,
          targetHeightCm,
          gloveConfig,
          leftGlovePosData,
          leftGloveOriData,
          rightGlovePosData,
          rightGloveOriData,
          cameraKeypointData);
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
  solverOptions.regularization = config.regularization;

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

/// Update glove constraints for both hands for a given frame.
void updateGloveConstraintsForBothHands(
    std::optional<GloveErrorFunctions>& left,
    std::optional<GloveErrorFunctions>& right,
    size_t iFrame) {
  if (left) {
    updateGloveConstraintsForFrame(*left, iFrame);
  }
  if (right) {
    updateGloveConstraintsForFrame(*right, iFrame);
  }
}

/// Check if the global transform is zero by checking if any rigid parameters are non-zero.
///
/// This is used to determine whether initialization is needed for pose tracking.
/// If all rigid parameters are zero, we need to solve for an initial rigid transform.
///
/// @param dof The parameter vector to check
/// @param rigidParams Parameter set defining which parameters are rigid/global
/// @return True if global transform is zero (needs initialization), false otherwise
bool isGlobalTransformZero(const Eigen::VectorXf& dof, const ParameterSet& rigidParams) {
  for (Eigen::Index i = 0; i < dof.size(); ++i) {
    if (rigidParams.test(i) && dof[i] != 0.0f) {
      return false;
    }
  }
  return true;
}

/// Track poses independently per frame with fixed character identity.
///
/// This is the main production tracking function used after character calibration.
/// It solves each frame independently using a per-frame optimizer, which makes it
/// robust to tracking failures. The character identity (scaling, locators, blend shapes)
/// is fixed from calibration and only pose parameters are solved.
///
/// @param markerData Marker observations for each frame
/// @param character The character model with calibrated identity parameters
/// @param globalParams Fixed global parameters (scaling, locators, etc.) from calibration
/// @param config Tracking configuration settings
/// @param frameStride Process every frameStride-th frame (1 = all frames)
/// @return Solved motion parameters matrix (parameters x frames) with fixed identity
Eigen::MatrixXf trackPosesPerframe(
    const std::span<const std::vector<Marker>> markerData,
    const Character& character,
    const ModelParameters& globalParams,
    const TrackingConfig& config,
    const size_t frameStride,
    std::span<const GloveFrameData> leftGloveData,
    std::span<const GloveFrameData> rightGloveData,
    const std::optional<GloveConfig>& gloveConfig) {
  const size_t numFrames = markerData.size();
  MT_CHECK(numFrames > 0, "Input marker data is empty.");
  MT_CHECK(frameStride > 0, "frameStride must be > 0.");

  // Generate frame indices from stride
  std::vector<size_t> frameIndices;
  for (size_t iFrame = 0; iFrame < numFrames; iFrame += frameStride) {
    frameIndices.push_back(iFrame);
  }

  // Convert globalParams to initial motion matrix
  MatrixXf initialMotion(globalParams.v.size(), numFrames);
  for (size_t i = 0; i < numFrames; ++i) {
    initialMotion.col(i) = globalParams.v;
  }

  // Determine if tracking should be continuous (temporal coherence)
  bool isContinuous = (frameStride < 5);

  return trackPosesForFrames(
      markerData,
      character,
      initialMotion,
      config,
      frameIndices,
      isContinuous,
      leftGloveData,
      rightGloveData,
      gloveConfig);
}

/// Solve for rigid parameters only as initialization for a frame.
///
/// When starting tracking (or re-starting for non-continuous mode), we first solve
/// for only the rigid body parameters to get a rough global pose before solving
/// for the full set of pose parameters.
///
/// @return false (to clear the needsInit flag)
bool solveRigidInitialization(
    GaussNewtonSolverQR& solver,
    GaussNewtonSolverQROptions& solverOptions,
    Eigen::VectorXf& dof,
    const ParameterTransform& pt,
    const ParameterSet& poseParams,
    const std::shared_ptr<ModelParametersErrorFunction>& smoothConstrFunc,
    const TrackingConfig& config,
    bool isContinuous,
    size_t iFrame) {
  MT_LOGI_IF(config.debug && isContinuous, "Solving for an initial rigid pose at frame {}", iFrame);

  // Set up different config for initialization
  solverOptions.maxIterations = 50; // make sure it converges
  solver.setOptions(solverOptions);
  solver.setEnabledParameters(pt.getRigidParameters());
  if (smoothConstrFunc) {
    smoothConstrFunc->setWeight(0.0); // turn off smoothing - it doesn't affect rigid dofs
  }

  solver.solve(dof);

  // Recover solver config
  solverOptions.maxIterations = config.maxIter;
  solver.setOptions(solverOptions);
  solver.setEnabledParameters(poseParams);
  if (smoothConstrFunc) {
    smoothConstrFunc->setWeight(config.smoothing);
  }

  return false;
}

/// Track poses independently for specific frame indices with fixed character identity.
///
/// Similar to trackPosesPerframe, but only solves for the specified frame indices
/// rather than processing frames with a stride. This is particularly useful during
/// calibration when you want to solve poses only for carefully selected keyframes
/// that have been sampled for maximum parameter variance.
///
/// @param markerData Marker observations for each frame
/// @param character The character model with fixed identity parameters
/// @param initialMotion Initial parameter values (parameters x frames)
/// @param config Tracking configuration settings
/// @param frameIndices Vector of specific frame indices to solve
/// @param isContinuous Whether to use temporal coherence between frames
/// @return Solved motion parameters matrix (parameters x frames) with poses for selected frames
Eigen::MatrixXf trackPosesForFrames(
    const std::span<const std::vector<Marker>> markerData,
    const Character& character,
    const MatrixXf& initialMotion,
    const TrackingConfig& config,
    const std::vector<size_t>& frameIndices,
    bool isContinuous,
    std::span<const GloveFrameData> leftGloveData,
    std::span<const GloveFrameData> rightGloveData,
    const std::optional<GloveConfig>& gloveConfig,
    const std::string& progressLabel) {
  const size_t numFrames = markerData.size();
  MT_CHECK(numFrames > 0, "Input marker data is empty.");
  MT_CHECK(
      initialMotion.cols() >= numFrames,
      "Number of frames in data {} exceeds input motion columns {}",
      numFrames,
      initialMotion.cols());
  MT_CHECK(
      initialMotion.rows() == character.parameterTransform.numAllModelParameters(),
      "Input motion parameters {} do not match character model parameters {}",
      initialMotion.rows(),
      character.parameterTransform.numAllModelParameters());
  MT_CHECK(
      leftGloveData.empty() || leftGloveData.size() == numFrames,
      "Left glove data has {} frames but marker data has {} frames",
      leftGloveData.size(),
      numFrames);
  MT_CHECK(
      rightGloveData.empty() || rightGloveData.size() == numFrames,
      "Right glove data has {} frames but marker data has {} frames",
      rightGloveData.size(),
      numFrames);

  const ParameterTransform& pt = character.parameterTransform;

  std::vector<size_t> sortedFrames = frameIndices;
  std::sort(sortedFrames.begin(), sortedFrames.end());

  // pose parameters need to exclude "locators"
  ParameterSet poseParams = pt.getPoseParameters();
  const auto& locatorSet = pt.parameterSets.find("locators");
  if (locatorSet != pt.parameterSets.end()) {
    poseParams &= ~locatorSet->second;
  }

  // Exclude glove DOFs from per-frame pose solving
  if (gloveConfig) {
    poseParams &= ~pt.getParameterSet("gloves", true);
  }

  // Apply caller-specified parameter mask
  if (config.activeParams) {
    poseParams &= *config.activeParams;
  }

  // set up the solver
  auto solverFunc = SkeletonSolverFunction(character, pt);
  GaussNewtonSolverQROptions solverOptions;
  solverOptions.maxIterations = config.maxIter;
  solverOptions.minIterations = 2;
  solverOptions.doLineSearch = false;
  solverOptions.verbose = config.debug;
  solverOptions.threshold = 1.f;
  solverOptions.regularization = config.regularization;
  auto solver = GaussNewtonSolverQR(solverOptions, &solverFunc);
  solver.setEnabledParameters(poseParams);

  // parameter limits constraint
  auto limitConstrFunc = std::make_shared<LimitErrorFunction>(character);
  limitConstrFunc->setWeight(0.1);
  solverFunc.addErrorFunction(limitConstrFunc);

  // positional constraint function for markers
  auto posConstrFunc = std::make_shared<PositionErrorFunction>(character, config.lossAlpha);
  posConstrFunc->setWeight(PositionErrorFunction::kLegacyWeight * config.markerWeight);
  solverFunc.addErrorFunction(posConstrFunc);

  auto skinnedLocatorPosConstrFunc = std::make_shared<SkinnedLocatorErrorFunction>(character);
  skinnedLocatorPosConstrFunc->setWeight(
      PositionErrorFunction::kLegacyWeight * config.markerWeight);
  solverFunc.addErrorFunction(skinnedLocatorPosConstrFunc);

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

  // Pre-process marker gaps (interpolate temporary gaps, blend off permanent dropouts).
  // Creates a mutable copy since the input span is const.
  auto processedMarkerData = std::vector<std::vector<Marker>>(markerData.begin(), markerData.end());
  preprocessMarkerGaps(processedMarkerData, config.gapFillConfig);

  // marker constraint data (uses gap-filled markers)
  auto constrData = createConstraintData(processedMarkerData, character.locators);
  auto skinnedConstrData =
      createSkinnedConstraintData(processedMarkerData, character.skinnedLocators);

  // smoothness constraint only for the joints and exclude global dofs because the global transform
  // needs to be accurate (may not matter in practice?)
  // Only use temporal smoothness if isContinuous is true
  std::shared_ptr<ModelParametersErrorFunction> smoothConstrFunc;
  if (isContinuous) {
    smoothConstrFunc = std::make_shared<ModelParametersErrorFunction>(
        character, poseParams & ~pt.getRigidParameters());
    smoothConstrFunc->setWeight(config.smoothing);
    solverFunc.addErrorFunction(smoothConstrFunc);
  }

  // add collision error
  std::shared_ptr<CollisionErrorFunction> collisionErrorFunction;
  if (config.collisionErrorWeight != 0 && character.collision != nullptr) {
    collisionErrorFunction = std::make_shared<CollisionErrorFunction>(character);
    collisionErrorFunction->setWeight(config.collisionErrorWeight);
    solverFunc.addErrorFunction(collisionErrorFunction);
  }

  // set up glove error functions (registered once, constraints swapped per frame)
  std::optional<GloveErrorFunctions> leftGloveFuncs, rightGloveFuncs;
  if (gloveConfig) {
    leftGloveFuncs =
        setupGloveErrorFunctions(solverFunc, character, leftGloveData, *gloveConfig, 0);
    rightGloveFuncs =
        setupGloveErrorFunctions(solverFunc, character, rightGloveData, *gloveConfig, 1);
  }

  // initialize parameters to contain identity information
  // the identity fields will be used but untouched during optimization
  // globalParams could also be repurposed to pass in initial pose value
  Eigen::VectorXf dof = initialMotion.col(sortedFrames.empty() ? 0 : sortedFrames[0]);
  size_t solverFrame = 0;
  double priorError = 0.0;
  double error = 0.0;

  // Use the initial global transform if it's not zero
  bool needsInit = isGlobalTransformZero(dof, pt.getRigidParameters());

  MatrixXf outMotion = initialMotion;
  Eigen::Index outputIndex = 0;
  { // scope the ProgressBar so it returns
    ProgressBar progress(progressLabel, sortedFrames.size());
    for (const auto iFrame : sortedFrames) {
      // For continuous tracking, keep the solved dof from previous frame (temporal coherence)
      // For non-continuous tracking, always start from initial motion (independent solving)
      if (!isContinuous) {
        dof = initialMotion.col(iFrame);
        needsInit = true;
      }
      // For continuous tracking, dof is preserved from previous iteration (or initial value)

      if ((constrData.at(iFrame).size() + skinnedConstrData.at(iFrame).size()) >
          processedMarkerData[iFrame].size() * config.minVisPercent) {
        // add positional constraints
        posConstrFunc->clearConstraints(); // clear constraint data from the previous frame
        posConstrFunc->setConstraints(constrData.at(iFrame));

        skinnedLocatorPosConstrFunc->clearConstraints();
        skinnedLocatorPosConstrFunc->setConstraints(skinnedConstrData.at(iFrame));

        // update glove constraints for this frame
        updateGloveConstraintsForBothHands(leftGloveFuncs, rightGloveFuncs, iFrame);

        // initialization: solve only for the rigid parameters as preprocessing
        if (needsInit) {
          needsInit = solveRigidInitialization(
              solver,
              solverOptions,
              dof,
              pt,
              poseParams,
              smoothConstrFunc,
              config,
              isContinuous,
              iFrame);
        }

        // set smoothness target as the last pose for continuous tracking
        if (smoothConstrFunc) {
          smoothConstrFunc->setTargetParameters(dof, smoothConstrFunc->getTargetWeights());
        }

        priorError += solverFunc.getError(dof);
        error += solver.solve(dof);
        ++solverFrame;
      }

      // store result
      while (outputIndex <= iFrame) {
        outMotion.col(outputIndex++) = dof;
      }
      progress.increment();
    }

    while (outputIndex < numFrames) {
      outMotion.col(outputIndex++) = dof;
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

namespace {

/// Compute character height after calibration.
///
/// This function computes the height of a calibrated character by:
/// 1. Setting scale and shape parameters from identity
/// 2. Zeroing out all pose parameters
/// 3. Running full skinning with shape
/// 4. Computing the distance between largest and smallest Y values across all mesh vertices
///
/// @param character Character model with calibrated identity
/// @param identity Identity parameters containing scale and shape
/// @return Height of the character in world units
float computeCharacterHeight(const Character& character, const ModelParameters& identity) {
  if (!character.mesh) {
    MT_LOGW("Character has no mesh, cannot compute height");
    return 0.0f;
  }

  const ParameterTransform& pt = character.parameterTransform;

  // Create a model parameters vector with scale and shape, but zero pose
  ModelParameters neutralParams = identity;

  // Zero out all pose parameters
  const ParameterSet poseParams = pt.getPoseParameters();
  for (size_t i = 0; i < neutralParams.v.size(); ++i) {
    if (poseParams.test(i)) {
      neutralParams.v[i] = 0.0f;
    }
  }

  // Create skeleton state with the neutral (zero pose) parameters
  const auto jointParams = pt.apply(neutralParams.v);
  SkeletonState state;
  state.set(jointParams, character.skeleton, false);

  // Apply blend shape skinning to get the deformed mesh
  Mesh skinnedMesh;
  skinWithBlendShapes(character, state, neutralParams, skinnedMesh);

  // Find min and max Y values across all vertices
  if (skinnedMesh.vertices.empty()) {
    MT_LOGW("Skinned mesh has no vertices, cannot compute height");
    return 0.0f;
  }

  float minY = std::numeric_limits<float>::max();
  float maxY = std::numeric_limits<float>::lowest();

  for (const auto& vertex : skinnedMesh.vertices) {
    minY = std::min(minY, vertex.y());
    maxY = std::max(maxY, vertex.y());
  }

  const float height = maxY - minY;
  return height;
}

// Log per-locator mesh distances with skin offsets for debugging calibration.
void logSkinnedLocatorMeshDistances(const Character& character, const std::string& label) {
  if (character.skinnedLocators.empty() || !character.mesh) {
    return;
  }
  const auto distances = computeSkinnedLocatorMeshDistances(character);
  MT_LOGI("{} skinned locator mesh distances:", label);
  for (size_t i = 0; i < distances.size(); ++i) {
    const auto& [name, dist] = distances[i];
    const float skinOff = character.skinnedLocators[i].skinOffset;
    MT_LOGI("  {}: {:.4f} cm (skinOffset: {:.2f} cm)", name, dist, skinOff);
  }
}

void logKeypointObservationSummary(
    std::span<const CameraKeypointData> cameraKeypointData,
    float projectionWeight) {
  if (cameraKeypointData.empty() || projectionWeight <= 0.0f) {
    return;
  }
  size_t totalObs = 0;
  for (const auto& camData : cameraKeypointData) {
    for (const auto& frame : camData.frameData) {
      totalObs += frame.size();
    }
  }
  MT_LOGI(
      "2D keypoint constraints: {} cameras, {} total observations, "
      "projectionWeight={}",
      cameraKeypointData.size(),
      totalObs,
      projectionWeight);
}

} // namespace

Character addSkinnedLocatorParametersToTransform(Character character) {
  if (character.skinnedLocators.empty()) {
    return character;
  }

  std::vector<bool> activeSkinnedLocators(character.skinnedLocators.size(), true);
  std::vector<std::string> locatorNames;
  for (const auto& sl : character.skinnedLocators) {
    locatorNames.push_back(sl.name);
  }
  std::tie(character.parameterTransform, character.parameterLimits) = addSkinnedLocatorParameters(
      character.parameterTransform, character.parameterLimits, activeSkinnedLocators, locatorNames);
  return character;
}

void logKeypointReprojectionDiagnostic(
    const Character& character,
    const ParameterTransform& transform,
    const Eigen::MatrixXf& motion,
    std::span<const CameraKeypointData> cameraKeypointData,
    const CalibrationConfig& config,
    const std::string& label) {
  if (!config.debug || cameraKeypointData.empty() || config.projectionWeight <= 0.0f) {
    return;
  }

  const size_t diagFrame = 0;
  SkeletonState skelState;
  skelState.set(
      character.parameterTransform.apply(
          ModelParameters(motion.col(diagFrame).head(transform.numAllModelParameters()))),
      character.skeleton);

  MT_LOGI("=== Keypoint reprojection diagnostic (frame {}, {}) ===", diagFrame, label);
  for (size_t iCam = 0; iCam < cameraKeypointData.size(); ++iCam) {
    const auto& camData = cameraKeypointData[iCam];
    if (diagFrame >= camData.frameData.size() || camData.frameData[diagFrame].empty()) {
      continue;
    }
    size_t nObs = 0, nSkipped = 0;
    float sumPxErr = 0, maxPxErr = 0;
    float sumRadErr = 0, maxRadErr = 0;
    for (const auto& obs : camData.frameData[diagFrame]) {
      if (obs.confidence <= 0.0f || obs.locatorIndex >= character.locators.size()) {
        nSkipped++;
        continue;
      }
      const auto& locator = character.locators[obs.locatorIndex];
      if (locator.parent >= skelState.jointState.size()) {
        nSkipped++;
        continue;
      }
      const Eigen::Vector3f p_world =
          skelState.jointState[locator.parent].transform * locator.offset;
      const auto [projected, valid] = camData.camera.project(p_world);
      if (!valid) {
        nSkipped++;
        continue;
      }
      const float pxErr = (projected.head<2>() - obs.target).norm();
      sumPxErr += pxErr;
      maxPxErr = std::max(maxPxErr, pxErr);
      const auto [M, mValid] = projectionMatrixRadians(camData.camera, obs.target);
      float radErr = 0;
      if (mValid) {
        const Eigen::Vector4f ph(p_world[0], p_world[1], p_world[2], 1.0f);
        const Eigen::Vector3f h = M * ph;
        if (std::abs(h[2]) > 1e-6f) {
          radErr = Eigen::Vector2f(h[0] / h[2], h[1] / h[2]).norm();
        }
      }
      sumRadErr += radErr;
      maxRadErr = std::max(maxRadErr, radErr);
      nObs++;
    }
    MT_LOGI(
        "cam[{}]: {} obs, {} skip, mean={:.1f}px/{:.4f}rad, max={:.1f}px/{:.4f}rad",
        iCam,
        nObs,
        nSkipped,
        nObs > 0 ? sumPxErr / nObs : 0.0f,
        nObs > 0 ? sumRadErr / nObs : 0.0f,
        maxPxErr,
        maxRadErr);
  }
}

void calibrateModel(
    const std::span<const std::vector<Marker>> markerData,
    const CalibrationConfig& config,
    Character& character,
    ModelParameters& identity,
    const std::array<float, 3>& regularizerWeights,
    std::span<const GloveFrameData> leftGloveData,
    std::span<const GloveFrameData> rightGloveData,
    const std::optional<GloveConfig>& gloveConfig,
    std::span<const CameraKeypointData> cameraKeypointData,
    std::vector<size_t>* selectedFrameIndices,
    Eigen::MatrixXf* selectedFrameMotion) {
  const size_t numFrames = markerData.size();
  MT_THROW_IF(numFrames < 2, "Calibration requires at least 2 frames, got {}.", numFrames);
  MT_THROW_IF(
      identity.v.size() != character.parameterTransform.numAllModelParameters(),
      "Input identity parameters {} do not match character parameters {}",
      identity.v.size(),
      character.parameterTransform.numAllModelParameters());
  MT_THROW_IF(
      !leftGloveData.empty() && leftGloveData.size() != numFrames,
      "Left glove data has {} frames but marker data has {} frames",
      leftGloveData.size(),
      numFrames);
  MT_THROW_IF(
      !rightGloveData.empty() && rightGloveData.size() != numFrames,
      "Right glove data has {} frames but marker data has {} frames",
      rightGloveData.size(),
      numFrames);
  validateCameraKeypointFrameCount(cameraKeypointData, numFrames);

  MT_THROW_IF(config.calibFrames == 0, "calibFrames must be > 0.");
  if (numFrames < config.calibFrames) {
    MT_LOGW(
        "Only {} frames available for calibration (requested {}); using all frames.",
        numFrames,
        config.calibFrames);
  }

  logKeypointObservationSummary(cameraKeypointData, config.projectionWeight);

  MT_LOGI(
      "calibrateModel: {} frames, {} locators, {} skinnedLocators, {} joints, "
      "leftGlove={}, rightGlove={}, keypoints={} cameras",
      numFrames,
      character.locators.size(),
      character.skinnedLocators.size(),
      character.skeleton.joints.size(),
      leftGloveData.size(),
      rightGloveData.size(),
      cameraKeypointData.size());

  // uniformly sample frames for calibration
  const size_t frameStride = computeSampleStride(numFrames, config.calibFrames);

  // create a solving character with markers as bones
  Character solvingCharacter = character;
  solvingCharacter = createLocatorCharacter(solvingCharacter, "locator_");
  if (!solvingCharacter.skinnedLocators.empty()) {
    solvingCharacter = addSkinnedLocatorParametersToTransform(solvingCharacter);
  }

  // Add glove bones to the solving character if glove data is provided
  if (gloveConfig && (!leftGloveData.empty() || !rightGloveData.empty())) {
    solvingCharacter = createGloveCharacter(solvingCharacter, *gloveConfig);
  }

  // Extended quantities are for the solvingCharacter, which includes locators as bones
  // w/o Extended quantities are for character with fixed locators
  const ParameterTransform& transformExtended = solvingCharacter.parameterTransform;
  const ParameterTransform& transform = character.parameterTransform;

  ParameterSet locatorSet = transformExtended.getParameterSet("locators", true) |
      transformExtended.getParameterSet("skinnedLocators", true);
  ParameterSet gloveSet = transformExtended.getParameterSet("gloves", true);
  ParameterSet calibBodySetExtended;
  ParameterSet calibBodySet;
  if (config.globalScaleOnly) {
    const size_t paramIndex = transform.getParameterIdByName("scale_global");
    const size_t paramIndexExt = transformExtended.getParameterIdByName("scale_global");
    MT_THROW_IF(
        paramIndex == kInvalidIndex || paramIndexExt == kInvalidIndex,
        "Can't calibrate global scale since it is not defined in the parameter list!");

    calibBodySetExtended.set(paramIndexExt);
    calibBodySet.set(paramIndex);
  } else {
    calibBodySetExtended = transformExtended.getScalingParameters();
    calibBodySet = transform.getScalingParameters();

    if (config.calibShape) {
      calibBodySetExtended |= transformExtended.getBlendShapeParameters();
      calibBodySet |= transform.getBlendShapeParameters();
    }
  }

  // special trackingConfig for initialization: zero out smoothness and collision
  TrackingConfig trackingConfig;
  trackingConfig.minVisPercent = config.minVisPercent;
  trackingConfig.lossAlpha = config.lossAlpha;
  trackingConfig.maxIter = config.maxIter;
  trackingConfig.regularization = config.regularization;
  trackingConfig.debug = config.debug;
  trackingConfig.meshConstraintWeight = config.meshConstraintWeight;
  trackingConfig.projectionWeight = config.projectionWeight;

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
          firstFrame,
          false, // Not continuous for calibration keyframes
          {},
          {},
          std::nullopt,
          "Calibrating first frame");
      motion.topRows(transform.numAllModelParameters()) = trackSequence(
          markerData,
          character,
          calibBodySet,
          motion.topRows(transform.numAllModelParameters()),
          trackingConfig,
          firstFrame,
          regularizerWeights.at(0),
          config.enforceFloorInFirstFrame,
          config.firstFramePoseConstraintSet,
          config.targetHeightCm); // still solving a subset
      std::tie(identity.v, character.locators, character.skinnedLocators) =
          extractIdAndLocatorsFromParams(motion.col(0), solvingCharacter, character);

      // track sequence with selected stride. we need to sample at least config.calibFrames frames
      // so make sure we have a stride that allows that
      const size_t sampleStride =
          computeSampleStride(numFrames, config.calibFrames, config.greedySampling);

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

    } else {
      motion.topRows(transform.numAllModelParameters()) =
          trackPosesPerframe(markerData, character, identity, trackingConfig, frameStride);

      // Build uniform-stride frame indices.
      for (size_t i = 0; i < numFrames; i += frameStride) {
        frameIndices.push_back(i);
      }
    }

    // Solve for identity and poses with fixed locators, initialized with solved poses.
    // This works using "character" because additional parameters for the locators are appended at
    // the end, so the indices work out using topRows() without special treatment.
    motion.topRows(transform.numAllModelParameters()) = trackSequence(
        markerData,
        character,
        calibBodySet,
        motion.topRows(transform.numAllModelParameters()),
        trackingConfig,
        frameIndices,
        regularizerWeights.at(0),
        config.enforceFloorInFirstFrame,
        config.firstFramePoseConstraintSet,
        config.targetHeightCm);
  }

  if (selectedFrameIndices != nullptr) {
    *selectedFrameIndices = frameIndices;
  }

  logKeypointReprojectionDiagnostic(
      character, transform, motion, cameraKeypointData, config, "after Stage 0");

  // Solve everything together for a few iterations
  for (size_t iIter = 0; iIter < config.majorIter; ++iIter) {
    MT_LOGI_IF(config.debug, "Iteration {} of calibration", iIter);

    motion = trackSequence(
        markerData,
        solvingCharacter,
        locatorSet | calibBodySetExtended | gloveSet,
        motion,
        trackingConfig,
        frameIndices,
        regularizerWeights.at(1),
        config.enforceFloorInFirstFrame,
        config.firstFramePoseConstraintSet,
        config.targetHeightCm,
        leftGloveData,
        rightGloveData,
        gloveConfig,
        cameraKeypointData);
    // extract solving results to identity and character so we can pass them to trackPosesPerframe
    // below.
    std::tie(identity.v, character.locators, character.skinnedLocators) =
        extractIdAndLocatorsFromParams(motion.col(0), solvingCharacter, character);
    if (config.calibShape && solvingCharacter.blendShape) {
      auto blendShapeParams =
          extractBlendWeights(solvingCharacter.parameterTransform, ModelParameters(motion.col(0)));
      MT_LOGI("Solved for blend shape coeffs: {}", blendShapeParams.v.transpose());
      // NOTE: We intentionally do NOT bake the mesh here.  The caller
      // (Python) is responsible for calling bake_blend_shape() which
      // both updates the mesh and strips the blend shape parameters
      // from the parameter transform.
    }

    // The sequence solve above could get stuck with euler singularity but per-frame solve could get
    // it out. Pass in the first frame from previous solve as a better initial guess than the zero
    // pose.
    motion.topRows(transform.numAllModelParameters()) = trackPosesForFrames(
        markerData,
        character,
        motion.topRows(transform.numAllModelParameters()),
        trackingConfig,
        frameIndices,
        false, // Not continuous for calibration keyframes
        {},
        {},
        std::nullopt,
        "Calibrating per-frame");
  }

  // Finally, fine tune marker offsets with fix identity.
  MT_LOGI_IF(config.debug, "Fine-tune marker offsets");

  motion = trackSequence(
      markerData,
      solvingCharacter,
      locatorSet | gloveSet,
      motion,
      trackingConfig,
      frameIndices,
      regularizerWeights.at(2),
      config.enforceFloorInFirstFrame,
      config.firstFramePoseConstraintSet,
      config.targetHeightCm,
      leftGloveData,
      rightGloveData,
      gloveConfig,
      cameraKeypointData);
  std::tie(identity.v, character.locators, character.skinnedLocators) =
      extractIdAndLocatorsFromParams(motion.col(0), solvingCharacter, character);

  // Bake solved glove offsets into the character skeleton, following the same
  // pattern as locator extraction above. No-op if gloveConfig is nullopt.
  bakeGloveOffsetsFromParams(
      character, ModelParameters(motion.col(0)), solvingCharacter, gloveConfig);

  // TODO: A hack to return the solved first frame as initialization for tracking later.
  identity.v = motion.col(0).head(transform.numAllModelParameters());

  // Extract solved motion for the selected calibration frames.
  if (selectedFrameMotion != nullptr && !frameIndices.empty()) {
    const auto nParams = transform.numAllModelParameters();
    selectedFrameMotion->resize(nParams, frameIndices.size());
    for (size_t i = 0; i < frameIndices.size(); ++i) {
      selectedFrameMotion->col(i) = motion.col(frameIndices[i]).head(nParams);
    }
  }

  // Log final per-locator mesh distances
  logSkinnedLocatorMeshDistances(character, "Final");

  // Log the calibrated character height
  const float height = computeCharacterHeight(character, identity);
  MT_LOGI("Calibrated character height: {:.4f} cm", height);
}

/// Calibrate only locator positions with fixed character identity parameters.
///
/// This is a specialized calibration function that only solves for locator positions
/// while keeping all other character parameters (scaling, blend shapes) fixed.
/// This is useful when you have reliable identity parameters and only need to
/// fine-tune marker positions.
///
/// @param markerData Marker observations for each frame
/// @param config Calibration configuration settings
/// @param identity Fixed identity parameters (scaling, blend shapes)
/// @param character Character model to calibrate (locators modified in-place)
void calibrateLocators(
    const std::span<const std::vector<Marker>> markerData,
    const CalibrationConfig& config,
    const ModelParameters& identity,
    Character& character) {
  const size_t numFrames = markerData.size();
  MT_THROW_IF(numFrames < 2, "Calibration requires at least 2 frames, got {}.", numFrames);
  MT_THROW_IF(
      identity.v.size() != character.parameterTransform.numAllModelParameters(),
      "Input identity parameters {} do not match character parameters {}",
      identity.v.size(),
      character.parameterTransform.numAllModelParameters());

  MT_THROW_IF(config.calibFrames == 0, "calibFrames must be > 0.");
  if (numFrames < config.calibFrames) {
    MT_LOGW(
        "Only {} frames available for calibration (requested {}); using all frames.",
        numFrames,
        config.calibFrames);
  }

  // create a solving character with locators as bones
  Character solvingCharacter = createLocatorCharacter(character, "locator_");

  // Extended quantities are for the solvingCharacter, which includes locators as bones
  // w/o Extended quantities are for character with fixed locators
  const ParameterTransform& transformExtended = solvingCharacter.parameterTransform;
  const ParameterTransform& transform = character.parameterTransform;

  ParameterSet locatorSet = transformExtended.parameterSets.find("locators")->second;

  // special trackingConfig for initialization: zero out smoothness and collision
  TrackingConfig trackingConfig;
  trackingConfig.minVisPercent = config.minVisPercent;
  trackingConfig.lossAlpha = config.lossAlpha;
  trackingConfig.maxIter = config.maxIter;
  trackingConfig.regularization = config.regularization;
  trackingConfig.debug = config.debug;
  trackingConfig.meshConstraintWeight = config.meshConstraintWeight;

  // only keep one motion for both character and solvingCharacter; no need to duplicate.
  // identity information will be initialized and updated in the motion matrix throughout all the
  // solves.
  MatrixXf motion = MatrixXf::Zero(transformExtended.numAllModelParameters(), numFrames);
  CharacterParameters fullParams;

  // pick frames to solve
  std::vector<size_t> frameIndices;
  if (config.greedySampling > 0) {
    // track sequence with selected stride. we need to sample at least config.calibFrames frames
    // so make sure we have a stride that allows that
    const size_t sampleStride =
        computeSampleStride(numFrames, config.calibFrames, config.greedySampling);
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
  } else {
    // uniformly sample frames for calibration
    const size_t frameStride = computeSampleStride(numFrames, config.calibFrames);
    for (size_t fi = 0; fi < numFrames; fi += frameStride) {
      frameIndices.emplace_back(fi);
    }

    motion.topRows(transform.numAllModelParameters()) =
        trackPosesPerframe(markerData, character, identity, trackingConfig, frameStride);
  }

  // Iterate for a few times
  for (size_t iIter = 0; iIter < config.majorIter; ++iIter) {
    MT_LOGI_IF(config.debug, "Iteration {} of locator calibration", iIter);

    // Solve only for poses using solved locators; it helps to adjust poses to get out of bad
    // solutions.
    motion.topRows(transform.numAllModelParameters()) = trackPosesForFrames(
        markerData,
        character,
        motion.topRows(transform.numAllModelParameters()),
        trackingConfig,
        frameIndices,
        false, // Not continuous for calibration keyframes
        {},
        {},
        std::nullopt,
        "Calibrating locators per-frame");

    // Solve for both markers and poses.
    // TODO: add a small regularization to prevent too large a change
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
    // Extract solved locators
    fullParams.pose = motion.col(0);
    character.locators = extractLocatorsFromCharacter(solvingCharacter, fullParams);
  }
}

/// Refine existing motion by smoothing and optionally recalibrating identity/locators.
///
/// This is a post-processing function that takes an already tracked motion and improves it
/// by applying temporal smoothness constraints, and optionally recalibrating character identity
/// or locator positions. It uses the sequence solver to enforce temporal coherence across frames.
///
/// @param markerData Marker observations for each frame
/// @param motion Initial motion to refine (parameters x frames)
/// @param config Refinement configuration settings
/// @param character Character model (may be modified if calibration is enabled)
/// @return Refined motion parameters matrix with improved temporal consistency
MatrixXf refineMotion(
    std::span<const std::vector<momentum::Marker>> markerData,
    const MatrixXf& motion,
    const RefineConfig& config,
    momentum::Character& character) {
  MT_THROW_IF(markerData.empty(), "Input marker data is empty.");
  MT_THROW_IF(
      markerData.size() != motion.cols(),
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

    std::tie(std::ignore, character.locators, character.skinnedLocators) =
        extractIdAndLocatorsFromParams(newMotion.col(0), solvingCharacter, character);
    newMotion.conservativeResize(numParams, Eigen::NoChange_t::NoChange);
  }

  return newMotion;
}

/// Compute average and maximum marker tracking errors across all frames.
///
/// This is a utility function for evaluating tracking quality by measuring the
/// Euclidean distance between observed marker positions and their corresponding
/// locator positions on the character. It provides both average error per frame
/// and the maximum error encountered across all markers and frames.
///
/// This function supports both regular locators and skinned locators.
///
/// @param markerData Marker observations for each frame
/// @param motion Solved motion parameters matrix (parameters x frames)
/// @param character Character model with locators
/// @return Pair of (average_error, max_error) in world units
std::pair<float, float> getLocatorError(
    std::span<const std::vector<momentum::Marker>> markerData,
    const MatrixXf& motion,
    momentum::Character& character) {
  const size_t numFrames = markerData.size();
  MT_CHECK(numFrames > 0, "Input marker data is empty.");
  MT_CHECK(
      motion.cols() >= numFrames,
      "Motion has {} columns but marker data has {} frames",
      motion.cols(),
      numFrames);
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

  // map skinned locator name to its index
  std::unordered_map<std::string, size_t> skinnedLocatorLookup;
  for (size_t i = 0; i < character.skinnedLocators.size(); i++) {
    skinnedLocatorLookup[character.skinnedLocators[i].name] = i;
  }

  SkeletonState state;

  // go over all frames and pose the locators and compute the error
  double error = 0.0;
  double maxError = 0.0;
  size_t frameNum = 0.0;
  std::string markerName;
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

      std::optional<Vector3f> locatorPos;

      // First try regular locators
      auto query = locatorLookup.find(jMarker.name);
      if (query != locatorLookup.end()) {
        size_t locatorIdx = query->second;
        if (locatorIdx < character.locators.size()) {
          const auto& locator = character.locators[locatorIdx];
          if (locator.parent < state.jointState.size()) {
            locatorPos = state.jointState[locator.parent].transform * locator.offset;
          }
        }
      }

      // If not found, try skinned locators
      if (!locatorPos.has_value()) {
        auto skinnedQuery = skinnedLocatorLookup.find(jMarker.name);
        if (skinnedQuery != skinnedLocatorLookup.end()) {
          size_t locatorIdx = skinnedQuery->second;
          if (locatorIdx < character.skinnedLocators.size()) {
            const auto& skinnedLocator = character.skinnedLocators[locatorIdx];
            locatorPos = getSkinnedLocatorPosition(
                skinnedLocator, skinnedLocator.position, character.inverseBindPose, state);
          }
        }
      }

      if (!locatorPos.has_value()) {
        continue;
      }

      const Vector3f diff = *locatorPos - jMarker.pos.cast<float>();
      const float markerError = diff.norm();
      frameError += markerError;
      if (markerError > maxError) {
        maxError = markerError;
        frameNum = iFrame;
        markerName = jMarker.name;
      }
      validMarkers++;
    }

    if (validMarkers > 0) {
      error += frameError / validMarkers;
    }
  }
  MT_LOGI("Max marker error: {} at frame {} for marker {}", maxError, frameNum, markerName);
  return {static_cast<float>(error / numFrames), static_cast<float>(maxError)};
}

} // namespace momentum
