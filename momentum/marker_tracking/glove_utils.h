/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/types.h>
#include <momentum/character_sequence_solver/fwd.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/joint_to_joint_orientation_error_function.h>
#include <momentum/character_solver/joint_to_joint_position_error_function.h>
#include <momentum/math/types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace momentum {

/// Single glove sensor observation for one finger joint in one frame.
///
/// Represents a measurement from a data glove sensor, providing position
/// and orientation of a finger joint in the glove's local coordinate frame.
struct GloveSensorObservation {
  /// Skeleton joint name (e.g. "b_l_thumb0").
  std::string jointName;

  /// Position in glove-local frame.
  Vector3f position = Vector3f::Zero();

  /// Orientation in glove-local frame.
  Eigen::Quaternionf orientation = Eigen::Quaternionf::Identity();

  /// Whether this observation is valid (false if sensor data is missing/occluded).
  bool valid = true;
};

/// Per-frame collection of glove sensor observations.
using GloveFrameData = std::vector<GloveSensorObservation>;

/// Configuration for glove constraints.
///
/// Controls how glove data is integrated into the solver, including
/// constraint weights and which wrist joints to attach glove bones to.
struct GloveConfig {
  /// Weight for position constraints between glove and finger joints.
  /// Scaled internally by PositionErrorFunction::kLegacyWeight for consistency
  /// with marker constraint weighting.
  float positionWeight = 1.0f;

  /// Weight for orientation constraints between glove and finger joints.
  /// Scaled internally by OrientationErrorFunction::kLegacyWeight for consistency
  /// with marker constraint weighting.
  float orientationWeight = 1.0f;

  /// Names of the left and right wrist joints in the skeleton.
  std::array<std::string, 2> wristJointNames = {"l_wrist", "r_wrist"};
};

/// Calibrated glove-to-wrist offset for one hand.
///
/// Stores the solved translation and rotation offset between the glove's
/// coordinate frame and the wrist joint's coordinate frame.
struct GloveOffset {
  /// Translation offset from wrist to glove origin.
  Vector3f translation = Vector3f::Zero();

  /// Rotation offset as Euler angles (X, Y, Z) in radians.
  Vector3f rotationEulerXYZ = Vector3f::Zero();
};

/// Add glove joints to the skeleton with given offsets but no model parameters.
///
/// For each wrist joint specified in the config, adds a child joint with the
/// given translationOffset and preRotation (defaulting to zero/identity).
/// Does NOT add model parameters or register a parameter set.
///
/// @param[in] character Source character to extend (taken by value for move semantics).
/// @param[in] cfg Glove configuration specifying wrist joint names.
/// @param[in] offsets Per-hand translation and rotation offsets to bake into the joints.
/// @param[in] prefix Name prefix for the added glove joints.
/// @return The modified character with glove bones added to the skeleton.
Character addGloveBones(
    Character character,
    const GloveConfig& cfg,
    const std::array<GloveOffset, 2>& offsets = {},
    const std::string& prefix = "glove_");

/// Add 6-DOF model parameters for existing glove bones.
///
/// Takes a character that already has glove bones (from addGloveBones()) and
/// adds TX/TY/TZ/RX/RY/RZ model parameters for each glove bone. Registers
/// a "gloves" parameter set containing all glove DOFs.
///
/// @param[in] character Character with glove bones already in the skeleton (taken by value for move
/// semantics).
/// @param[in] cfg Glove configuration specifying wrist joint names.
/// @param[in] prefix Name prefix used when adding glove joints.
/// @return The modified character with glove model parameters added.
Character addGloveCalibrationParameters(
    Character character,
    const GloveConfig& cfg,
    const std::string& prefix = "glove_");

/// Add glove bones with 6-DOF model parameters (convenience wrapper).
///
/// Calls addGloveBones() then addGloveCalibrationParameters(). Equivalent to
/// the original createGloveCharacter() behavior.
///
/// @param[in] src Source character to extend.
/// @param[in] cfg Glove configuration specifying wrist joint names.
/// @param[in] prefix Name prefix for the added glove joints.
/// @return A new character with glove bones and model parameters added.
Character createGloveCharacter(
    const Character& src,
    const GloveConfig& cfg,
    const std::string& prefix = "glove_");

/// Extract calibrated glove offsets from solved parameters.
///
/// Reads the translation and rotation DOFs from the glove bones in a
/// character that was created by createGloveCharacter(), returning the
/// solved offsets for each hand.
///
/// @param[in] gloveChar Character with glove bones (from createGloveCharacter()).
/// @param[in] params Solved character parameters containing glove offsets.
/// @param[in] cfg Glove configuration used when creating the character.
/// @return Array of two GloveOffset values, one per hand (left=0, right=1).
std::array<GloveOffset, 2> extractGloveOffsetsFromCharacter(
    const Character& gloveChar,
    const CharacterParameters& params,
    const GloveConfig& cfg);

/// Bake solved glove offsets from calibration parameters into the skeleton.
///
/// Reads the solved TX/TY/TZ/RX/RY/RZ from the glove bones in
/// solvingCharacter's parameter transform, converts them to joint
/// translationOffset and preRotation, and adds (or updates) glove bones
/// in the target character. After this call the character has glove bones
/// with the calibrated offset baked in, but no glove model parameters.
///
/// @param[in,out] character Character to add/update glove bones on.
/// @param[in] solvedParams Model parameters from the calibration solve.
/// @param[in] solvingCharacter Character with glove calibration parameters.
/// @param[in] cfg Glove configuration specifying wrist joint names (no-op if nullopt).
void bakeGloveOffsetsFromParams(
    Character& character,
    const ModelParameters& solvedParams,
    const Character& solvingCharacter,
    const std::optional<GloveConfig>& cfg);

/// Convert per-frame glove observations to JointToJoint position constraints.
///
/// For each frame and each valid observation, creates a constraint between the
/// finger joint (source) and the glove bone (reference), with the observation's
/// position as the target in the glove bone's coordinate frame.
///
/// @param[in] gloveData Per-frame glove sensor observations.
/// @param[in] character Character with glove bones (from createGloveCharacter()).
/// @param[in] cfg Glove configuration.
/// @param[in] handIndex Hand index (0=left, 1=right).
/// @return Vector of per-frame constraint vectors.
std::vector<std::vector<JointToJointPositionDataT<float>>> createGlovePositionConstraintData(
    std::span<const GloveFrameData> gloveData,
    const Character& character,
    const GloveConfig& cfg,
    size_t handIndex);

/// Convert per-frame glove observations to JointToJoint orientation constraints.
///
/// For each frame and each valid observation, creates a constraint between the
/// finger joint (source) and the glove bone (reference), with the observation's
/// orientation as the target relative orientation.
///
/// @param[in] gloveData Per-frame glove sensor observations.
/// @param[in] character Character with glove bones (from createGloveCharacter()).
/// @param[in] cfg Glove configuration.
/// @param[in] handIndex Hand index (0=left, 1=right).
/// @return Vector of per-frame constraint vectors.
std::vector<std::vector<JointToJointOrientationDataT<float>>> createGloveOrientationConstraintData(
    std::span<const GloveFrameData> gloveData,
    const Character& character,
    const GloveConfig& cfg,
    size_t handIndex);

/// Add glove error functions to a SequenceSolverFunction for one frame.
///
/// Creates JointToJointPosition/OrientationErrorFunction instances, populates
/// them from pre-built constraint data for the given frame, and adds them to
/// the solver at the specified solver frame index. Called from the per-frame
/// loop in trackSequence().
///
/// @param[in,out] solverFunc Sequence solver to add error functions to.
/// @param[in] solverFrame Frame index within the solver.
/// @param[in] character Character with glove bones.
/// @param[in] posData Pre-built per-frame position constraint data.
/// @param[in] oriData Pre-built per-frame orientation constraint data.
/// @param[in] iFrame Index into the constraint data arrays (source frame).
/// @param[in] posWeight Position constraint weight.
/// @param[in] oriWeight Orientation constraint weight.
void addGloveConstraintsToSequenceSolver(
    SequenceSolverFunctionT<float>& solverFunc,
    size_t solverFrame,
    const Character& character,
    const std::vector<std::vector<JointToJointPositionDataT<float>>>& posData,
    const std::vector<std::vector<JointToJointOrientationDataT<float>>>& oriData,
    size_t iFrame,
    float posWeight,
    float oriWeight);

/// Holds per-frame glove error functions for the single-frame solver.
///
/// Used by trackPosesForFrames() to register error functions once and then
/// swap constraint data per frame via updateGloveConstraintsForFrame().
struct GloveErrorFunctions {
  /// Position error function registered with the solver.
  std::shared_ptr<JointToJointPositionErrorFunctionT<float>> posFunc;

  /// Orientation error function registered with the solver.
  std::shared_ptr<JointToJointOrientationErrorFunctionT<float>> oriFunc;

  /// Pre-built per-frame position constraint data.
  std::vector<std::vector<JointToJointPositionDataT<float>>> posData;

  /// Pre-built per-frame orientation constraint data.
  std::vector<std::vector<JointToJointOrientationDataT<float>>> oriData;
};

/// Create and register glove error functions for the per-frame solver.
///
/// Creates error function instances, adds them to the SkeletonSolverFunction,
/// and builds the per-frame constraint data. Returns nullopt if the glove data
/// is empty or the glove bone is not found.
///
/// @param[in,out] solverFunc Skeleton solver to register error functions with.
/// @param[in] character Character with glove bones.
/// @param[in] gloveData Per-frame glove sensor observations.
/// @param[in] cfg Glove configuration.
/// @param[in] handIndex Hand index (0=left, 1=right).
/// @return GloveErrorFunctions if successfully set up, nullopt otherwise.
std::optional<GloveErrorFunctions> setupGloveErrorFunctions(
    SkeletonSolverFunctionT<float>& solverFunc,
    const Character& character,
    std::span<const GloveFrameData> gloveData,
    const GloveConfig& cfg,
    size_t handIndex);

/// Swap per-frame glove constraint data on already-registered error functions.
///
/// Clears existing constraints and loads the constraint data for the specified
/// frame index. Called in the per-frame loop of trackPosesForFrames().
///
/// @param[in,out] funcs Error functions to update.
/// @param[in] iFrame Frame index into the constraint data arrays.
void updateGloveConstraintsForFrame(GloveErrorFunctions& funcs, size_t iFrame);

} // namespace momentum
