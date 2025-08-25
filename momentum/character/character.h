/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/collision_geometry.h>
#include <momentum/character/fwd.h>
#include <momentum/character/locator.h>
#include <momentum/character/parameter_limits.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skinned_locator.h>
#include <momentum/character/types.h>
#include <momentum/math/fwd.h>

namespace momentum {

/// A character model with skeletal structure, mesh, and optional components.
template <typename T>
struct CharacterT {
  /// @{ @name Required components

  /// Skeletal structure defining the character's joints and hierarchy
  Skeleton skeleton;

  /// Maps model parameters to joint parameters
  ParameterTransform parameterTransform;

  /// @}

  /// @{ @name Optional components

  /// Constraints on model parameters
  ParameterLimits parameterLimits;

  /// Points of interest attached to joints
  LocatorList locators;

  /// Points of interest attached to joints, with skinning weights
  SkinnedLocatorList skinnedLocators;

  /// 3D mesh representing the character's surface
  Mesh_u mesh;

  /// Defines how mesh vertices are influenced by joints
  SkinWeights_u skinWeights;

  /// Pose-dependent shape corrections
  PoseShape_u poseShapes;

  /// Collision volumes for the character
  CollisionGeometry_u collision;

  /// Shape variations that can be blended
  BlendShape_const_p blendShape;

  /// Facial expression blend shapes
  BlendShapeBase_const_p faceExpressionBlendShape;

  /// Inverse of the bind pose transformations for each joint
  TransformationList inverseBindPose;

  /// Maps from original joint indices to simplified joint indices
  std::vector<size_t> jointMap;

  /// Character identifier
  std::string name;

  /// @}

  /// Default constructor
  CharacterT();

  /// Destructor
  ~CharacterT();

  /// Constructs a character with the specified components
  ///
  /// @param s Skeleton defining joint hierarchy
  /// @param pt Parameter transform mapping model parameters to joint parameters
  /// @param pl Optional parameter limits/constraints
  /// @param l Optional locators attached to joints
  /// @param m Optional mesh representing the character's surface
  /// @param sw Optional skin weights defining how mesh vertices are influenced by joints
  /// @param cg Optional collision geometry
  /// @param bs Optional pose-dependent shape corrections
  /// @param blendShapes Optional shape variations that can be blended
  /// @param faceExpressionBlendShapes Optional facial expression blend shapes
  /// @param nameIn Optional character identifier
  /// @param inverseBindPose Optional inverse bind pose transformations
  CharacterT(
      const Skeleton& s,
      const ParameterTransform& pt,
      const ParameterLimits& pl = ParameterLimits(),
      const LocatorList& l = LocatorList(),
      const Mesh* m = nullptr,
      const SkinWeights* sw = nullptr,
      const CollisionGeometry* cg = nullptr,
      const PoseShape* bs = nullptr,
      BlendShape_const_p blendShapes = {},
      BlendShapeBase_const_p faceExpressionBlendShapes = {},
      const std::string& nameIn = "",
      const momentum::TransformationList& inverseBindPose = {},
      const SkinnedLocatorList& skinnedLocators = {});

  /// Copy constructor
  CharacterT(const CharacterT& c);

  /// Move constructor
  CharacterT(CharacterT&& c) noexcept;

  /// Copy assignment operator
  CharacterT& operator=(const CharacterT& rhs);

  /// Move assignment operator
  CharacterT& operator=(CharacterT&& rhs) noexcept;

  /// Creates a simplified character with only joints affected by the specified parameters
  ///
  /// @param activeParams Parameters to keep (defaults to all parameters)
  /// @return A new character with simplified skeleton and parameter transform
  [[nodiscard]] CharacterT simplify(const ParameterSet& activeParams = ParameterSet().flip()) const;

  /// Creates a simplified character with only the specified joints
  ///
  /// @param activeJoints Boolean vector indicating which joints to keep
  /// @return A new character with only the requested joints
  [[nodiscard]] CharacterT simplifySkeleton(const std::vector<bool>& activeJoints) const;

  /// Creates a simplified character with only the specified parameters
  ///
  /// @param parameterSet Set of parameters to keep
  /// @return A new character with only the requested parameters
  [[nodiscard]] CharacterT simplifyParameterTransform(const ParameterSet& parameterSet) const;

  /// Remaps skin weights from original character to simplified version
  ///
  /// @param skinWeights Original skin weights
  /// @param originalCharacter Original character containing the full joint hierarchy
  /// @return Remapped skin weights for the simplified character
  [[nodiscard]] SkinWeights remapSkinWeights(
      const SkinWeights& skinWeights,
      const CharacterT& originalCharacter) const;

  /// Remaps parameter limits from original character to simplified version
  ///
  /// @param limits Original parameter limits
  /// @param originalCharacter Original character containing the full joint hierarchy
  /// @return Remapped parameter limits for the simplified character
  [[nodiscard]] ParameterLimits remapParameterLimits(
      const ParameterLimits& limits,
      const CharacterT& originalCharacter) const;

  /// Remaps locators from original character to simplified version
  ///
  /// @param locs Original locators
  /// @param originalCharacter Original character containing the full joint hierarchy
  /// @return Remapped locators for the simplified character
  [[nodiscard]] LocatorList remapLocators(
      const LocatorList& locs,
      const CharacterT& originalCharacter) const;

  [[nodiscard]] SkinnedLocatorList remapSkinnedLocators(
      const SkinnedLocatorList& locs,
      const CharacterT& originalCharacter) const;

  /// Determines which joints are affected by the specified parameters
  ///
  /// @param parameterSet Set of parameters to check
  /// @return Boolean vector indicating which joints are affected by the parameters
  [[nodiscard]] std::vector<bool> parametersToActiveJoints(const ParameterSet& parameterSet) const;

  /// Determines which parameters affect the specified joints
  ///
  /// @param activeJoints Boolean vector indicating which joints to check
  /// @return Set of parameters that affect the specified joints
  [[nodiscard]] ParameterSet activeJointsToParameters(const std::vector<bool>& activeJoints) const;

  /// Returns parameters representing the character's bind pose
  ///
  /// The bind pose is the rest pose when all model parameters and joint offsets are zero.
  /// When forward kinematics is applied to the bind pose, it results in the rest pose skeleton.
  [[nodiscard]] CharacterParameters bindPose() const;

  /// Initializes the parameter transform with the correct dimensions for this character
  void initParameterTransform();

  /// Resets the joint map to identity (each joint maps to itself)
  void resetJointMap();

  /// Initializes the inverse bind pose transformations
  ///
  /// The inverse bind pose is a set of affine transformations for each joint that
  /// map from world space to local joint space in the bind pose configuration.
  void initInverseBindPose();

  /// Splits character parameters into active and inactive components
  ///
  /// @param character Character to use for the parameter transform
  /// @param parameters Input parameters to split
  /// @param parameterSet Set indicating which parameters are active
  /// @return Parameters with active parameters zeroed in pose and applied to offsets
  [[nodiscard]] static CharacterParameters splitParameters(
      const CharacterT& character,
      const CharacterParameters& parameters,
      const ParameterSet& parameterSet);

  /// Creates a new character with the specified blend shapes
  ///
  /// @param blendShape_in Blend shapes to add to the character
  /// @param maxBlendShapes Maximum number of blend shape parameters to add (use all if <= 0)
  /// @return A new character with the specified blend shapes
  [[nodiscard]] CharacterT withBlendShape(
      BlendShape_const_p blendShape_in,
      Eigen::Index maxBlendShapes) const;

  /// Creates a new character with the specified face expression blend shapes
  ///
  /// @param blendShape_in Face expression blend shapes to add to the character
  /// @param maxBlendShapes Maximum number of blend shape parameters to add (use all if <= 0)
  /// @return A new character with the specified face expression blend shapes
  [[nodiscard]] CharacterT withFaceExpressionBlendShape(
      BlendShapeBase_const_p blendShape_in,
      Eigen::Index maxBlendShapes = -1) const;

  /// Adds blend shapes to this character
  ///
  /// @param blendShape_in Blend shapes to add to the character
  /// @param maxBlendShapes Maximum number of blend shape parameters to add (use all if <= 0)
  void addBlendShape(const BlendShape_const_p& blendShape_in, Eigen::Index maxBlendShapes);

  /// Adds face expression blend shapes to this character
  ///
  /// @param blendShape_in Face expression blend shapes to add to the character
  /// @param maxBlendShapes Maximum number of blend shape parameters to add (use all if <= 0)
  void addFaceExpressionBlendShape(
      const BlendShapeBase_const_p& blendShape_in,
      Eigen::Index maxBlendShapes = -1);

  /// Creates a new character with blend shapes baked into the mesh
  ///
  /// @param modelParams Model parameters containing blend shape weights
  /// @return A new character with blend shapes baked into the mesh
  [[nodiscard]] CharacterT bakeBlendShape(const ModelParameters& modelParams) const;

  /// Creates a new character with blend shapes baked into the mesh
  ///
  /// @param blendWeights Blend shape weights to apply
  /// @return A new character with blend shapes baked into the mesh
  [[nodiscard]] CharacterT bakeBlendShape(const BlendWeights& blendWeights) const;

  /// Generic "bake-out" for turning a character into self-contained geometry.
  ///
  /// @param[in] modelParams Current pose/scale/blend-shape parameters.
  /// @param[in] bakeBlendShapes Set true (default) to apply blend-shape deltas and remove their
  /// parameters from the character.
  /// @param[in] bakeScales Set true (default) to evaluate the posed skeleton, run
  /// Linear-Blend-Skinning once, and remove all scaling parameters from the character.
  ///
  /// The returned character contains a static mesh with all requested deformations baked in, while
  /// still supporting any parameters you elected to keep.
  [[nodiscard]] CharacterT bake(
      const ModelParameters& modelParams,
      bool bakeBlendShapes = true,
      bool bakeScales = true) const;
};

} // namespace momentum
