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
#include <momentum/character/pose_shape.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/types.h>
#include <momentum/math/fwd.h>

namespace momentum {

template <typename T>
struct CharacterT {
  // non-optional components of a character
  Skeleton skeleton;
  ParameterTransform parameterTransform;

  // optional components
  ParameterLimits parameterLimits;
  LocatorList locators;
  Mesh_u mesh;
  SkinWeights_u skinWeights;
  PoseShape_u poseShapes;
  CollisionGeometry_u collision;
  BlendShape_const_p blendShape;
  BlendShapeBase_const_p faceExpressionBlendShape;

  TransformationList inverseBindPose;
  std::vector<size_t> jointMap;

  std::string name;

  // character constructor
  CharacterT();
  ~CharacterT();

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
      const momentum::TransformationList& inverseBindPose = {});

  CharacterT(const CharacterT& c);
  CharacterT& operator=(const CharacterT& rhs);

  // create a simplified character from the current one.  This simplified character will
  // only have the joints that are actually affected by the passed-in parameters.
  CharacterT simplify(const ParameterSet& enabledParameters = ParameterSet().flip()) const;

  // Create a smaller character that just has the requested active joints.
  CharacterT simplifySkeleton(const std::vector<bool>& activeJoints) const;

  // Create a smaller character that just has the requested parameters.
  CharacterT simplifyParameterTransform(const ParameterSet& parameterSet) const;

  // remap SkinWeights from original character to simplified version
  SkinWeights remapSkinWeights(const SkinWeights& skinWeights, const CharacterT& originalCharacter)
      const;

  // remap limits from original character to simplified version
  ParameterLimits remapParameterLimits(
      const ParameterLimits& limits,
      const CharacterT& originalCharacter) const;

  // remap locators from original character to simplified version
  LocatorList remapLocators(const LocatorList& locs, const CharacterT& originalCharacter) const;

  // Go from a set of parameters to the set of joints that those parameters affect:
  std::vector<bool> parametersToActiveJoints(const ParameterSet& parameterSet) const;

  // Go from a set of joints to all the parameters that affect those joints:
  ParameterSet activeJointsToParameters(const std::vector<bool>& activeJoints) const;

  // Returns bindpose parameters for this character. "Bind pose" is the rest pose for a character
  // model when all model parameters and joint offsets are zero. When a FK pass is done on the bind
  // pose, it would result in a rest pose skeleton states
  CharacterParameters bindPose() const;

  // Helpers to init parameterTransform, reset jointMap and initialize the inverse bind pose
  void initParameterTransform();
  void resetJointMap();
  // Initializes the inverse bind pose of this character. "Inverse bind pose" is a set of affine
  // transformations for each joint that can map skeleton states (world pose of joints) to the
  // joint parameters (local pose of joints in parent joint coordinate frame).
  void initInverseBindPose();

  // make sure character parameters are split into parameters/offsets according to the parameterset
  static CharacterParameters splitParameters(
      const CharacterT& character,
      const CharacterParameters& parameters,
      const ParameterSet& parameterSet);

  // Create new CharacterT object that is a copy of this object, but with blendShape_in as
  // blendShape and resized paramTransform.
  // param[in] blendShape_in - BlendShape object to be set as blendShape.
  // param[in] maxBlendShapes - number of blendshape params to add to paramTransform. If it is <= 0,
  // all blendshapes from blendShape_in are added.
  // param[in] overwriteBaseShape (optional) - if true, base shape of blendShape_in will be set to
  // new character's mesh
  CharacterT withBlendShape(
      BlendShape_p blendShape_in,
      Eigen::Index maxBlendShapes,
      bool overwriteBaseShape = true) const;

  // Create new CharacterT object that is a copy of this object, but with blendShape_in as
  // faceExpressionBlendShape and resized paramTransform.
  // param[in] blendShape_in - BlendShapeBase object to set as faceExpressionBlendShape
  // param[in] maxBlendShapes (optional) - number of blendshape params to add to paramTransform.
  // If it is <= 0, all blendshapes from blendShape_in are added.
  CharacterT withFaceExpressionBlendShape(
      BlendShapeBase_const_p blendShape_in,
      Eigen::Index maxBlendShapes = -1) const;

  // Set blendShape_in as blendShape and resize paramTransform accordingly.
  // param[in] blendShape_in - BlendShape object to be set as blendShape.
  // param[in] maxBlendShapes - number of blendshape params to add to paramTransform.
  // If it is <= 0, all blendshapes from blendShape_in are added.
  // param[in] overwriteBaseShape (optional)- if true, base shape of blendShape_in will be set to
  // this character's mesh
  void addBlendShape(
      const BlendShape_p& blendShape_in,
      Eigen::Index maxBlendShapes,
      bool overwriteBaseShape = true);

  // Set blendShape_in as faceExpressionBlendShape and resize paramTransform accordingly.
  // param[in] blendShape_in - BlendShapeBase object to be set as faceExpressionBlendShape
  // param[in] maxBlendShapes (optional) - number of blendshape params to add to paramTransform.
  // If it is <= 0,  all blendshapes from blendShape_in are added.
  void addFaceExpressionBlendShape(
      const BlendShapeBase_const_p& blendShape_in,
      Eigen::Index maxBlendShapes = -1);

  // Bake the blend shapes from the passed-in model parameters into the mesh
  // and strip the blend shapes out of the parameter transform:
  CharacterT bakeBlendShape(const ModelParameters& modelParams) const;
  CharacterT bakeBlendShape(const BlendWeights& blendWeights) const;

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
  CharacterT bake(
      const ModelParameters& modelParams,
      bool bakeBlendShapes = true,
      bool bakeScales = true) const;
};

} // namespace momentum
