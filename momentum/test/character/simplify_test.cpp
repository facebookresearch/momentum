/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include "momentum/character/types.h"

#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

TEST(Simplify, SimplifyParameterTransform) {
  const auto character_orig = withTestBlendShapes(createTestCharacter());
  ParameterSet activeParams;
  ASSERT_GE(character_orig.parameterTransform.numAllModelParameters(), 6);
  activeParams.set(2);
  activeParams.set(5);
  ASSERT_GE(character_orig.parameterTransform.blendShapeParameters(0), 0);
  activeParams.set(character_orig.parameterTransform.blendShapeParameters(0));

  const auto character_new = character_orig.simplifyParameterTransform(activeParams);
  // Should not change skeleton:
  ASSERT_EQ(character_orig.skeleton.getJointNames(), character_new.skeleton.getJointNames());
  ASSERT_EQ(character_new.parameterTransform.numAllModelParameters(), 3);
  ASSERT_EQ(character_orig.parameterTransform.name[2], character_new.parameterTransform.name[0]);
  ASSERT_EQ(character_orig.parameterTransform.name[5], character_new.parameterTransform.name[1]);

  ModelParameters modelParams_orig =
      ModelParameters::Zero(character_orig.parameterTransform.numAllModelParameters());
  modelParams_orig[2] = 0.5f;
  modelParams_orig[5] = 1.0f;

  ModelParameters modelParams_new =
      ModelParameters::Zero(character_new.parameterTransform.numAllModelParameters());
  modelParams_new[0] = 0.5;
  modelParams_new[1] = 1.0f;

  // Joint params should be identical:
  const JointParameters jointParams_orig =
      character_orig.parameterTransform.apply(modelParams_orig);
  const JointParameters jointParams_new = character_new.parameterTransform.apply(modelParams_new);

  ASSERT_LE((jointParams_new.v - jointParams_orig.v).norm(), 1e-5f);

  const SkeletonState skelState_orig(jointParams_orig, character_orig.skeleton);
  const SkeletonState skelState_new(jointParams_new, character_new.skeleton);

  // Skeletons should match:
  ASSERT_EQ(skelState_new.jointState.size(), skelState_orig.jointState.size());
  for (size_t i = 0; i < skelState_orig.jointState.size(); ++i) {
    ASSERT_LE(
        (skelState_new.jointState[i].transform.toAffine3().matrix() -
         skelState_orig.jointState[i].transform.toAffine3().matrix())
            .norm(),
        1e-5f);
  }

  // Blend shape parameters got updated:
  ASSERT_EQ(2, character_new.parameterTransform.blendShapeParameters(0));
  ASSERT_EQ(-1, character_new.parameterTransform.blendShapeParameters(1));
}

TEST(Simplify, SimplifySkeleton) {
  const auto character_orig = createTestCharacter();

  std::vector<bool> activeBones(character_orig.skeleton.joints.size());
  activeBones[0] = true;
  activeBones[2] = true;

  const auto character_new = character_orig.simplifySkeleton(activeBones);

  // Make sure we removed the middle joint:
  ASSERT_EQ(character_new.skeleton.joints.size(), 2);
  ASSERT_EQ(character_orig.skeleton.joints[0].name, character_new.skeleton.joints[0].name);
  ASSERT_EQ(character_orig.skeleton.joints[2].name, character_new.skeleton.joints[1].name);

  // Shouldn't change the parameter transform:
  ASSERT_EQ(character_orig.parameterTransform.name, character_new.parameterTransform.name);

  ASSERT_EQ(character_new.skeleton.joints.size(), 2);

  // Apply some parameters that don't affect the second joint:
  ModelParameters modelParameters =
      ModelParameters::Zero(character_new.parameterTransform.numAllModelParameters());
  modelParameters.v << 1.0, 1.0, 1.0, pi(), 0.0, -pi(), 0.1, 0, 0, -pi();

  JointParameters jointParameters_orig = character_orig.parameterTransform.apply(modelParameters);
  JointParameters jointParameters_new = character_new.parameterTransform.apply(modelParameters);

  const SkeletonState skelState_orig(jointParameters_orig, character_orig.skeleton);
  const SkeletonState skelState_new(jointParameters_new, character_new.skeleton);

  // Make sure all the locators are in the same places:
  for (const auto& l_orig : character_orig.locators) {
    auto l_new_itr = std::find_if(
        character_new.locators.begin(), character_new.locators.end(), [&](const Locator& l_new) {
          return l_new.name == l_orig.name;
        });
    ASSERT_NE(character_new.locators.end(), l_new_itr);

    const Eigen::Vector3f p_orig =
        skelState_orig.jointState[l_orig.parent].transform * l_orig.offset;
    const Eigen::Vector3f p_new =
        skelState_new.jointState[l_new_itr->parent].transform * l_new_itr->offset;
    ASSERT_LE((p_new - p_orig).norm(), 1e-4f);
  }

  ASSERT_TRUE(bool(character_orig.mesh));
  ASSERT_TRUE(bool(character_new.mesh));

  ASSERT_EQ(character_orig.mesh->vertices.size(), character_new.mesh->vertices.size());

  // Make sure the mesh matches:
  Mesh posedMesh_orig = *character_orig.mesh;
  applySSD(
      character_orig.inverseBindPose,
      *character_orig.skinWeights,
      *character_orig.mesh,
      skelState_orig,
      posedMesh_orig);

  Mesh posedMesh_new = *character_new.mesh;
  applySSD(
      character_new.inverseBindPose,
      *character_new.skinWeights,
      *character_new.mesh,
      skelState_new,
      posedMesh_new);

  for (size_t i = 0; i < character_orig.mesh->vertices.size(); ++i) {
    ASSERT_LE((character_orig.mesh->vertices[i] - character_new.mesh->vertices[i]).norm(), 1e-4f);
  }
}
