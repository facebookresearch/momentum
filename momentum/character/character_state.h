/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/locator_state.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/math/fwd.h>

namespace momentum {

/// Represents the complete state of a character at a specific point in time.
///
/// Stores the character's parameters, skeleton state, locator positions,
/// skinned mesh, and collision geometry state. This class is used for
/// character animation and rendering.
template <typename T>
struct CharacterStateT {
  /// Model parameters (`pose`) and joint parameter offsets driving this state.
  CharacterParameters parameters;

  SkeletonState skeletonState;

  LocatorState locatorState;

  /// Skinned mesh; null when constructed/updated with `updateMesh=false` or
  /// when the reference character has no mesh or skin weights.
  Mesh_u meshState;

  /// Collision geometry state; null when constructed/updated with
  /// `updateCollision=false` or when the reference character has no collision
  /// geometry.
  CollisionGeometryState_u collisionState;

  CharacterStateT();

  /// Creates a deep copy of another character state, including owned
  /// `meshState` and `collisionState`.
  explicit CharacterStateT(const CharacterStateT& other);

  CharacterStateT(CharacterStateT&& c) noexcept;

  /// Copy assignment is deleted; use the explicit copy constructor instead.
  CharacterStateT& operator=(const CharacterStateT& rhs) = delete;

  CharacterStateT& operator=(CharacterStateT&& rhs) noexcept;

  ~CharacterStateT();

  /// Creates a character state in `referenceCharacter`'s bind pose.
  ///
  /// @param updateMesh If true, allocate and populate `meshState` by skinning
  ///   `referenceCharacter.mesh` (requires non-null mesh and skin weights).
  /// @param updateCollision If true, allocate and populate `collisionState`
  ///   from `referenceCharacter.collision` (requires non-null collision).
  explicit CharacterStateT(
      const CharacterT<T>& referenceCharacter,
      bool updateMesh = true,
      bool updateCollision = true);

  /// Creates a character state by applying `parameters` to `referenceCharacter`.
  ///
  /// @param applyLimits If true, clamp passive joint parameters to the
  ///   character's `parameterLimits` after applying the parameter transform.
  /// @see set() for the meaning of `updateMesh` and `updateCollision` and
  ///   for the requirements on `parameters.pose` and `parameters.offsets`.
  CharacterStateT(
      const CharacterParameters& parameters,
      const CharacterT<T>& referenceCharacter,
      bool updateMesh = true,
      bool updateCollision = true,
      bool applyLimits = true);

  /// Updates the character state by applying `parameters` to `referenceCharacter`.
  ///
  /// @pre `parameters.pose.size() == referenceCharacter.parameterTransform.numAllModelParameters()`
  /// @pre `parameters.offsets` is either empty or has size
  ///   `referenceCharacter.parameterTransform.numJointParameters()`. When
  ///   empty, it is initialized to zeros.
  ///
  /// @param updateMesh If true, allocate and populate `meshState` by skinning
  ///   `referenceCharacter.mesh` (no-op if mesh or skin weights are null).
  ///   When pose blend shapes or blend shapes are present on the character,
  ///   they are applied before linear skinning.
  /// @param updateCollision If true, allocate and populate `collisionState`
  ///   from `referenceCharacter.collision` (no-op if collision is null).
  /// @param applyLimits If true, clamp passive joint parameters to the
  ///   character's `parameterLimits` after applying the parameter transform.
  void set(
      const CharacterParameters& parameters,
      const CharacterT<T>& referenceCharacter,
      bool updateMesh = true,
      bool updateCollision = true,
      bool applyLimits = true);

  /// Sets the character state to `referenceCharacter`'s bind pose.
  ///
  /// @see set() for the meaning of `updateMesh` and `updateCollision`.
  void setBindPose(
      const CharacterT<T>& referenceCharacter,
      bool updateMesh = true,
      bool updateCollision = true);
};

} // namespace momentum
