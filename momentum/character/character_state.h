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
  /// Model parameters and joint offsets
  CharacterParameters parameters;

  /// Joint transformations
  SkeletonState skeletonState;

  /// Locator positions and orientations
  LocatorState locatorState;

  /// Skinned mesh (may be null if updateMesh=false)
  Mesh_u meshState;

  /// Collision geometry state (may be null if updateCollision=false)
  CollisionGeometryState_u collisionState;

  /// Creates an empty character state
  CharacterStateT();

  /// Creates a deep copy of another character state
  explicit CharacterStateT(const CharacterStateT& other);

  /// Move constructor
  CharacterStateT(CharacterStateT&& c) noexcept;

  /// Copy assignment is disabled
  CharacterStateT& operator=(const CharacterStateT& rhs) = delete;

  /// Move assignment operator
  CharacterStateT& operator=(CharacterStateT&& rhs) noexcept;

  /// Destructor
  ~CharacterStateT();

  /// Creates a character state in bind pose
  ///
  /// @param referenceCharacter The character to use as reference
  /// @param updateMesh Whether to compute the skinned mesh
  /// @param updateCollision Whether to update collision geometry
  explicit CharacterStateT(
      const CharacterT<T>& referenceCharacter,
      bool updateMesh = true,
      bool updateCollision = true);

  /// Creates a character state with specific parameters
  ///
  /// @param parameters The character parameters to use
  /// @param referenceCharacter The character to use as reference
  /// @param updateMesh Whether to compute the skinned mesh
  /// @param updateCollision Whether to update collision geometry
  /// @param applyLimits Whether to apply joint parameter limits
  CharacterStateT(
      const CharacterParameters& parameters,
      const CharacterT<T>& referenceCharacter,
      bool updateMesh = true,
      bool updateCollision = true,
      bool applyLimits = true);

  /// Updates the character state with specific parameters
  ///
  /// If parameters.offsets is empty, it will be initialized with zeros.
  ///
  /// @param parameters The character parameters to use
  /// @param referenceCharacter The character to use as reference
  /// @param updateMesh Whether to compute the skinned mesh
  /// @param updateCollision Whether to update collision geometry
  /// @param applyLimits Whether to apply joint parameter limits
  void set(
      const CharacterParameters& parameters,
      const CharacterT<T>& referenceCharacter,
      bool updateMesh = true,
      bool updateCollision = true,
      bool applyLimits = true);

  /// Sets the character state to the bind pose
  ///
  /// @param referenceCharacter The character to use as reference
  /// @param updateMesh Whether to compute the skinned mesh
  /// @param updateCollision Whether to update collision geometry
  void setBindPose(
      const CharacterT<T>& referenceCharacter,
      bool updateMesh = true,
      bool updateCollision = true);
};

} // namespace momentum
