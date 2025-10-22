/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/types.h>
#include <momentum/math/mesh.h>
#include <momentum/math/types.h>

namespace momentum {

/// Represents the complete state of a model's mesh
///
/// Stores the mesh in its various states throughout the deformation pipeline:
/// - neutralMesh: Base mesh before any deformations
/// - restMesh: Mesh after blend shapes/expressions are applied
/// - posedMesh: Final mesh after skinning is applied
template <typename T>
struct MeshStateT {
  std::unique_ptr<MeshT<T>>
      neutralMesh_; // Rest mesh without facial expression basis,
                    // used to restore the neutral shape after facial expressions are applied.
                    // Not used with there is a shape basis.
  std::unique_ptr<MeshT<T>> restMesh_; // The rest positions of the mesh after shape basis
                                       // (and potentially facial expression) has been applied
  std::unique_ptr<MeshT<T>>
      posedMesh_; // The posed mesh after the skeleton transforms have been applied.

  /// Creates an empty mesh state
  MeshStateT() noexcept = default;
  MeshStateT(
      const ModelParametersT<T>& parameters,
      const SkeletonStateT<T>& state,
      const Character& character) noexcept;
  ~MeshStateT() noexcept;

  /// Copy constructor
  MeshStateT(const MeshStateT& other);

  /// Copy assignment operator
  MeshStateT& operator=(const MeshStateT& other);

  /// Move constructor
  MeshStateT(MeshStateT&& other) noexcept = default;

  /// Move assignment operator
  MeshStateT& operator=(MeshStateT&& other) noexcept = default;

  /// Updates the mesh state based on model parameters and skeleton state
  ///
  /// This applies the full deformation pipeline:
  /// 1. Apply blend shapes to create rest mesh
  /// 2. Apply face expression blend shapes
  /// 3. Update normals
  /// 4. Apply skinning to create posed mesh
  ///
  /// @param parameters Model parameters containing blend weights
  /// @param state Current skeleton state for skinning
  /// @param character Character containing mesh, blend shapes, and skinning data
  void update(
      const ModelParametersT<T>& parameters,
      const SkeletonStateT<T>& state,
      const Character& character);
};

} // namespace momentum
