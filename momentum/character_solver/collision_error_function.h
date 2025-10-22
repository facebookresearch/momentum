/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/collision_geometry.h>
#include <momentum/character/collision_geometry_state.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>

#include <axel/Bvh.h>

#include <unordered_set>
#include <vector>

namespace momentum {

template <typename T>
class CollisionErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  explicit CollisionErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const CollisionGeometry& cg);

  explicit CollisionErrorFunctionT(const Character& character);

  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) final;

  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<VectorX<T>> gradient) override;

  double getJacobian(
      const ModelParametersT<T>& /*unused*/,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Ref<MatrixX<T>> jacobian,
      Ref<VectorX<T>> residual,
      int& usedRows) override;

  [[nodiscard]] size_t getJacobianSize() const final;

  [[nodiscard]] std::vector<Vector2i> getCollisionPairs() const;

 protected:
  void updateCollisionPairs();

  // Update collisionState_ and bvh_ given the new skeleton state.
  void computeBroadPhase(const SkeletonStateT<T>& state);

  using PairId = std::pair<size_t, size_t>;

  struct PairHash {
    std::size_t operator()(const std::pair<size_t, size_t>& p) const {
      // Use the smaller value as the high-order bits and the larger value as the low-order bits
      return (static_cast<uint64_t>(std::min(p.first, p.second)) << (sizeof(size_t) * 8 / 2)) |
          std::max(p.first, p.second);
    }
  };

  struct PairEqual {
    bool operator()(const PairId& a, const PairId& b) const {
      return (a.first == b.first && a.second == b.second) ||
          (a.first == b.second && a.second == b.first);
    }
  };

  const CollisionGeometry collisionGeometry_;

  std::unordered_set<PairId, PairHash, PairEqual> excludingPairIds_;

  size_t jacobianSize_ = 0;

  CollisionGeometryStateT<T> collisionState_;

  // weights for the error functions
  static constexpr T kCollisionWeight = 5e-3f;

  axel::Bvh<T> bvh_;
};

} // namespace momentum
