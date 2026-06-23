/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/mesh_state.h>
#include <momentum/character/sdf_collision_geometry.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/skin_weights.h>
#include <momentum/character_sequence_solver/fwd.h>
#include <momentum/character_sequence_solver/sequence_error_function.h>
#include <momentum/character_solver/sdf_collision_utility.h>
#include <momentum/character_solver/skinning_weight_iterator.h>
#include <momentum/common/checks.h>
#include <momentum/common/profile.h>
#include <momentum/math/mesh.h>
#include <momentum/math/transform.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace momentum {

/// Continuous (swept) SDF collision error function over two consecutive frames.
///
/// The single-frame `SDFCollisionErrorFunctionT` only sees penetration at the sampled poses, so a
/// vertex moving fast enough to pass entirely through a thin collider (e.g. a finger) between frame
/// t and t+1 is never penalized — it "tunnels". This sequence error function closes that gap: for
/// each participating vertex it sweeps the straight segment connecting the vertex's position at the
/// two frames and penalizes the *deepest* penetration that segment reaches inside a collider.
///
/// The sweep is performed in the collider's local SDF frame: each endpoint is mapped through *its
/// own frame's* world-to-collider transform and the two local positions are interpolated linearly.
/// This captures the relative motion of vertex and collider (both may be joint-driven), keeps the
/// interpolation exactly differentiable (no slerp-derivative approximation), and makes the
/// sphere-trace used to locate crossings exactly conservative because distances are queried in the
/// same local units the march steps in.
///
/// The penalty is `w * phiMax^2` per penetrating sub-interval, where `phiMax` is the greatest
/// penetration depth (`phi = max(0, -sdf)`) reached along the swept segment. It is a *max*, not an
/// integral, so it depends only on how deep the pass-through gets, never on how long the segment
/// stays inside — there is no "stretch the trajectory to dilute the penalty" degeneracy.
///
/// The deepest point `s*` is located exactly (bisecting `d'(s) = 0`, since `d` decreases then
/// increases across a both-crossing interval). Because `s*` is a true interior extremum, the
/// envelope theorem makes the penalty's gradient independent of how `s*` drifts with the
/// parameters: it is simply the single-frame contact gradient evaluated at `s*`, split across the
/// two frames by the interpolation weights `(1 - s*)` (frame t) and `s*` (frame t+1). One residual
/// per interval, gradient and Jacobian both finite-difference-clean.
///
/// Only sub-intervals bounded by two real surface crossings (entry `+ -> -` and exit `- -> +`) are
/// penalized, targeting the pure pass-through case and leaving at-frame penetration to the
/// per-frame error function.
///
/// Derivatives reuse the single-frame chain-rule helpers (`detail_sdf_collision`). The local SDF
/// gradient is mapped to world space including the world-to-collider scale factor, which the
/// derivatives require whenever the collider's joint carries a non-unit scale.
///
/// The `SdfColliderType` is duck-typed; see `SDFCollisionErrorFunctionT` for the required
/// interface.
template <typename T, typename SdfColliderType = SDFColliderT<float>>
class SDFCollisionSequenceErrorFunctionT : public SequenceErrorFunctionT<T> {
 public:
  /// @param character The character containing skeleton, mesh, and skin weights
  /// @param sdfColliders The SDF collision geometry to sweep against
  /// @param participatingVertices Vertex indices to include (empty means all vertices)
  /// @param vertexWeights Per-vertex collision weights (empty means uniform weight of 1.0)
  explicit SDFCollisionSequenceErrorFunctionT(
      const Character& character,
      const std::vector<SdfColliderType>& sdfColliders,
      const std::vector<int>& participatingVertices = {},
      const std::vector<T>& vertexWeights = {});

  [[nodiscard]] size_t numFrames() const final {
    return 2;
  }

  [[nodiscard]] bool needsMesh() const final {
    return true;
  }

  double getError(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates) const final;

  double getGradient(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates,
      Eigen::Ref<Eigen::VectorX<T>> gradient) const final;

  double getJacobian(
      std::span<const ModelParametersT<T>> modelParameters,
      std::span<const SkeletonStateT<T>> skelStates,
      std::span<const MeshStateT<T>> meshStates,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) const final;

  [[nodiscard]] size_t getJacobianSize() const final;

  [[nodiscard]] const Character& getCharacter() const {
    return character_;
  }

  /// Penetration penalty weight, matching the single-frame `SDFCollisionErrorFunctionT`.
  static constexpr T kSDFCollisionWeight = T(5e-3);

  /// Upper bound on the number of penetrating sub-intervals penalized per (vertex, collider).
  static constexpr size_t kMaxIntervals = 2;

 private:
  /// A penetrating sub-interval of the swept segment, bounded by two surface crossings in the
  /// interpolation parameter `s`.
  struct SweptIntervalT {
    T sIn;
    T sOut;
  };

  /// Heap-free, fixed-capacity set of penetrating sub-intervals. Capacity is the compile-time bound
  /// `kMaxIntervals`, so building one never allocates in the per-(vertex, collider) hot loop.
  struct SweptIntervalsT {
    std::array<SweptIntervalT, kMaxIntervals> data{};
    size_t count = 0;
    [[nodiscard]] const SweptIntervalT* begin() const {
      return data.data();
    }
    [[nodiscard]] const SweptIntervalT* end() const {
      return data.data() + count;
    }
  };

  /// Locate the penetrating sub-intervals of the local-space segment `localPos0 + s * delta` via
  /// sphere-tracing, refining each crossing by bisection. Only intervals with both an entry and an
  /// exit crossing are returned (a segment that starts or ends inside the collider contributes only
  /// its fully-crossed sub-intervals).
  [[nodiscard]] SweptIntervalsT findPenetratingIntervals(
      const Eigen::Vector3<T>& localPos0,
      const Eigen::Vector3<T>& delta,
      T segLen,
      const SdfColliderType& collider) const;

  /// Refine a surface crossing bracketed by `[sLo, sHi]` (where the SDF value changes sign) to a
  /// tight estimate of the `phi = 0` point.
  [[nodiscard]] T refineCrossing(
      const Eigen::Vector3<T>& localPos0,
      const Eigen::Vector3<T>& delta,
      const SdfColliderType& collider,
      T sLo,
      T sHi,
      T dLo) const;

  /// Find the deepest point of the swept segment within `interval` by bisecting the along-segment
  /// derivative `d'(s) = gradLocal(s) . delta` (negative while `d` is still decreasing, positive
  /// once it is increasing). Returns the interpolation parameter `s*` of the minimum of `d`.
  [[nodiscard]] T findDeepestPoint(
      const Eigen::Vector3<T>& localPos0,
      const Eigen::Vector3<T>& delta,
      const SweptIntervalT& interval,
      const SdfColliderType& collider) const;

  // Sphere-trace step budget for crossing detection. Generous: the |sdf|-clamped step cannot skip a
  // surface, so this only bounds work, it does not affect accuracy.
  static constexpr int kMaxMarchSteps = 64;
  // Bisection iterations for crossing / deepest-point refinement; ~1e-9 of the segment, well below
  // any FD epsilon.
  static constexpr int kBisectionIters = 30;

  const Character& character_;
  std::vector<SdfColliderType> sdfColliders_;
  std::vector<int> participatingVertices_;
  std::vector<T> vertexWeights_;
};

using SDFCollisionSequenceErrorFunction = SDFCollisionSequenceErrorFunctionT<float>;
using SDFCollisionSequenceErrorFunctiond = SDFCollisionSequenceErrorFunctionT<double>;

// ============================================================================
// Template implementation
// ============================================================================

template <typename T, typename SdfColliderType>
SDFCollisionSequenceErrorFunctionT<T, SdfColliderType>::SDFCollisionSequenceErrorFunctionT(
    const Character& character,
    const std::vector<SdfColliderType>& sdfColliders,
    const std::vector<int>& participatingVertices,
    const std::vector<T>& vertexWeights)
    : SequenceErrorFunctionT<T>(character.skeleton, character.parameterTransform),
      character_(character),
      sdfColliders_(sdfColliders) {
  MT_CHECK_NOTNULL(character_.mesh, "Character must have a mesh for SDF collision detection");
  MT_CHECK_NOTNULL(
      character_.skinWeights, "Character must have skin weights for SDF collision detection");

  if (participatingVertices.empty()) {
    participatingVertices_.resize(character_.mesh->vertices.size());
    for (int i = 0; i < static_cast<int>(character_.mesh->vertices.size()); ++i) {
      participatingVertices_[i] = i;
    }
  } else {
    participatingVertices_ = participatingVertices;
  }

  for (const int vertexIndex : participatingVertices_) {
    MT_CHECK(
        vertexIndex >= 0 && vertexIndex < static_cast<int>(character_.mesh->vertices.size()),
        "Invalid vertex index: {} (mesh has {} vertices)",
        vertexIndex,
        character_.mesh->vertices.size());
  }

  if (vertexWeights.empty()) {
    vertexWeights_.assign(participatingVertices_.size(), T(1));
  } else {
    MT_CHECK(
        vertexWeights.size() == participatingVertices_.size(),
        "Weight count ({}) must match vertex count ({})",
        vertexWeights.size(),
        participatingVertices_.size());
    vertexWeights_ = vertexWeights;
  }
}

template <typename T, typename SdfColliderType>
T SDFCollisionSequenceErrorFunctionT<T, SdfColliderType>::refineCrossing(
    const Eigen::Vector3<T>& localPos0,
    const Eigen::Vector3<T>& delta,
    const SdfColliderType& collider,
    T sLo,
    T sHi,
    T dLo) const {
  T lo = sLo;
  T hi = sHi;
  for (int i = 0; i < kBisectionIters; ++i) {
    const T mid = T(0.5) * (lo + hi);
    const Eigen::Vector3<T> pos = localPos0 + mid * delta;
    const T dMid = collider.evaluate(pos);
    // Keep the half-bracket that still straddles the sign change.
    if ((dMid < T(0)) == (dLo < T(0))) {
      lo = mid;
      dLo = dMid;
    } else {
      hi = mid;
    }
  }
  return T(0.5) * (lo + hi);
}

template <typename T, typename SdfColliderType>
typename SDFCollisionSequenceErrorFunctionT<T, SdfColliderType>::SweptIntervalsT
SDFCollisionSequenceErrorFunctionT<T, SdfColliderType>::findPenetratingIntervals(
    const Eigen::Vector3<T>& localPos0,
    const Eigen::Vector3<T>& delta,
    T segLen,
    const SdfColliderType& collider) const {
  SweptIntervalsT intervals;

  // Floor on the marching step (in s) so a far-outside query still makes progress; the sphere-trace
  // takes larger steps when the surface is distant and shrinks to |sdf| near it, so a surface is
  // never stepped over regardless of this floor.
  const T minStepS = T(1) / static_cast<T>(kMaxMarchSteps);

  T s = T(0);
  T dPrev = collider.evaluate(Eigen::Vector3<T>(localPos0));
  T intervalStart = T(-1); // invalid until an entry crossing is seen
  int steps = 0;
  while (s < T(1) && steps < 2 * kMaxMarchSteps) {
    ++steps;
    const T stepS = std::max(minStepS, std::abs(dPrev) / segLen);
    const T sNext = std::min(T(1), s + stepS);
    const Eigen::Vector3<T> posNext = localPos0 + sNext * delta;
    const T dNext = collider.evaluate(posNext);

    if (dPrev >= T(0) && dNext < T(0)) {
      // Entry crossing: vertex passes from outside to inside.
      intervalStart = refineCrossing(localPos0, delta, collider, s, sNext, dPrev);
    } else if (dPrev < T(0) && dNext >= T(0) && intervalStart >= T(0)) {
      // Exit crossing closing a previously-opened interval.
      const T sOut = refineCrossing(localPos0, delta, collider, s, sNext, dPrev);
      intervals.data[intervals.count++] = {intervalStart, sOut};
      intervalStart = T(-1);
      if (intervals.count >= kMaxIntervals) {
        break;
      }
    }

    s = sNext;
    dPrev = dNext;
  }

  return intervals;
}

template <typename T, typename SdfColliderType>
T SDFCollisionSequenceErrorFunctionT<T, SdfColliderType>::findDeepestPoint(
    const Eigen::Vector3<T>& localPos0,
    const Eigen::Vector3<T>& delta,
    const SweptIntervalT& interval,
    const SdfColliderType& collider) const {
  // d(s) dips negative between the two crossings, so d'(s) = gradLocal(s) . delta runs from
  // negative (still descending) to positive (ascending); bisect for the d' = 0 minimum.
  T lo = interval.sIn;
  T hi = interval.sOut;
  for (int i = 0; i < kBisectionIters; ++i) {
    const T mid = T(0.5) * (lo + hi);
    const Eigen::Vector3<T> pos = localPos0 + mid * delta;
    const auto [d, g] = collider.evaluateWithGradient(pos);
    if (g.dot(delta) < T(0)) {
      lo = mid; // d still decreasing: minimum is to the right
    } else {
      hi = mid; // d increasing: minimum is to the left
    }
  }
  return T(0.5) * (lo + hi);
}

template <typename T, typename SdfColliderType>
double SDFCollisionSequenceErrorFunctionT<T, SdfColliderType>::getError(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> meshStates) const {
  MT_PROFILE_FUNCTION();
  MT_CHECK(skelStates.size() == 2);
  MT_CHECK(meshStates.size() == 2);
  MT_CHECK_NOTNULL(meshStates[0].posedMesh_);
  MT_CHECK_NOTNULL(meshStates[1].posedMesh_);

  double totalError = 0.0;

  for (const auto& collider : sdfColliders_) {
    if (!collider.isValid()) {
      continue;
    }
    const auto worldToCollider0 =
        detail_sdf_collision::computeWorldToColliderTransform(collider, skelStates[0]);
    const auto worldToCollider1 =
        detail_sdf_collision::computeWorldToColliderTransform(collider, skelStates[1]);

    for (size_t i = 0; i < participatingVertices_.size(); ++i) {
      const int vertexIndex = participatingVertices_[i];
      const T vertexWeight = vertexWeights_[i];
      const Eigen::Vector3<T> localPos0 =
          worldToCollider0 * meshStates[0].posedMesh_->vertices[vertexIndex];
      const Eigen::Vector3<T> localPos1 =
          worldToCollider1 * meshStates[1].posedMesh_->vertices[vertexIndex];
      const Eigen::Vector3<T> delta = localPos1 - localPos0;
      const T segLen = delta.norm();
      if (segLen < std::numeric_limits<T>::epsilon()) {
        continue; // no motion: static penetration is the per-frame function's job
      }

      const auto intervals = findPenetratingIntervals(localPos0, delta, segLen, collider);
      for (const auto& interval : intervals) {
        const T sStar = findDeepestPoint(localPos0, delta, interval, collider);
        const Eigen::Vector3<T> pos = localPos0 + sStar * delta;
        const T d = collider.evaluate(pos);
        if (d >= T(0)) {
          continue;
        }
        const T phi = -d;
        totalError +=
            static_cast<double>(vertexWeight * phi * phi * kSDFCollisionWeight * this->weight_);
      }
    }
  }

  return totalError;
}

template <typename T, typename SdfColliderType>
double SDFCollisionSequenceErrorFunctionT<T, SdfColliderType>::getGradient(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> meshStates,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_PROFILE_FUNCTION();
  MT_CHECK(skelStates.size() == 2);
  MT_CHECK(meshStates.size() == 2);
  MT_CHECK_NOTNULL(meshStates[0].posedMesh_);
  MT_CHECK_NOTNULL(meshStates[0].restMesh_);
  MT_CHECK_NOTNULL(meshStates[1].posedMesh_);
  MT_CHECK_NOTNULL(meshStates[1].restMesh_);

  const Eigen::Index np = this->parameterTransform_.numAllModelParameters();
  double totalError = 0.0;

  for (const auto& collider : sdfColliders_) {
    if (!collider.isValid()) {
      continue;
    }
    const std::array<TransformT<T>, 2> worldToCollider = {
        detail_sdf_collision::computeWorldToColliderTransform(collider, skelStates[0]),
        detail_sdf_collision::computeWorldToColliderTransform(collider, skelStates[1])};
    const std::array<TransformT<T>, 2> colliderToWorld = {
        worldToCollider[0].inverse(), worldToCollider[1].inverse()};

    for (size_t i = 0; i < participatingVertices_.size(); ++i) {
      const int vertexIndex = participatingVertices_[i];
      const T vertexWeight = vertexWeights_[i];
      const std::array<Eigen::Vector3<T>, 2> worldVertex = {
          meshStates[0].posedMesh_->vertices[vertexIndex],
          meshStates[1].posedMesh_->vertices[vertexIndex]};
      const Eigen::Vector3<T> localPos0 = worldToCollider[0] * worldVertex[0];
      const Eigen::Vector3<T> localPos1 = worldToCollider[1] * worldVertex[1];
      const Eigen::Vector3<T> delta = localPos1 - localPos0;
      const T segLen = delta.norm();
      if (segLen < std::numeric_limits<T>::epsilon()) {
        continue;
      }

      const size_t commonAncestor = detail_sdf_collision::findCommonAncestorForVertex(
          *character_.skinWeights, this->skeleton_, vertexIndex, collider.parentJoint());

      const auto intervals = findPenetratingIntervals(localPos0, delta, segLen, collider);
      for (const auto& interval : intervals) {
        const T sStar = findDeepestPoint(localPos0, delta, interval, collider);
        const Eigen::Vector3<T> pos = localPos0 + sStar * delta;
        const auto [d, localGradient] = collider.evaluateWithGradient(pos);
        if (d >= T(0)) {
          continue;
        }
        const T phi = -d;
        totalError +=
            static_cast<double>(vertexWeight * phi * phi * kSDFCollisionWeight * this->weight_);

        // d(w*phi^2)/dp = 2*w*phi * d(phi)/dp; phi = -d(s*) with d'(s*) = 0, so the deepest point
        // s* drifting with the parameters contributes nothing (envelope theorem).
        const T wgt = T(2) * vertexWeight * kSDFCollisionWeight * this->weight_ * phi;

        for (size_t f = 0; f < 2; ++f) {
          const T alpha = (f == 0) ? (T(1) - sStar) : sStar;
          const Eigen::Vector3<T> sdfGradient =
              worldToCollider[f].scale * (colliderToWorld[f].rotation * localGradient);
          Eigen::Ref<Eigen::VectorX<T>> gradBlock = gradient.segment(f * np, np);

          SkinningWeightIteratorT<T> skinningIter(
              character_, *meshStates[f].restMesh_, skelStates[f], vertexIndex);
          while (!skinningIter.finished()) {
            const auto [jointIndex, boneWeight, transformedVertex] = skinningIter.next();
            if (jointIndex == commonAncestor) {
              break;
            }
            if (std::abs(boneWeight) < std::numeric_limits<T>::epsilon()) {
              continue;
            }
            const auto& jointState = skelStates[f].jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;
            const Eigen::Vector3<T> posd = transformedVertex - jointState.translation();
            detail_sdf_collision::accumulateJointGradient<T>(
                jointState,
                paramIndex,
                posd,
                -wgt * alpha * boneWeight * sdfGradient,
                this->activeJointParams_,
                this->parameterTransform_,
                gradBlock);
          }

          if (collider.parentJoint() != kInvalidIndex) {
            detail_sdf_collision::accumulateColliderHierarchyGradient<T>(
                skelStates[f],
                collider.parentJoint(),
                commonAncestor,
                worldVertex[f],
                wgt * alpha * sdfGradient,
                this->skeleton_,
                this->activeJointParams_,
                this->parameterTransform_,
                gradBlock);
          }
        }
      }
    }
  }

  return totalError;
}

template <typename T, typename SdfColliderType>
double SDFCollisionSequenceErrorFunctionT<T, SdfColliderType>::getJacobian(
    std::span<const ModelParametersT<T>> /* modelParameters */,
    std::span<const SkeletonStateT<T>> skelStates,
    std::span<const MeshStateT<T>> meshStates,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) const {
  MT_PROFILE_FUNCTION();
  MT_CHECK(skelStates.size() == 2);
  MT_CHECK(meshStates.size() == 2);
  MT_CHECK_NOTNULL(meshStates[0].posedMesh_);
  MT_CHECK_NOTNULL(meshStates[0].restMesh_);
  MT_CHECK_NOTNULL(meshStates[1].posedMesh_);
  MT_CHECK_NOTNULL(meshStates[1].restMesh_);

  const Eigen::Index np = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(jacobian.cols() == 2 * np, "Jacobian column count mismatch");

  double totalError = 0.0;
  int currentRow = 0;

  for (const auto& collider : sdfColliders_) {
    if (!collider.isValid()) {
      continue;
    }
    const std::array<TransformT<T>, 2> worldToCollider = {
        detail_sdf_collision::computeWorldToColliderTransform(collider, skelStates[0]),
        detail_sdf_collision::computeWorldToColliderTransform(collider, skelStates[1])};
    const std::array<TransformT<T>, 2> colliderToWorld = {
        worldToCollider[0].inverse(), worldToCollider[1].inverse()};

    for (size_t i = 0; i < participatingVertices_.size(); ++i) {
      const int vertexIndex = participatingVertices_[i];
      const T vertexWeight = vertexWeights_[i];
      const std::array<Eigen::Vector3<T>, 2> worldVertex = {
          meshStates[0].posedMesh_->vertices[vertexIndex],
          meshStates[1].posedMesh_->vertices[vertexIndex]};
      const Eigen::Vector3<T> localPos0 = worldToCollider[0] * worldVertex[0];
      const Eigen::Vector3<T> localPos1 = worldToCollider[1] * worldVertex[1];
      const Eigen::Vector3<T> delta = localPos1 - localPos0;
      const T segLen = delta.norm();
      if (segLen < std::numeric_limits<T>::epsilon()) {
        continue;
      }

      const size_t commonAncestor = detail_sdf_collision::findCommonAncestorForVertex(
          *character_.skinWeights, this->skeleton_, vertexIndex, collider.parentJoint());

      const auto intervals = findPenetratingIntervals(localPos0, delta, segLen, collider);
      for (const auto& interval : intervals) {
        const T sStar = findDeepestPoint(localPos0, delta, interval, collider);
        const Eigen::Vector3<T> pos = localPos0 + sStar * delta;
        const auto [d, localGradient] = collider.evaluateWithGradient(pos);
        if (d >= T(0)) {
          continue;
        }
        const T phi = -d;
        MT_CHECK(
            currentRow < jacobian.rows() && currentRow < residual.rows(),
            "Insufficient Jacobian/residual rows");

        const T wgt = std::sqrt(vertexWeight * kSDFCollisionWeight * this->weight_);
        residual[currentRow] = wgt * phi;
        totalError +=
            static_cast<double>(vertexWeight * phi * phi * kSDFCollisionWeight * this->weight_);
        jacobian.row(currentRow).setZero();

        for (size_t f = 0; f < 2; ++f) {
          const T alpha = (f == 0) ? (T(1) - sStar) : sStar;
          const Eigen::Vector3<T> sdfGradient =
              worldToCollider[f].scale * (colliderToWorld[f].rotation * localGradient);

          SkinningWeightIteratorT<T> skinningIter(
              character_, *meshStates[f].restMesh_, skelStates[f], vertexIndex);
          while (!skinningIter.finished()) {
            const auto [jointIndex, boneWeight, transformedVertex] = skinningIter.next();
            if (jointIndex == commonAncestor) {
              break;
            }
            if (std::abs(boneWeight) < std::numeric_limits<T>::epsilon()) {
              continue;
            }
            const auto& jointState = skelStates[f].jointState[jointIndex];
            const size_t paramIndex = jointIndex * kParametersPerJoint;
            const Eigen::Vector3<T> posd = transformedVertex - jointState.translation();
            detail_sdf_collision::accumulateJointJacobian<T>(
                jointState,
                paramIndex,
                posd,
                -wgt * alpha * boneWeight * sdfGradient,
                this->activeJointParams_,
                this->parameterTransform_,
                jacobian.block(currentRow, f * np, 1, np));
          }

          if (collider.parentJoint() != kInvalidIndex) {
            detail_sdf_collision::accumulateColliderHierarchyJacobian<T>(
                skelStates[f],
                collider.parentJoint(),
                commonAncestor,
                worldVertex[f],
                wgt * alpha * sdfGradient,
                this->skeleton_,
                this->activeJointParams_,
                this->parameterTransform_,
                jacobian.block(currentRow, f * np, 1, np));
          }
        }
        ++currentRow;
      }
    }
  }

  usedRows = currentRow;
  return totalError;
}

template <typename T, typename SdfColliderType>
size_t SDFCollisionSequenceErrorFunctionT<T, SdfColliderType>::getJacobianSize() const {
  return kMaxIntervals * participatingVertices_.size() * sdfColliders_.size();
}

} // namespace momentum
