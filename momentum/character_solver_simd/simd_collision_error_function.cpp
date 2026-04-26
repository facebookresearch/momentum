/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver_simd/simd_collision_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"
#include "momentum/simd/simd.h"

namespace momentum {

namespace {

template <typename T>
[[nodiscard]] drjit::Array<T, 2> toDrJitVec(const Eigen::Vector2<T>& v) {
  return {v.x(), v.y()};
}

template <typename T>
[[nodiscard]] drjit::Array<T, 3> toDrJitVec(const Eigen::Vector3<T>& v) {
  return {v.x(), v.y(), v.z()};
}

template <typename T>
[[nodiscard]] Eigen::Vector3<T> extractSingleElement(const Vector3P<T>& vec, int index) {
  return {vec.x()[index], vec.y()[index], vec.z()[index]};
}

// It is much friendlier to SIMD than the version currently in the momentum code base
// because it includes fewer branches.
//
// Long term we should probably merge this with the other implementation, if
// properly templated we can probably even use the same code for both.
template <typename VecType1, typename VecType2>
[[nodiscard]] std::pair<Packet<typename VecType1::Scalar>, Packet<typename VecType1::Scalar>>
closestPointOnTwoSegments(
    const VecType1& p0,
    const VecType1& d0,
    const VecType2& p1,
    const VecType2& d1) {
  static_assert(std::is_same_v<typename VecType1::Scalar, typename VecType2::Scalar>);

  using T = typename VecType1::Scalar;

  const Vector3P<T> p = p1 - p0;
  const Packet<T> pd0 = drjit::dot(p, d0);
  const Packet<T> pd1 = drjit::dot(p, d1);

  const Packet<T> d00 = drjit::squared_norm(d0);
  const Packet<T> d11 = drjit::squared_norm(d1);
  const Packet<T> d01 = drjit::dot(d0, d1);
  const Packet<T> div = d00 * d11 - d01 * d01;

  auto t0 = drjit::zeros<Packet<T>>();
  auto t1 = drjit::zeros<Packet<T>>();

  // If the segments are nearly parallel, the initial assignment
  // to t0 may be garbage. The following assignments are all stable,
  // however, so after a ping-pong-ping finding the closest point
  // on the other line to the current point, we are guaranteed to
  // have stable output points.
  //
  // Note that the actual returned values are inherently unstable
  // if the lines are near parallel, but the distance from
  // (p0 + t0*d0) to (p1 + t1*d1) will be stable.
  t0 = drjit::clip(drjit::select(div > 0.0f, (pd0 * d11 - pd1 * d01) / div, t0), 0.0f, 1.0f);
  t1 = drjit::clip(drjit::select(d11 > 0.0f, (t0 * d01 - pd1) / d11, t1), 0.0f, 1.0f);
  t0 = drjit::clip(drjit::select(d00 > 0.0f, (t1 * d01 + pd0) / d00, t0), 0.0f, 1.0f);

  return {t0, t1};
}

} // namespace

template <typename T>
SimdCollisionErrorFunctionT<T>::SimdCollisionErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    const CollisionGeometry& collisionGeometry)
    : SkeletonErrorFunctionT<T>(skel, pt), collisionGeometry_(collisionGeometry) {
  updateCollisionPairs();
}

template <typename T>
SimdCollisionErrorFunctionT<T>::SimdCollisionErrorFunctionT(const Character& character)
    : SimdCollisionErrorFunctionT(
          character.skeleton,
          character.parameterTransform,
          character.collision
              ? *character.collision
              : throw std::invalid_argument(
                    "Attempting to create collision error function with a character that has no collision geometries")) {
  // Do nothing
}

template <typename T>
void SimdCollisionErrorFunctionT<T>::updateCollisionPairs() {
  validPairs_.clear();

  const auto n = collisionGeometry_.size();

  const SkeletonStateT<T> state(
      this->parameterTransform_.zero().template cast<T>(), this->skeleton_);
  collisionState_.update(state, collisionGeometry_);

  aabbs_.resize(n);
  for (size_t i = 0; i < n; ++i) {
    aabbs_[i].id = gsl::narrow_cast<axel::Index>(i);
    updateAabb(
        aabbs_[i],
        collisionState_.origin[i],
        collisionState_.direction[i],
        collisionState_.radius[i]);
  }

  if (n == 0) {
    return;
  }

  commonAncestors_.assign(n * n, kInvalidIndex);
  collisionPairs_.resize(n);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (isValidCollisionPair(collisionState_, collisionGeometry_, this->skeleton_, i, j)) {
        const size_t lca = this->skeleton_.commonAncestor(
            collisionGeometry_[i].parent, collisionGeometry_[j].parent);
        validPairs_.push_back({i, j, lca});
        commonAncestors_[i * n + j] = lca;
        commonAncestors_[j * n + i] = lca;
      }
    }
  }
}

template <typename T>
void SimdCollisionErrorFunctionT<T>::computeBroadPhase(const SkeletonStateT<T>& state) {
  {
    MT_PROFILE_EVENT("Collision: updateState");
    collisionState_.update(state, collisionGeometry_);
  }

  for (size_t i = 0; i < aabbs_.size(); ++i) {
    updateAabb(
        aabbs_[i],
        collisionState_.origin[i],
        collisionState_.direction[i],
        collisionState_.radius[i]);
  }

  for (auto& pairs : collisionPairs_) {
    pairs.clear();
  }
  for (const auto& pair : validPairs_) {
    if (aabbs_[pair.indexA].intersects(aabbs_[pair.indexB])) {
      collisionPairs_[pair.indexA].push_back(static_cast<int>(pair.indexB));
    }
  }
}

template <typename T>
double SimdCollisionErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /*unused*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */) {
  if (state.jointState.empty()) {
    return 0.0;
  }

  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  computeBroadPhase(state);

  auto error = drjit::zeros<DoubleP>();

  for (size_t iCol = 0; iCol < collisionPairs_.size(); ++iCol) {
    const auto& candidatesCur = collisionPairs_[iCol];

    const Vector3P<T> p_i = toDrJitVec(collisionState_.origin[iCol]);
    const Vector3P<T> d_i = toDrJitVec(collisionState_.direction[iCol]);
    const Vector2P<T> radius_i = toDrJitVec(collisionState_.radius[iCol]);

    for (auto [candidateIndices, mask] : drjit::range<IntP>(candidatesCur.size())) {
      const IntP jCol = drjit::gather<IntP>(candidatesCur.data(), candidateIndices, mask);
      const auto p_j = drjit::gather<Vector3P<T>>(collisionState_.origin.data(), jCol, mask);
      const auto d_j = drjit::gather<Vector3P<T>>(collisionState_.direction.data(), jCol, mask);
      const auto radius_j = drjit::gather<Vector2P<T>>(collisionState_.radius.data(), jCol, mask);

      const auto [t_i, t_j] = closestPointOnTwoSegments(p_i, d_i, p_j, d_j);

      const Vector3P<T> closestPoint_i = p_i + t_i * d_i;
      const Vector3P<T> closestPoint_j = p_j + t_j * d_j;
      const Packet<T> distance = drjit::norm(closestPoint_i - closestPoint_j);

      const Packet<T> radius = (radius_i.x() * (1.0f - t_i) + radius_i.y() * t_i) +
          (radius_j.x() * (1.0f - t_j) + radius_j.y() * t_j);

      const auto validMask =
          drjit::PacketMask<int, kSimdPacketSize>(mask) && distance >= Eps<T>(1e-8f, 1e-17);
      drjit::masked(error, validMask) += drjit::square(drjit::maximum(radius - distance, 0));
    }
  }

  return drjit::sum(error) * kCollisionWeight * this->weight_;
}

template <typename T>
double SimdCollisionErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /*unused*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<VectorX<T>> gradient) {
  if (state.jointState.empty()) {
    return 0.0;
  }

  computeBroadPhase(state);

  const auto numCapsules = collisionGeometry_.size();
  auto error = drjit::zeros<DoubleP>();

  for (size_t iCol = 0; iCol < collisionPairs_.size(); ++iCol) {
    const auto& candidatesCur = collisionPairs_[iCol];

    Vector3P<T> p_i = toDrJitVec(collisionState_.origin[iCol]);
    Vector3P<T> d_i = toDrJitVec(collisionState_.direction[iCol]);
    Vector2P<T> radius_i = toDrJitVec(collisionState_.radius[iCol]);

    for (auto [candidateIndices, mask] : drjit::range<IntP>(candidatesCur.size())) {
      const IntP jCol = drjit::gather<IntP>(candidatesCur.data(), candidateIndices, mask);
      const auto p_j = drjit::gather<Vector3P<T>>(collisionState_.origin.data(), jCol, mask);
      const auto d_j = drjit::gather<Vector3P<T>>(collisionState_.direction.data(), jCol, mask);
      const auto radius_j = drjit::gather<Vector2P<T>>(collisionState_.radius.data(), jCol, mask);

      const auto [t_i, t_j] = closestPointOnTwoSegments(p_i, d_i, p_j, d_j);

      const Vector3P<T> position_i = p_i + t_i * d_i;
      const Vector3P<T> position_j = p_j + t_j * d_j;
      const Vector3P<T> direction = position_i - position_j;
      const Packet<T> distance = drjit::norm(direction);

      const Packet<T> radius = (radius_i.x() * (1.0f - t_i) + radius_i.y() * t_i) +
          (radius_j.x() * (1.0f - t_j) + radius_j.y() * t_j);
      const Packet<T> overlap = radius - distance;

      const auto finalMask = drjit::PacketMask<int, kSimdPacketSize>(mask) &&
          distance >= Eps<T>(1e-8f, 1e-17) && distance < radius;

      if (!drjit::any(finalMask)) {
        continue;
      }

      const Packet<T> overlapFraction = overlap / distance;
      const Packet<T> wgt = -2.0f * kCollisionWeight * this->weight_ * overlapFraction;

      drjit::masked(error, finalMask) +=
          kCollisionWeight * this->weight_ * drjit::square(radius - distance);

      const size_t iJoint = collisionGeometry_[iCol].parent;
      for (uint32_t k = 0; k < kSimdPacketSize; ++k) {
        if (!finalMask[k]) {
          continue;
        }

        const size_t jJoint = collisionGeometry_[jCol[k]].parent;
        const auto commonAncestor = commonAncestors_[iCol * numCapsules + jCol[k]];

        const Eigen::Vector3<T> direction_k = extractSingleElement(direction, k);
        const Eigen::Vector3<T> position_ik = extractSingleElement(position_i, k);
        const Eigen::Vector3<T> position_jk = extractSingleElement(position_j, k);
        const T wgt_k = wgt[k];
        const T distance_k = distance[k];

        // Effective radii at closest points (scalar extraction)
        const T radiusA_at_cp_k =
            collisionState_.radius[iCol][0] + t_i[k] * collisionState_.delta[iCol];
        const T radiusB_at_cp_k =
            collisionState_.radius[jCol[k]][0] + t_j[k] * collisionState_.delta[jCol[k]];
        const T scaleCorr_A = -distance_k * radiusA_at_cp_k * ln2<T>();
        const T scaleCorr_B = distance_k * radiusB_at_cp_k * ln2<T>();

        // -----------------------------------
        //  process first joint
        // -----------------------------------
        size_t jointIndex = iJoint;
        while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Vector3<T> posd = position_ik - jointState.translation();

          for (size_t d = 0; d < 3; d++) {
            if (this->activeJointParams_[paramIndex + d]) {
              const T val = direction_k.dot(jointState.getTranslationDerivative(d)) * wgt_k;
              gradient_jointParams_to_modelParams(
                  val, paramIndex + d, this->parameterTransform_, gradient);
            }
            if (this->activeJointParams_[paramIndex + 3 + d]) {
              const T val = direction_k.dot(jointState.getRotationDerivative(d, posd)) * wgt_k;
              gradient_jointParams_to_modelParams(
                  val, paramIndex + 3 + d, this->parameterTransform_, gradient);
            }
          }
          if (this->activeJointParams_[paramIndex + 6]) {
            const T val =
                (direction_k.dot(jointState.getScaleDerivative(posd)) + scaleCorr_A) * wgt_k;
            gradient_jointParams_to_modelParams(
                val, paramIndex + 6, this->parameterTransform_, gradient);
          }

          jointIndex = this->skeleton_.joints[jointIndex].parent;
        }

        // -----------------------------------
        //  process second joint
        // -----------------------------------
        jointIndex = jJoint;
        while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Vector3<T> posd = position_jk - jointState.translation();

          for (size_t d = 0; d < 3; d++) {
            if (this->activeJointParams_[paramIndex + d]) {
              const T val = direction_k.dot(jointState.getTranslationDerivative(d)) * -wgt_k;
              gradient_jointParams_to_modelParams(
                  val, paramIndex + d, this->parameterTransform_, gradient);
            }
            if (this->activeJointParams_[paramIndex + 3 + d]) {
              const T val = direction_k.dot(jointState.getRotationDerivative(d, posd)) * -wgt_k;
              gradient_jointParams_to_modelParams(
                  val, paramIndex + 3 + d, this->parameterTransform_, gradient);
            }
          }
          if (this->activeJointParams_[paramIndex + 6]) {
            const T val =
                (direction_k.dot(jointState.getScaleDerivative(posd)) + scaleCorr_B) * -wgt_k;
            gradient_jointParams_to_modelParams(
                val, paramIndex + 6, this->parameterTransform_, gradient);
          }

          jointIndex = this->skeleton_.joints[jointIndex].parent;
        }

        // -----------------------------------
        //  process scale derivatives above common ancestor
        // -----------------------------------
        {
          const T netScaleVal =
              (direction_k.squaredNorm() - distance_k * (radiusA_at_cp_k + radiusB_at_cp_k)) *
              ln2<T>() * wgt_k;
          size_t ancestorIndex = commonAncestor;
          while (ancestorIndex != kInvalidIndex) {
            const size_t paramIndex = ancestorIndex * kParametersPerJoint;
            if (this->activeJointParams_[paramIndex + 6]) {
              gradient_jointParams_to_modelParams(
                  netScaleVal, paramIndex + 6, this->parameterTransform_, gradient);
            }
            ancestorIndex = this->skeleton_.joints[ancestorIndex].parent;
          }
        }
      }
    }
  }

  MT_CHECK(
      state.jointParameters.size() ==
      gsl::narrow<Eigen::Index>(this->skeleton_.joints.size() * kParametersPerJoint));

  return drjit::sum(error);
}

template <typename T>
double SimdCollisionErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /*unused*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /* meshState */,
    Ref<MatrixX<T>> jacobian,
    Ref<VectorX<T>> residual,
    int& usedRows) {
  computeBroadPhase(state);

  const auto numCapsules = collisionGeometry_.size();
  const T wgt = std::sqrt(kCollisionWeight * this->weight_);

  auto error = drjit::zeros<DoubleP>();

  int row = 0;
  for (size_t iCol = 0; iCol < collisionPairs_.size(); ++iCol) {
    const auto& candidatesCur = collisionPairs_[iCol];

    Vector3P<T> p_i = toDrJitVec(collisionState_.origin[iCol]);
    Vector3P<T> d_i = toDrJitVec(collisionState_.direction[iCol]);
    Vector2P<T> radius_i = toDrJitVec(collisionState_.radius[iCol]);

    for (auto [candidateIndices, mask] : drjit::range<IntP>(candidatesCur.size())) {
      const IntP jCol = drjit::gather<IntP>(candidatesCur.data(), candidateIndices, mask);
      const auto p_j = drjit::gather<Vector3P<T>>(collisionState_.origin.data(), jCol, mask);
      const auto d_j = drjit::gather<Vector3P<T>>(collisionState_.direction.data(), jCol, mask);
      const auto radius_j = drjit::gather<Vector2P<T>>(collisionState_.radius.data(), jCol, mask);

      const auto [t_i, t_j] = closestPointOnTwoSegments(p_i, d_i, p_j, d_j);

      const Vector3P<T> position_i = p_i + t_i * d_i;
      const Vector3P<T> position_j = p_j + t_j * d_j;
      const Vector3P<T> direction = position_i - position_j;
      const Packet<T> distance = drjit::norm(direction);

      const Packet<T> radius = (radius_i.x() * (1.0f - t_i) + radius_i.y() * t_i) +
          (radius_j.x() * (1.0f - t_j) + radius_j.y() * t_j);
      const Packet<T> overlap = radius - distance;

      const auto finalMask = drjit::PacketMask<int, kSimdPacketSize>(mask) &&
          distance >= Eps<T>(1e-8f, 1e-17) && distance < radius;

      if (!drjit::any(finalMask)) {
        continue;
      }

      drjit::masked(error, finalMask) +=
          kCollisionWeight * this->weight_ * drjit::square(radius - distance);

      const Packet<T> inverseDistance = 1.0 / distance;
      const Packet<T> fac = inverseDistance * wgt;

      const size_t iJoint = collisionGeometry_[iCol].parent;
      for (uint32_t k = 0; k < kSimdPacketSize; ++k) {
        if (!finalMask[k]) {
          continue;
        }

        const size_t jJoint = collisionGeometry_[jCol[k]].parent;
        const auto commonAncestor = commonAncestors_[iCol * numCapsules + jCol[k]];

        const Eigen::Vector3<T> direction_k = extractSingleElement(direction, k);
        const T fac_k = fac[k];
        const T distance_k = distance[k];

        // Effective radii at closest points (scalar extraction)
        const T radiusA_at_cp_k =
            collisionState_.radius[iCol][0] + t_i[k] * collisionState_.delta[iCol];
        const T radiusB_at_cp_k =
            collisionState_.radius[jCol[k]][0] + t_j[k] * collisionState_.delta[jCol[k]];
        const T scaleCorr_A = -distance_k * radiusA_at_cp_k * ln2<T>();
        const T scaleCorr_B = distance_k * radiusB_at_cp_k * ln2<T>();

        // -----------------------------------
        //  process first joint
        // -----------------------------------
        size_t jointIndex = iJoint;
        while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Eigen::Vector3<T> posd =
              extractSingleElement(position_i, k) - jointState.translation();

          for (size_t d = 0; d < 3; d++) {
            if (this->activeJointParams_[paramIndex + d]) {
              const T val = direction_k.dot(jointState.getTranslationDerivative(d)) * -fac_k;
              jacobian_jointParams_to_modelParams(
                  val, paramIndex + d, row, this->parameterTransform_, jacobian);
            }
            if (this->activeJointParams_[paramIndex + 3 + d]) {
              const T val = direction_k.dot(jointState.getRotationDerivative(d, posd)) * -fac_k;
              jacobian_jointParams_to_modelParams(
                  val, paramIndex + 3 + d, row, this->parameterTransform_, jacobian);
            }
          }
          if (this->activeJointParams_[paramIndex + 6]) {
            const T val =
                (direction_k.dot(jointState.getScaleDerivative(posd)) + scaleCorr_A) * -fac_k;
            jacobian_jointParams_to_modelParams(
                val, paramIndex + 6, row, this->parameterTransform_, jacobian);
          }

          jointIndex = this->skeleton_.joints[jointIndex].parent;
        }

        // -----------------------------------
        //  process second joint
        // -----------------------------------
        jointIndex = jJoint;
        while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
          const auto& jointState = state.jointState[jointIndex];
          const size_t paramIndex = jointIndex * kParametersPerJoint;
          const Vector3<T> posd = extractSingleElement(position_j, k) - jointState.translation();

          for (size_t d = 0; d < 3; d++) {
            if (this->activeJointParams_[paramIndex + d]) {
              const T val = direction_k.dot(jointState.getTranslationDerivative(d)) * fac_k;
              jacobian_jointParams_to_modelParams(
                  val, paramIndex + d, row, this->parameterTransform_, jacobian);
            }
            if (this->activeJointParams_[paramIndex + 3 + d]) {
              const T val = direction_k.dot(jointState.getRotationDerivative(d, posd)) * fac_k;
              jacobian_jointParams_to_modelParams(
                  val, paramIndex + 3 + d, row, this->parameterTransform_, jacobian);
            }
          }
          if (this->activeJointParams_[paramIndex + 6]) {
            const T val =
                (direction_k.dot(jointState.getScaleDerivative(posd)) + scaleCorr_B) * fac_k;
            jacobian_jointParams_to_modelParams(
                val, paramIndex + 6, row, this->parameterTransform_, jacobian);
          }

          jointIndex = this->skeleton_.joints[jointIndex].parent;
        }

        // -----------------------------------
        //  process scale derivatives above common ancestor
        // -----------------------------------
        {
          const T netScaleVal = -fac_k * ln2<T>() * direction_k.squaredNorm() +
              wgt * (radiusA_at_cp_k + radiusB_at_cp_k) * ln2<T>();
          size_t ancestorIndex = commonAncestor;
          while (ancestorIndex != kInvalidIndex) {
            const size_t paramIndex = ancestorIndex * kParametersPerJoint;
            if (this->activeJointParams_[paramIndex + 6]) {
              jacobian_jointParams_to_modelParams(
                  netScaleVal, paramIndex + 6, row, this->parameterTransform_, jacobian);
            }
            ancestorIndex = this->skeleton_.joints[ancestorIndex].parent;
          }
        }

        residual(row) = overlap[k] * wgt;
        row++;
      } // for each element of the packet k
    } // for each simd packet j
  } // for each collision geometry i

  usedRows = row;

  return drjit::sum(error);
}

template <typename T>
size_t SimdCollisionErrorFunctionT<T>::getJacobianSize() const {
  return validPairs_.size();
}

template class SimdCollisionErrorFunctionT<float>;
template class SimdCollisionErrorFunctionT<double>;

} // namespace momentum
