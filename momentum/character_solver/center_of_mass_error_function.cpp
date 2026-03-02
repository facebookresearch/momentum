/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/center_of_mass_error_function.h"

#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"

#include <numeric>

namespace momentum {

// CenterOfMassConstraintT methods
template <typename T>
T CenterOfMassConstraintT<T>::totalMass() const {
  return std::accumulate(masses.begin(), masses.end(), T(0));
}

template struct CenterOfMassConstraintT<float>;
template struct CenterOfMassConstraintT<double>;

// CenterOfMassErrorFunctionT methods
template <typename T>
CenterOfMassErrorFunctionT<T>::CenterOfMassErrorFunctionT(
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform)
    : MultiSourceErrorFunctionT<T, 3>(skeleton, parameterTransform) {}

template <typename T>
void CenterOfMassErrorFunctionT<T>::addConstraint(const ConstraintType& constraint) {
  MT_CHECK(
      constraint.jointIndices.size() == constraint.masses.size(),
      "Joint indices and masses must have the same size");
  MT_CHECK(!constraint.jointIndices.empty(), "Constraint must have at least one joint");
  MT_CHECK(constraint.totalMass() > T(0), "Total mass must be positive");

  constraints_.push_back(constraint);
  contributionsCacheValid_ = false;
}

template <typename T>
void CenterOfMassErrorFunctionT<T>::setConstraints(std::span<const ConstraintType> constraints) {
  constraints_.assign(constraints.begin(), constraints.end());
  contributionsCacheValid_ = false;
}

template <typename T>
void CenterOfMassErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
  contributionsCache_.clear();
  contributionsCacheValid_ = false;
}

template <typename T>
const std::vector<CenterOfMassConstraintT<T>>& CenterOfMassErrorFunctionT<T>::getConstraints()
    const {
  return constraints_;
}

template <typename T>
size_t CenterOfMassErrorFunctionT<T>::getNumConstraints() const {
  return constraints_.size();
}

template <typename T>
T CenterOfMassErrorFunctionT<T>::getConstraintWeight(size_t constrIndex) const {
  return static_cast<T>(constraints_[constrIndex].weight);
}

template <typename T>
void CenterOfMassErrorFunctionT<T>::rebuildContributionsCache() const {
  contributionsCache_.clear();
  contributionsCache_.resize(constraints_.size());

  for (size_t i = 0; i < constraints_.size(); ++i) {
    const auto& constraint = constraints_[i];
    auto& contributions = contributionsCache_[i];
    contributions.reserve(constraint.jointIndices.size());

    for (size_t j = 0; j < constraint.jointIndices.size(); ++j) {
      // Each joint contributes its origin position (local offset = zero)
      contributions.push_back(
          SourceT<T>::jointPoint(constraint.jointIndices[j], Eigen::Vector3<T>::Zero()));
    }
  }

  contributionsCacheValid_ = true;
}

template <typename T>
std::span<const SourceT<T>> CenterOfMassErrorFunctionT<T>::getContributions(
    size_t constrIndex) const {
  MT_CHECK(constrIndex < constraints_.size(), "Constraint index out of range");

  if (!contributionsCacheValid_) {
    rebuildContributionsCache();
  }

  return std::span<const SourceT<T>>(contributionsCache_[constrIndex]);
}

template <typename T>
void CenterOfMassErrorFunctionT<T>::evalFunction(
    size_t constrIndex,
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& /*meshState*/,
    std::span<const Eigen::Vector3<T>> worldVecs,
    FuncType& f,
    std::span<DfdvType> dfdv) const {
  MT_CHECK(constrIndex < constraints_.size(), "Constraint index out of range");

  const auto& constraint = constraints_[constrIndex];
  MT_CHECK(
      worldVecs.size() == constraint.jointIndices.size(),
      "World vectors size must match joint indices size");

  const T invTotalMass = T(1) / constraint.totalMass();

  // Compute center of mass: CoM = (Σ m_k * p_k) / (Σ m_k)
  Eigen::Vector3<T> com = Eigen::Vector3<T>::Zero();
  for (size_t i = 0; i < worldVecs.size(); ++i) {
    com += constraint.masses[i] * worldVecs[i];
  }
  com *= invTotalMass;

  // Residual: f = CoM - target
  f = com - constraint.target;

  // Compute derivatives if requested
  if (!dfdv.empty()) {
    MT_CHECK(dfdv.size() == worldVecs.size(), "dfdv size must match world vectors size");

    // df/dv_k = (m_k / total_mass) * I
    // where I is the 3x3 identity matrix
    for (size_t i = 0; i < dfdv.size(); ++i) {
      const T normalizedMass = constraint.masses[i] * invTotalMass;
      dfdv[i] = normalizedMass * Eigen::Matrix<T, 3, 3>::Identity();
    }
  }
}

// Explicit template instantiations
template class CenterOfMassErrorFunctionT<float>;
template class CenterOfMassErrorFunctionT<double>;

} // namespace momentum
