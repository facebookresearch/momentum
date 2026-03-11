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
#include "momentum/common/profile.h"

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
    : SkeletonErrorFunctionT<T>(skeleton, parameterTransform),
      skeletonDerivative_(
          skeleton,
          parameterTransform,
          this->activeJointParams_,
          this->enabledParameters_) {}

template <typename T>
void CenterOfMassErrorFunctionT<T>::addConstraint(const ConstraintType& constraint) {
  MT_CHECK(
      constraint.jointIndices.size() == constraint.masses.size(),
      "Joint indices and masses must have the same size");
  MT_CHECK(!constraint.jointIndices.empty(), "Constraint must have at least one joint");
  MT_CHECK(constraint.totalMass() > T(0), "Total mass must be positive");

  // Validate joint indices are within bounds
  for (const auto jointIdx : constraint.jointIndices) {
    MT_CHECK(jointIdx < this->skeleton_.joints.size(), "Invalid joint index: {}", jointIdx);
  }

  constraints_.push_back(constraint);
}

template <typename T>
void CenterOfMassErrorFunctionT<T>::setConstraints(std::span<const ConstraintType> constraints) {
  constraints_.assign(constraints.begin(), constraints.end());
}

template <typename T>
void CenterOfMassErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
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
double CenterOfMassErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /*meshState*/) {
  if (state.jointState.empty() || constraints_.empty()) {
    return 0.0;
  }

  double error = 0.0;

  for (const auto& constr : constraints_) {
    const T invTotalMass = T(1) / constr.totalMass();

    // Compute center of mass: CoM = (sum m_k * p_k) / (sum m_k)
    Eigen::Vector3<T> com = Eigen::Vector3<T>::Zero();
    for (size_t i = 0; i < constr.jointIndices.size(); ++i) {
      const Eigen::Vector3<T> worldPos = state.jointState[constr.jointIndices[i]].translation();
      com += constr.masses[i] * worldPos;
    }
    com *= invTotalMass;

    // Residual: f = CoM - target
    const Eigen::Vector3<T> residual = com - constr.target;
    error += constr.weight * residual.squaredNorm() * this->weight_;
  }

  return error;
}

template <typename T>
double CenterOfMassErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /*meshState*/,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  if (state.jointState.empty() || constraints_.empty()) {
    return 0.0;
  }

  double error = 0.0;

  // Pre-allocate joint gradient storage
  Eigen::VectorX<T> jointGrad =
      Eigen::VectorX<T>::Zero(this->parameterTransform_.numAllModelParameters());

  for (const auto& constr : constraints_) {
    const T invTotalMass = T(1) / constr.totalMass();

    // Compute center of mass
    Eigen::Vector3<T> com = Eigen::Vector3<T>::Zero();
    for (size_t i = 0; i < constr.jointIndices.size(); ++i) {
      const Eigen::Vector3<T> worldPos = state.jointState[constr.jointIndices[i]].translation();
      com += constr.masses[i] * worldPos;
    }
    com *= invTotalMass;

    // Residual: f = CoM - target
    const Eigen::Vector3<T> residual = com - constr.target;
    error += constr.weight * residual.squaredNorm() * this->weight_;

    const T wgt = T(2) * constr.weight * this->weight_;

    // The weighted residual for accumulateJointGradient is 2 * weight * f.
    // The dfdv encodes the normalizedMass factor: dfdv = (m_k / totalMass) * I_3x3
    // accumulateJointGradient computes: gradient += weightedResidual^T * dfdv * d(worldPos)/dq
    const Eigen::Vector3<T> weightedRes = wgt * residual;

    for (size_t i = 0; i < constr.jointIndices.size(); ++i) {
      const size_t jointIdx = constr.jointIndices[i];
      const T normalizedMass = constr.masses[i] * invTotalMass;
      const Eigen::Vector3<T> worldPos = state.jointState[jointIdx].translation();

      const Eigen::Matrix<T, 3, 3> dfdv = normalizedMass * Eigen::Matrix<T, 3, 3>::Identity();
      std::array<Eigen::Vector3<T>, 1> worldVecs = {worldPos};
      std::array<Eigen::Matrix<T, 3, 3>, 1> dfdvArr = {dfdv};
      std::array<uint8_t, 1> isPoints = {1};

      skeletonDerivative_.template accumulateJointGradient<3>(
          jointIdx,
          std::span<const Eigen::Vector3<T>>(worldVecs.data(), 1),
          std::span<const Eigen::Matrix<T, 3, 3>>(dfdvArr.data(), 1),
          std::span<const uint8_t>(isPoints.data(), 1),
          weightedRes,
          state.jointState,
          jointGrad);
    }
  }

  gradient += jointGrad;
  return error;
}

template <typename T>
double CenterOfMassErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& /*meshState*/,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  if (state.jointState.empty() || constraints_.empty()) {
    usedRows = 0;
    return 0.0;
  }

  double error = 0.0;
  size_t rowIndex = 0;

  for (const auto& constr : constraints_) {
    const T invTotalMass = T(1) / constr.totalMass();

    // Compute center of mass
    Eigen::Vector3<T> com = Eigen::Vector3<T>::Zero();
    for (size_t i = 0; i < constr.jointIndices.size(); ++i) {
      const Eigen::Vector3<T> worldPos = state.jointState[constr.jointIndices[i]].translation();
      com += constr.masses[i] * worldPos;
    }
    com *= invTotalMass;

    // Residual: f = CoM - target
    const Eigen::Vector3<T> res = com - constr.target;
    error += constr.weight * res.squaredNorm() * this->weight_;

    const T wgt = std::sqrt(constr.weight * this->weight_);
    residual.template segment<3>(rowIndex) = wgt * res;

    // For each joint, accumulate Jacobian
    for (size_t i = 0; i < constr.jointIndices.size(); ++i) {
      const size_t jointIdx = constr.jointIndices[i];
      const T normalizedMass = constr.masses[i] * invTotalMass;
      const Eigen::Vector3<T> worldPos = state.jointState[jointIdx].translation();

      const Eigen::Matrix<T, 3, 3> dfdv = normalizedMass * Eigen::Matrix<T, 3, 3>::Identity();
      std::array<Eigen::Vector3<T>, 1> worldVecs = {worldPos};
      std::array<Eigen::Matrix<T, 3, 3>, 1> dfdvArr = {dfdv};
      std::array<uint8_t, 1> isPoints = {1};

      skeletonDerivative_.template accumulateJointJacobian<3>(
          jointIdx,
          std::span<const Eigen::Vector3<T>>(worldVecs.data(), 1),
          std::span<const Eigen::Matrix<T, 3, 3>>(dfdvArr.data(), 1),
          std::span<const uint8_t>(isPoints.data(), 1),
          wgt,
          state.jointState,
          jacobian,
          rowIndex);
    }

    rowIndex += 3;
  }

  usedRows = static_cast<int>(rowIndex);
  return error;
}

template <typename T>
size_t CenterOfMassErrorFunctionT<T>::getJacobianSize() const {
  return 3 * constraints_.size();
}

// Explicit template instantiations
template class CenterOfMassErrorFunctionT<float>;
template class CenterOfMassErrorFunctionT<double>;

} // namespace momentum
