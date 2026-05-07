/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/vertex_vertex_distance_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/skeleton_derivative.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"

namespace momentum {

template <typename T>
VertexVertexDistanceErrorFunctionT<T>::VertexVertexDistanceErrorFunctionT(
    const Character& character)
    : SkeletonErrorFunctionT<T>(character.skeleton, character.parameterTransform),
      character_(character) {
  MT_CHECK(static_cast<bool>(character.mesh));
  MT_CHECK(static_cast<bool>(character.skinWeights));
}

template <typename T>
VertexVertexDistanceErrorFunctionT<T>::~VertexVertexDistanceErrorFunctionT() = default;

template <typename T>
void VertexVertexDistanceErrorFunctionT<T>::addConstraint(
    int vertexIndex1,
    int vertexIndex2,
    T weight,
    T targetDistance) {
  MT_CHECK(vertexIndex1 >= 0 && ((size_t)vertexIndex1) < character_.mesh->vertices.size());
  MT_CHECK(vertexIndex2 >= 0 && ((size_t)vertexIndex2) < character_.mesh->vertices.size());
  MT_CHECK(vertexIndex1 != vertexIndex2);

  constraints_.push_back(
      VertexVertexDistanceConstraintT<T>{vertexIndex1, vertexIndex2, weight, targetDistance});
}

template <typename T>
void VertexVertexDistanceErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
double VertexVertexDistanceErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& /* state */,
    const MeshStateT<T>& meshState) {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState.posedMesh_);

  double error = 0.0;

  for (const auto& constraint : constraints_) {
    const auto& pos1 = meshState.posedMesh_->vertices[constraint.vertexIndex1];
    const auto& pos2 = meshState.posedMesh_->vertices[constraint.vertexIndex2];

    const T actualDistance = (pos1 - pos2).norm();
    const T distanceDiff = actualDistance - constraint.targetDistance;

    error += constraint.weight * distanceDiff * distanceDiff;
  }

  return error * this->weight_;
}

template <typename T>
double VertexVertexDistanceErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      meshState.posedMesh_, "MeshState must have posed mesh for VertexVertexDistanceErrorFunction");

  double error = 0.0;

  for (const auto& constraint : constraints_) {
    error += calculateGradient(modelParameters, state, meshState, constraint, gradient);
  }

  return error * this->weight_;
}

template <typename T>
double VertexVertexDistanceErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      jacobian.cols() == static_cast<Eigen::Index>(this->parameterTransform_.transform.cols()));
  MT_CHECK(jacobian.rows() >= static_cast<Eigen::Index>(constraints_.size()));
  MT_CHECK(residual.rows() >= static_cast<Eigen::Index>(constraints_.size()));

  MT_CHECK_NOTNULL(meshState.posedMesh_);

  double error = 0.0;

  for (size_t i = 0; i < constraints_.size(); i++) {
    T residualValue;
    error += calculateJacobian(
        modelParameters,
        state,
        meshState,
        constraints_[i],
        jacobian.block(i, 0, 1, modelParameters.size()),
        residualValue);
    residual(i) = residualValue;
  }

  usedRows = static_cast<int>(constraints_.size());
  return error;
}

template <typename T>
size_t VertexVertexDistanceErrorFunctionT<T>::getJacobianSize() const {
  return constraints_.size();
}

template <typename T>
double VertexVertexDistanceErrorFunctionT<T>::calculateJacobian(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    const VertexVertexDistanceConstraintT<T>& constraint,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    T& residual) const {
  MT_PROFILE_FUNCTION();

  const auto& pos1 = meshState.posedMesh_->vertices[constraint.vertexIndex1];
  const auto& pos2 = meshState.posedMesh_->vertices[constraint.vertexIndex2];

  const Eigen::Vector3<T> diff = pos1 - pos2;
  const T actualDistance = diff.norm();

  // Handle degenerate case where vertices are at the same position
  if (actualDistance < Eps<T>(1e-8f, 1e-14)) {
    residual = T(0);
    return T(0); // No meaningful jacobian when distance is zero
  }

  const T distanceDiff = actualDistance - constraint.targetDistance;
  const Eigen::Vector3<T> distanceGradient = diff / actualDistance; // normalized difference vector

  // Weight for the jacobian: sqrt(weight * this->weight_)
  const T wgt = std::sqrt(constraint.weight * this->weight_);

  // Set residual: wgt * distanceDiff
  residual = wgt * distanceDiff;

  // df/dv for distance function is the normalized direction (1×3 matrix)
  // For vertex1: positive contribution, for vertex2: negative contribution
  const Eigen::Matrix<T, 1, 3> dfdv1 = distanceGradient.transpose();
  const Eigen::Matrix<T, 1, 3> dfdv2 = -distanceGradient.transpose();

  SkeletonDerivativeT<T> deriv(
      this->skeleton_,
      this->parameterTransform_,
      this->activeJointParams_,
      this->enabledParameters_);

  // Calculate jacobian contribution from vertex1 (positive contribution)
  deriv.template accumulateVertexJacobian<1>(
      constraint.vertexIndex1, pos1, dfdv1, wgt, state, meshState, character_, jacobian, 0);

  // Calculate jacobian contribution from vertex2 (negative contribution)
  deriv.template accumulateVertexJacobian<1>(
      constraint.vertexIndex2, pos2, dfdv2, wgt, state, meshState, character_, jacobian, 0);

  return wgt * wgt * distanceDiff * distanceDiff;
}

template <typename T>
double VertexVertexDistanceErrorFunctionT<T>::calculateGradient(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    const VertexVertexDistanceConstraintT<T>& constraint,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_PROFILE_FUNCTION();

  const auto& pos1 = meshState.posedMesh_->vertices[constraint.vertexIndex1];
  const auto& pos2 = meshState.posedMesh_->vertices[constraint.vertexIndex2];

  const Eigen::Vector3<T> diff = pos1 - pos2;
  const T actualDistance = diff.norm();

  // Handle degenerate case where vertices are at the same position
  if (actualDistance < Eps<T>(1e-8f, 1e-14)) {
    return T(0); // No meaningful gradient when distance is zero
  }

  const T distanceDiff = actualDistance - constraint.targetDistance;
  const Eigen::Vector3<T> distanceGradient = diff / actualDistance; // normalized difference vector

  // df/dv for distance function is the normalized direction (1×3 matrix)
  // For vertex1: positive contribution, for vertex2: negative contribution
  const Eigen::Matrix<T, 1, 3> dfdv1 = distanceGradient.transpose();
  const Eigen::Matrix<T, 1, 3> dfdv2 = -distanceGradient.transpose();

  // weightedResidual = 2 * weight * loss_deriv * f
  // For distance: f = distanceDiff, loss = 0.5 * f^2, loss_deriv = 1
  // So weightedResidual = 2 * weight * distanceDiff
  const Eigen::Matrix<T, 1, 1> weightedResidual =
      Eigen::Matrix<T, 1, 1>::Constant(T(2) * constraint.weight * this->weight_ * distanceDiff);

  SkeletonDerivativeT<T> deriv(
      this->skeleton_,
      this->parameterTransform_,
      this->activeJointParams_,
      this->enabledParameters_);

  // Calculate gradient contribution from vertex1 (positive contribution)
  deriv.template accumulateVertexGradient<1>(
      constraint.vertexIndex1,
      pos1,
      dfdv1,
      weightedResidual,
      state,
      meshState,
      character_,
      gradient);

  // Calculate gradient contribution from vertex2 (negative contribution)
  deriv.template accumulateVertexGradient<1>(
      constraint.vertexIndex2,
      pos2,
      dfdv2,
      weightedResidual,
      state,
      meshState,
      character_,
      gradient);

  return constraint.weight * distanceDiff * distanceDiff;
}

// Explicit template instantiations
template class VertexVertexDistanceErrorFunctionT<float>;
template class VertexVertexDistanceErrorFunctionT<double>;

template struct VertexVertexDistanceConstraintT<float>;
template struct VertexVertexDistanceConstraintT<double>;

} // namespace momentum
