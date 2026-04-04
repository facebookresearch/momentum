/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/floor_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/character_solver/skinning_weight_iterator.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/mesh.h"

#include <queue>

namespace momentum {

template <typename T>
FloorErrorFunctionT<T>::FloorErrorFunctionT(
    const Character& character,
    std::vector<size_t> vertexIndices,
    T targetHeight,
    const Eigen::Vector3<T>& upDirection,
    size_t k)
    : SkeletonErrorFunctionT<T>(character.skeleton, character.parameterTransform),
      character_(character),
      vertexIndices_(std::move(vertexIndices)),
      targetHeight_(targetHeight),
      upDirection_(upDirection.normalized()),
      k_(k) {
  MT_CHECK(k > 0, "k must be greater than 0, got {}", k);
  MT_CHECK(!vertexIndices_.empty(), "FloorErrorFunction requires at least one vertex index");
  MT_CHECK(character.mesh != nullptr, "FloorErrorFunction requires a character with a mesh");
}

template <typename T>
FloorErrorFunctionT<T>::~FloorErrorFunctionT() = default;

template <typename T>
void FloorErrorFunctionT<T>::setTargetHeight(T height) {
  targetHeight_ = height;
}

template <typename T>
void FloorErrorFunctionT<T>::setUpDirection(const Eigen::Vector3<T>& upDirection) {
  upDirection_ = upDirection.normalized();
}

template <typename T>
void FloorErrorFunctionT<T>::setVertexIndices(std::vector<size_t> indices) {
  MT_CHECK(!indices.empty(), "FloorErrorFunction requires at least one vertex index");
  vertexIndices_ = std::move(indices);
}

template <typename T>
typename FloorErrorFunctionT<T>::MinHeightResult FloorErrorFunctionT<T>::calculateMinHeight(
    const MeshStateT<T>& meshState) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState.posedMesh_);

  const size_t effectiveK = std::min(k_, vertexIndices_.size());

  // Max heap for k lowest projections (top = largest of the k lowest)
  using ProjectionPair = std::pair<T, size_t>; // (projection, vertex_index)
  std::priority_queue<ProjectionPair, std::vector<ProjectionPair>, std::less<>> minQueue;

  for (const auto vertexIdx : vertexIndices_) {
    MT_CHECK(
        vertexIdx < meshState.posedMesh_->vertices.size(),
        "Vertex index {} out of range (mesh has {} vertices)",
        vertexIdx,
        meshState.posedMesh_->vertices.size());

    const T projection = upDirection_.dot(meshState.posedMesh_->vertices[vertexIdx]);

    if (minQueue.size() < effectiveK) {
      minQueue.push({projection, vertexIdx});
    } else if (projection < minQueue.top().first) {
      minQueue.pop();
      minQueue.push({projection, vertexIdx});
    }
  }

  MinHeightResult result;
  result.vertexIndices.reserve(effectiveK);
  result.vertexWeights.reserve(effectiveK);

  T projectionSum = T(0);
  while (!minQueue.empty()) {
    const auto& pair = minQueue.top();
    result.vertexIndices.push_back(pair.second);
    projectionSum += pair.first;
    minQueue.pop();
  }

  const T weight = T(1) / T(effectiveK);
  result.vertexWeights.resize(effectiveK, weight);
  result.avgProjection = projectionSum / T(effectiveK);

  return result;
}

template <typename T>
double FloorErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& /* state */,
    const MeshStateT<T>& meshState) {
  MT_PROFILE_FUNCTION();

  const auto minResult = calculateMinHeight(meshState);
  const T diff = minResult.avgProjection - targetHeight_;

  return this->weight_ * diff * diff;
}

template <typename T>
double FloorErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  const auto minResult = calculateMinHeight(meshState);
  const T diff = minResult.avgProjection - targetHeight_;

  const T wgt = T(2) * diff * this->weight_;

  for (size_t i = 0; i < minResult.vertexIndices.size(); ++i) {
    calculateVertexGradient(
        minResult.vertexIndices[i],
        state,
        meshState,
        wgt * minResult.vertexWeights[i] * upDirection_,
        gradient);
  }

  return this->weight_ * diff * diff;
}

template <typename T>
double FloorErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      jacobian.cols() == static_cast<Eigen::Index>(this->parameterTransform_.transform.cols()));
  MT_CHECK(jacobian.rows() >= 1);
  MT_CHECK(residual.rows() >= 1);

  const auto minResult = calculateMinHeight(meshState);
  const T diff = minResult.avgProjection - targetHeight_;

  const T wgt = std::sqrt(this->weight_);
  residual(0) = wgt * diff;

  jacobian.row(0).setZero();
  auto jac_row = jacobian.row(0);

  for (size_t i = 0; i < minResult.vertexIndices.size(); ++i) {
    calculateVertexJacobian(
        minResult.vertexIndices[i],
        state,
        meshState,
        wgt * minResult.vertexWeights[i] * upDirection_,
        jac_row);
  }

  usedRows = 1;
  return wgt * wgt * diff * diff;
}

template <typename T>
size_t FloorErrorFunctionT<T>::getJacobianSize() const {
  return 1;
}

template <typename T>
template <typename Derived>
void FloorErrorFunctionT<T>::calculateVertexJacobian(
    size_t vertexIndex,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    const Eigen::Vector3<T>& jacobianDirection,
    Eigen::MatrixBase<Derived>& jacobian) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState.restMesh_);

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *meshState.restMesh_, state, vertexIndex);

  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * jacobianDirection.dot(jointState.getTranslationDerivative(d)),
            paramIndex + d,
            this->parameterTransform_,
            jacobian);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * jacobianDirection.dot(jointState.getRotationDerivative(d, posd)),
            paramIndex + 3 + d,
            this->parameterTransform_,
            jacobian);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      jacobian_jointParams_to_modelParams<T>(
          boneWeight * jacobianDirection.dot(jointState.getScaleDerivative(posd)),
          paramIndex + 6,
          this->parameterTransform_,
          jacobian);
    }
  }
}

template <typename T>
void FloorErrorFunctionT<T>::calculateVertexGradient(
    size_t vertexIndex,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    const Eigen::Vector3<T>& gradientDirection,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState.restMesh_);

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *meshState.restMesh_, state, vertexIndex);

  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        gradient_jointParams_to_modelParams<T>(
            boneWeight * gradientDirection.dot(jointState.getTranslationDerivative(d)),
            paramIndex + d,
            this->parameterTransform_,
            gradient);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        gradient_jointParams_to_modelParams<T>(
            boneWeight * gradientDirection.dot(jointState.getRotationDerivative(d, posd)),
            paramIndex + 3 + d,
            this->parameterTransform_,
            gradient);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      gradient_jointParams_to_modelParams<T>(
          boneWeight * gradientDirection.dot(jointState.getScaleDerivative(posd)),
          paramIndex + 6,
          this->parameterTransform_,
          gradient);
    }
  }
}

template class FloorErrorFunctionT<float>;
template class FloorErrorFunctionT<double>;

} // namespace momentum
