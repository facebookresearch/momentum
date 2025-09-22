/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/vertex_vertex_distance_error_function.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/character_solver/skinning_weight_iterator.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/mesh.h"

namespace momentum {

template <typename T>
VertexVertexDistanceErrorFunctionT<T>::VertexVertexDistanceErrorFunctionT(
    const Character& character)
    : SkeletonErrorFunctionT<T>(character.skeleton, character.parameterTransform),
      character_(character) {
  MT_CHECK(static_cast<bool>(character.mesh));
  MT_CHECK(static_cast<bool>(character.skinWeights));

  this->restMesh_ = std::make_unique<MeshT<T>>(character.mesh->template cast<T>());
  this->posedMesh_ = std::make_unique<MeshT<T>>(character.mesh->template cast<T>());
}

template <typename T>
VertexVertexDistanceErrorFunctionT<T>::~VertexVertexDistanceErrorFunctionT() {}

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
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state) {
  MT_PROFILE_FUNCTION();

  updateMeshes(modelParameters, state);

  double error = 0.0;

  for (const auto& constraint : constraints_) {
    const auto& pos1 = posedMesh_->vertices[constraint.vertexIndex1];
    const auto& pos2 = posedMesh_->vertices[constraint.vertexIndex2];

    const T actualDistance = (pos1 - pos2).norm();
    const T distanceDiff = actualDistance - constraint.targetDistance;

    error += constraint.weight * distanceDiff * distanceDiff;
  }

  return error * this->weight_;
}

template <typename T>
void VertexVertexDistanceErrorFunctionT<T>::updateMeshes(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state) {
  MT_PROFILE_FUNCTION();

  // Update rest mesh with blend shapes if present
  bool doUpdateNormals = false;
  if (this->character_.blendShape) {
    const BlendWeightsT<T> blendWeights =
        extractBlendWeights(this->parameterTransform_, modelParameters);
    this->character_.blendShape->computeShape(blendWeights, this->restMesh_->vertices);
    doUpdateNormals = true;
  }

  if (doUpdateNormals) {
    this->restMesh_->updateNormals();
  }

  // Apply skinning to get the posed mesh
  applySSD(
      cast<T>(character_.inverseBindPose),
      *this->character_.skinWeights,
      *this->restMesh_,
      state,
      *this->posedMesh_);
}

template <typename T>
double VertexVertexDistanceErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  updateMeshes(modelParameters, state);

  double error = 0.0;

  for (const auto& constraint : constraints_) {
    error += calculateGradient(modelParameters, state, constraint, gradient);
  }

  return error * this->weight_;
}

template <typename T>
double VertexVertexDistanceErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      jacobian.cols() == static_cast<Eigen::Index>(this->parameterTransform_.transform.cols()));
  MT_CHECK(jacobian.rows() >= static_cast<Eigen::Index>(constraints_.size()));
  MT_CHECK(residual.rows() >= static_cast<Eigen::Index>(constraints_.size()));

  updateMeshes(modelParameters, state);

  double error = 0.0;

  for (size_t i = 0; i < constraints_.size(); i++) {
    T residualValue;
    error += calculateJacobian(
        modelParameters,
        state,
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
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const VertexVertexDistanceConstraintT<T>& constraint,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    T& residual) const {
  MT_PROFILE_FUNCTION();

  const auto& pos1 = posedMesh_->vertices[constraint.vertexIndex1];
  const auto& pos2 = posedMesh_->vertices[constraint.vertexIndex2];

  const Eigen::Vector3<T> diff = pos1 - pos2;
  const T actualDistance = diff.norm();

  // Handle degenerate case where vertices are at the same position
  if (actualDistance == T(0)) {
    residual = T(0);
    return T(0); // No meaningful jacobian when distance is zero
  }

  const T distanceDiff = actualDistance - constraint.targetDistance;
  const Eigen::Vector3<T> distanceGradient = diff / actualDistance; // normalized difference vector

  // Weight for the jacobian: sqrt(weight * this->weight_)
  const T wgt = std::sqrt(constraint.weight * this->weight_);

  // Set residual: wgt * distanceDiff
  residual = wgt * distanceDiff;

  // Calculate jacobian contribution from vertex1 (positive contribution)
  calculateVertexJacobian(
      modelParameters, state, constraint.vertexIndex1, wgt * distanceGradient, jacobian);

  // Calculate jacobian contribution from vertex2 (negative contribution)
  calculateVertexJacobian(
      modelParameters, state, constraint.vertexIndex2, -wgt * distanceGradient, jacobian);

  return wgt * wgt * distanceDiff * distanceDiff;
}

template <typename T>
void VertexVertexDistanceErrorFunctionT<T>::calculateVertexJacobian(
    const ModelParametersT<T>& /*modelParameters*/,
    const SkeletonStateT<T>& state,
    int vertexIndex,
    const Eigen::Vector3<T>& jacobianDirection,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian) const {
  MT_PROFILE_FUNCTION();

  SkinningWeightIteratorT<T> skinningIter(this->character_, *this->restMesh_, state, vertexIndex);

  // Handle derivatives wrt joint parameters
  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    // Check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // Calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        // Jacobian wrt translation:
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * jacobianDirection.dot(jointState.getTranslationDerivative(d)),
            paramIndex + d,
            this->parameterTransform_,
            jacobian);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // Jacobian wrt rotation:
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * jacobianDirection.dot(jointState.getRotationDerivative(d, posd)),
            paramIndex + 3 + d,
            this->parameterTransform_,
            jacobian);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // Jacobian wrt scale:
      jacobian_jointParams_to_modelParams<T>(
          boneWeight * jacobianDirection.dot(jointState.getScaleDerivative(posd)),
          paramIndex + 6,
          this->parameterTransform_,
          jacobian);
    }
  }

  // Handle derivatives wrt blend shape parameters
  if (this->character_.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < this->parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = this->parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          this->character_.blendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, vertexIndex, d_restPos, d_worldPos);

      jacobian(0, paramIdx) += jacobianDirection.dot(d_worldPos);
    }
  }

  // Handle derivatives wrt face expression blend shape parameters
  if (this->character_.faceExpressionBlendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < this->parameterTransform_.faceExpressionParameters.size();
         ++iBlendShape) {
      const auto paramIdx = this->parameterTransform_.faceExpressionParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          this->character_.faceExpressionBlendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, vertexIndex, d_restPos, d_worldPos);

      jacobian(0, paramIdx) += jacobianDirection.dot(d_worldPos);
    }
  }
}

template <typename T>
double VertexVertexDistanceErrorFunctionT<T>::calculateGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const VertexVertexDistanceConstraintT<T>& constraint,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_PROFILE_FUNCTION();

  const auto& pos1 = posedMesh_->vertices[constraint.vertexIndex1];
  const auto& pos2 = posedMesh_->vertices[constraint.vertexIndex2];

  const Eigen::Vector3<T> diff = pos1 - pos2;
  const T actualDistance = diff.norm();

  // Handle degenerate case where vertices are at the same position
  if (actualDistance == T(0)) {
    return T(0); // No meaningful gradient when distance is zero
  }

  const T distanceDiff = actualDistance - constraint.targetDistance;
  const Eigen::Vector3<T> distanceGradient = diff / actualDistance; // normalized difference vector

  // Weight for the gradient: 2 * weight * distanceDiff * this->weight_
  const T wgt = T(2) * constraint.weight * distanceDiff * this->weight_;

  // Calculate gradient contribution from vertex1 (positive contribution)
  calculateVertexGradient(
      modelParameters, state, constraint.vertexIndex1, wgt * distanceGradient, gradient);

  // Calculate gradient contribution from vertex2 (negative contribution)
  calculateVertexGradient(
      modelParameters, state, constraint.vertexIndex2, -wgt * distanceGradient, gradient);

  return constraint.weight * distanceDiff * distanceDiff;
}

template <typename T>
void VertexVertexDistanceErrorFunctionT<T>::calculateVertexGradient(
    const ModelParametersT<T>& /*modelParameters*/,
    const SkeletonStateT<T>& state,
    int vertexIndex,
    const Eigen::Vector3<T>& gradientDirection,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_PROFILE_FUNCTION();

  SkinningWeightIteratorT<T> skinningIter(this->character_, *this->restMesh_, state, vertexIndex);

  // Handle derivatives wrt joint parameters
  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    // Check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // Calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        // Gradient wrt translation:
        gradient_jointParams_to_modelParams(
            boneWeight * gradientDirection.dot(jointState.getTranslationDerivative(d)),
            paramIndex + d,
            this->parameterTransform_,
            gradient);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // Gradient wrt rotation:
        gradient_jointParams_to_modelParams(
            boneWeight * gradientDirection.dot(jointState.getRotationDerivative(d, posd)),
            paramIndex + 3 + d,
            this->parameterTransform_,
            gradient);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // Gradient wrt scale:
      gradient_jointParams_to_modelParams(
          boneWeight * gradientDirection.dot(jointState.getScaleDerivative(posd)),
          paramIndex + 6,
          this->parameterTransform_,
          gradient);
    }
  }

  // Handle derivatives wrt blend shape parameters
  if (this->character_.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < this->parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = this->parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          this->character_.blendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, vertexIndex, d_restPos, d_worldPos);

      gradient[paramIdx] += gradientDirection.dot(d_worldPos);
    }
  }

  // Handle derivatives wrt face expression blend shape parameters
  if (this->character_.faceExpressionBlendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < this->parameterTransform_.faceExpressionParameters.size();
         ++iBlendShape) {
      const auto paramIdx = this->parameterTransform_.faceExpressionParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          this->character_.faceExpressionBlendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, vertexIndex, d_restPos, d_worldPos);

      gradient[paramIdx] += gradientDirection.dot(d_worldPos);
    }
  }
}

template <typename T>
void VertexVertexDistanceErrorFunctionT<T>::calculateDWorldPos(
    const SkeletonStateT<T>& state,
    int vertexIndex,
    const Eigen::Vector3<T>& d_restPos,
    Eigen::Vector3<T>& d_worldPos) const {
  const auto& skinWeights = *character_.skinWeights;

  for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
    const auto w = skinWeights.weight(vertexIndex, i);
    const auto parentBone = skinWeights.index(vertexIndex, i);
    if (w > 0) {
      d_worldPos += w *
          (state.jointState[parentBone].transform.toLinear() *
           (character_.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
    }
  }
}

// Explicit template instantiations
template class VertexVertexDistanceErrorFunctionT<float>;
template class VertexVertexDistanceErrorFunctionT<double>;

template struct VertexVertexDistanceConstraintT<float>;
template struct VertexVertexDistanceConstraintT<double>;

} // namespace momentum
