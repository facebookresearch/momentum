/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/vertex_sequence_error_function.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/character_solver/skinning_weight_iterator.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/mesh.h"

#include <numeric>

namespace momentum {

template <typename T>
VertexSequenceErrorFunctionT<T>::VertexSequenceErrorFunctionT(const Character& character)
    : SequenceErrorFunctionT<T>(character.skeleton, character.parameterTransform),
      character_(character) {
  MT_CHECK(static_cast<bool>(character_.mesh));
  MT_CHECK(static_cast<bool>(character_.skinWeights));

  neutralMesh_ = std::make_unique<MeshT<T>>(character_.mesh->template cast<T>());
  restMesh_ = std::make_unique<MeshT<T>>(character_.mesh->template cast<T>());
  posedMesh0_ = std::make_unique<MeshT<T>>(character_.mesh->template cast<T>());
  posedMesh1_ = std::make_unique<MeshT<T>>(character_.mesh->template cast<T>());
}

template <typename T>
void VertexSequenceErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
void VertexSequenceErrorFunctionT<T>::addConstraint(
    int vertexIndex,
    T weight,
    const Eigen::Vector3<T>& targetVelocity) {
  MT_CHECK(vertexIndex >= 0 && ((size_t)vertexIndex) < character_.mesh->vertices.size());
  constraints_.push_back(VertexVelocityConstraintT<T>{vertexIndex, weight, targetVelocity});
}

template <typename T>
void VertexSequenceErrorFunctionT<T>::updateMeshes(
    gsl::span<const ModelParametersT<T>> modelParameters,
    gsl::span<const SkeletonStateT<T>> skelStates) const {
  MT_PROFILE_FUNCTION();
  MT_CHECK(modelParameters.size() == 2);
  MT_CHECK(skelStates.size() == 2);

  // Since the rest mesh is the same for both frames, we only need to update it once
  // We'll use the blend shape parameters from frame 0 (could be either frame)
  bool doUpdateNormals = false;
  if (character_.blendShape) {
    const BlendWeightsT<T> blendWeights =
        extractBlendWeights(this->parameterTransform_, modelParameters[0]);
    character_.blendShape->computeShape(blendWeights, restMesh_->vertices);
    doUpdateNormals = true;
  }
  if (character_.faceExpressionBlendShape) {
    if (!character_.blendShape) {
      Eigen::Map<Eigen::VectorX<T>> outputVec(
          &restMesh_->vertices[0][0], restMesh_->vertices.size() * 3);
      const Eigen::Map<Eigen::VectorX<T>> baseVec(
          &neutralMesh_->vertices[0][0], neutralMesh_->vertices.size() * 3);
      outputVec = baseVec.template cast<T>();
    }
    const BlendWeightsT<T> faceExpressionBlendWeights =
        extractFaceExpressionBlendWeights(this->parameterTransform_, modelParameters[0]);
    character_.faceExpressionBlendShape->applyDeltas(
        faceExpressionBlendWeights, restMesh_->vertices);
    doUpdateNormals = true;
  }
  if (doUpdateNormals) {
    restMesh_->updateNormals();
  }

  // Apply skinning for both frames using the same rest mesh
  applySSD(
      cast<T>(character_.inverseBindPose),
      *character_.skinWeights,
      *restMesh_,
      skelStates[0],
      *posedMesh0_);

  applySSD(
      cast<T>(character_.inverseBindPose),
      *character_.skinWeights,
      *restMesh_,
      skelStates[1],
      *posedMesh1_);
}

template <typename T>
double VertexSequenceErrorFunctionT<T>::getError(
    gsl::span<const ModelParametersT<T>> modelParameters,
    gsl::span<const SkeletonStateT<T>> skelStates,
    gsl::span<const MeshStateT<T>> /*meshStates*/) const {
  MT_PROFILE_FUNCTION();
  MT_CHECK(modelParameters.size() == 2);
  MT_CHECK(skelStates.size() == 2);

  updateMeshes(modelParameters, skelStates);

  double error = 0.0;

  // Calculate vertex velocities and compare with target velocities
  for (const auto& constraint : constraints_) {
    const Eigen::Vector3<T>& vertex0 = posedMesh0_->vertices[constraint.vertexIndex];
    const Eigen::Vector3<T>& vertex1 = posedMesh1_->vertices[constraint.vertexIndex];

    // Compute actual velocity (difference between frames)
    const Eigen::Vector3<T> actualVelocity = vertex1 - vertex0;

    // Compute velocity difference
    const Eigen::Vector3<T> velocityDiff = actualVelocity - constraint.targetVelocity;

    error += constraint.weight * velocityDiff.squaredNorm() * kVelocityWeight;
  }

  return error * this->weight_;
}

template <typename T>
double VertexSequenceErrorFunctionT<T>::calculateVelocityGradient(
    gsl::span<const SkeletonStateT<T>> skelStates,
    const VertexVelocityConstraintT<T>& constraint,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  const Eigen::Vector3<T>& vertex0 = posedMesh0_->vertices[constraint.vertexIndex];
  const Eigen::Vector3<T>& vertex1 = posedMesh1_->vertices[constraint.vertexIndex];

  const Eigen::Vector3<T> actualVelocity = vertex1 - vertex0;
  const Eigen::Vector3<T> velocityDiff = actualVelocity - constraint.targetVelocity;

  const T wgt = constraint.weight * 2.0f * kVelocityWeight * this->weight_;

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();

  Eigen::Ref<Eigen::VectorX<T>> grad0 = gradient.segment(0, nParam);
  Eigen::Ref<Eigen::VectorX<T>> grad1 = gradient.segment(nParam, nParam);

  // Gradient with respect to frame 0 (negative contribution to velocity)
  {
    SkinningWeightIteratorT<T> skinningIter(
        character_, *restMesh_, skelStates[0], constraint.vertexIndex);

    while (!skinningIter.finished()) {
      size_t jointIndex = 0;
      T boneWeight;
      Eigen::Vector3<T> pos;
      std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

      MT_CHECK(jointIndex < this->skeleton_.joints.size());

      const auto& jointState = skelStates[0].jointState[jointIndex];
      const size_t paramIndex = jointIndex * kParametersPerJoint;
      const Eigen::Vector3<T> posd = pos - jointState.translation();

      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          // Create a temporary gradient vector for frame 0
          gradient_jointParams_to_modelParams(
              -boneWeight * velocityDiff.dot(jointState.getTranslationDerivative(d)) * wgt,
              paramIndex + d,
              this->parameterTransform_,
              grad0);
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          gradient_jointParams_to_modelParams(
              -boneWeight * velocityDiff.dot(jointState.getRotationDerivative(d, posd)) * wgt,
              paramIndex + 3 + d,
              this->parameterTransform_,
              grad0);
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        gradient_jointParams_to_modelParams(
            -boneWeight * velocityDiff.dot(jointState.getScaleDerivative(posd)) * wgt,
            paramIndex + 6,
            this->parameterTransform_,
            grad0);
      }
    }
  }

  // Gradient with respect to frame 1 (positive contribution to velocity)
  {
    SkinningWeightIteratorT<T> skinningIter(
        character_, *restMesh_, skelStates[1], constraint.vertexIndex);

    while (!skinningIter.finished()) {
      size_t jointIndex = 0;
      T boneWeight;
      Eigen::Vector3<T> pos;
      std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

      MT_CHECK(jointIndex < this->skeleton_.joints.size());

      const auto& jointState = skelStates[1].jointState[jointIndex];
      const size_t paramIndex = jointIndex * kParametersPerJoint;
      const Eigen::Vector3<T> posd = pos - jointState.translation();

      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          // Create a temporary gradient vector for frame 1
          gradient_jointParams_to_modelParams(
              boneWeight * velocityDiff.dot(jointState.getTranslationDerivative(d)) * wgt,
              paramIndex + d,
              this->parameterTransform_,
              grad1);
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          gradient_jointParams_to_modelParams(
              boneWeight * velocityDiff.dot(jointState.getRotationDerivative(d, posd)) * wgt,
              paramIndex + 3 + d,
              this->parameterTransform_,
              grad1);
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        gradient_jointParams_to_modelParams(
            boneWeight * velocityDiff.dot(jointState.getScaleDerivative(posd)) * wgt,
            paramIndex + 6,
            this->parameterTransform_,
            grad1);
      }
    }
  }

  return constraint.weight * velocityDiff.squaredNorm() * kVelocityWeight;
}

template <typename T>
double VertexSequenceErrorFunctionT<T>::getGradient(
    gsl::span<const ModelParametersT<T>> modelParameters,
    gsl::span<const SkeletonStateT<T>> skelStates,
    gsl::span<const MeshStateT<T>> /*meshStates*/,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_PROFILE_FUNCTION();
  MT_CHECK(modelParameters.size() == 2);
  MT_CHECK(skelStates.size() == 2);

  updateMeshes(modelParameters, skelStates);

  double error = 0.0;

  // Process constraints sequentially without multithreading
  for (const auto& constraint : constraints_) {
    error += calculateVelocityGradient(skelStates, constraint, gradient);
  }

  return this->weight_ * error;
}

template <typename T>
double VertexSequenceErrorFunctionT<T>::calculateVelocityJacobian(
    gsl::span<const SkeletonStateT<T>> skelStates,
    const VertexVelocityConstraintT<T>& constraint,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    Eigen::Index startRow) const {
  const Eigen::Vector3<T>& vertex0 = posedMesh0_->vertices[constraint.vertexIndex];
  const Eigen::Vector3<T>& vertex1 = posedMesh1_->vertices[constraint.vertexIndex];

  const Eigen::Vector3<T> actualVelocity = vertex1 - vertex0;
  const Eigen::Vector3<T> velocityDiff = actualVelocity - constraint.targetVelocity;

  const T wgt = std::sqrt(constraint.weight * kVelocityWeight * this->weight_);

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();

  // Jacobian with respect to frame 0 (negative contribution to velocity)
  {
    SkinningWeightIteratorT<T> skinningIter(
        character_, *restMesh_, skelStates[0], constraint.vertexIndex);

    while (!skinningIter.finished()) {
      size_t jointIndex = 0;
      T boneWeight;
      Eigen::Vector3<T> pos;
      std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

      MT_CHECK(jointIndex < this->skeleton_.joints.size());

      const auto& jointState = skelStates[0].jointState[jointIndex];
      const size_t paramIndex = jointIndex * kParametersPerJoint;
      const Eigen::Vector3<T> posd = pos - jointState.translation();

      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          jacobian_jointParams_to_modelParams<T>(
              -boneWeight * wgt * jointState.getTranslationDerivative(d),
              paramIndex + d,
              this->parameterTransform_,
              jacobian.block(startRow, 0, 3, nParam));
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          jacobian_jointParams_to_modelParams<T>(
              -boneWeight * wgt * jointState.getRotationDerivative(d, posd),
              paramIndex + d + 3,
              this->parameterTransform_,
              jacobian.block(startRow, 0, 3, nParam));
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        jacobian_jointParams_to_modelParams<T>(
            -boneWeight * wgt * jointState.getScaleDerivative(posd),
            paramIndex + 6,
            this->parameterTransform_,
            jacobian.block(startRow, 0, 3, nParam));
      }
    }
  }

  // Jacobian with respect to frame 1 (positive contribution to velocity)
  {
    SkinningWeightIteratorT<T> skinningIter(
        character_, *restMesh_, skelStates[1], constraint.vertexIndex);

    while (!skinningIter.finished()) {
      size_t jointIndex = 0;
      T boneWeight;
      Eigen::Vector3<T> pos;
      std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

      MT_CHECK(jointIndex < this->skeleton_.joints.size());

      const auto& jointState = skelStates[1].jointState[jointIndex];
      const size_t paramIndex = jointIndex * kParametersPerJoint;
      const Eigen::Vector3<T> posd = pos - jointState.translation();

      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          jacobian_jointParams_to_modelParams<T>(
              boneWeight * wgt * jointState.getTranslationDerivative(d),
              paramIndex + d,
              this->parameterTransform_,
              jacobian.block(startRow, nParam, 3, nParam));
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          jacobian_jointParams_to_modelParams<T>(
              boneWeight * wgt * jointState.getRotationDerivative(d, posd),
              paramIndex + d + 3,
              this->parameterTransform_,
              jacobian.block(startRow, nParam, 3, nParam));
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * jointState.getScaleDerivative(posd),
            paramIndex + 6,
            this->parameterTransform_,
            jacobian.block(startRow, nParam, 3, nParam));
      }
    }
  }

  residual.segment(startRow, 3) = velocityDiff * wgt;

  return wgt * wgt * velocityDiff.squaredNorm();
}

template <typename T>
double VertexSequenceErrorFunctionT<T>::getJacobian(
    gsl::span<const ModelParametersT<T>> modelParameters,
    gsl::span<const SkeletonStateT<T>> skelStates,
    gsl::span<const MeshStateT<T>> /*meshStates*/,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) const {
  MT_PROFILE_FUNCTION();
  MT_CHECK(modelParameters.size() == 2);
  MT_CHECK(skelStates.size() == 2);

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(jacobian.cols() == 2 * nParam);
  MT_CHECK(jacobian.rows() >= (Eigen::Index)(3 * constraints_.size()));
  MT_CHECK(residual.rows() >= (Eigen::Index)(3 * constraints_.size()));

  updateMeshes(modelParameters, skelStates);

  double error = 0.0;

  // Process constraints sequentially without multithreading
  for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
    error +=
        calculateVelocityJacobian(skelStates, constraints_[iCons], jacobian, residual, 3 * iCons);
  }

  usedRows = gsl::narrow_cast<int>(3 * constraints_.size());
  return error;
}

template <typename T>
size_t VertexSequenceErrorFunctionT<T>::getJacobianSize() const {
  return 3 * constraints_.size();
}

template class VertexSequenceErrorFunctionT<float>;
template class VertexSequenceErrorFunctionT<double>;

} // namespace momentum
