/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/height_error_function.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
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

namespace momentum {

namespace {

/// Apply parameter transform to convert joint parameter gradient to model parameter gradient,
/// but only update model parameters that are active in the provided ParameterSet.
template <typename T>
void gradient_jointParams_to_modelParams_filtered(
    const T& grad_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform,
    const ParameterSet& activeModelParams,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    const auto modelParamIndex = parameterTransform.transform.innerIndexPtr()[index];
    if (activeModelParams[modelParamIndex]) {
      gradient[modelParamIndex] +=
          grad_jointParams * parameterTransform.transform.valuePtr()[index];
    }
  }
}

/// Apply parameter transform to convert joint parameter jacobian to model parameter jacobian,
/// but only update model parameters that are active in the provided ParameterSet.
template <typename T, typename Derived>
void jacobian_jointParams_to_modelParams_filtered(
    const T& jacobian_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform,
    const ParameterSet& activeModelParams,
    Eigen::MatrixBase<Derived>& jacobian) {
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    const auto modelParamIndex = parameterTransform.transform.innerIndexPtr()[index];
    if (activeModelParams[modelParamIndex]) {
      jacobian(0, modelParamIndex) +=
          jacobian_jointParams * parameterTransform.transform.valuePtr()[index];
    }
  }
}

} // namespace

template <typename T>
HeightErrorFunctionT<T>::HeightErrorFunctionT(
    const Character& character,
    T targetHeight,
    const Eigen::Vector3<T>& upDirection,
    size_t k)
    : SkeletonErrorFunctionT<T>(character.skeleton, character.parameterTransform),
      character_(character),
      targetHeight_(targetHeight),
      upDirection_(upDirection.normalized()),
      k_(k),
      skeletonState_(),
      meshState_() {
  MT_CHECK(k > 0, "k must be greater than 0, got {}", k);
  MT_CHECK(character.mesh != nullptr, "HeightErrorFunction requires a character with a mesh");

  // Set up active parameters (blend shapes, face expressions, and scale only)
  activeModelParams_ = character.parameterTransform.getBlendShapeParameters() |
      character.parameterTransform.getFaceExpressionParameters() |
      character.parameterTransform.getScalingParameters();
}

template <typename T>
HeightErrorFunctionT<T>::~HeightErrorFunctionT() = default;

template <typename T>
void HeightErrorFunctionT<T>::setTargetHeight(T height) {
  targetHeight_ = height;
}

template <typename T>
void HeightErrorFunctionT<T>::setUpDirection(const Eigen::Vector3<T>& upDirection) {
  upDirection_ = upDirection.normalized();
}

template <typename T>
ModelParametersT<T> HeightErrorFunctionT<T>::applyActiveParameters(
    const ModelParametersT<T>& modelParameters) const {
  MT_PROFILE_FUNCTION();

  ModelParametersT<T> result = modelParameters;

  // Zero out all inactive parameters
  for (Eigen::Index i = 0; i < result.size(); ++i) {
    if (!activeModelParams_.test(i)) {
      result[i] = T(0);
    }
  }

  return result;
}

template <typename T>
typename HeightErrorFunctionT<T>::HeightResult HeightErrorFunctionT<T>::calculateHeight() const {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState_.posedMesh_);

  if (meshState_.posedMesh_->vertices.empty()) {
    return {T(0), {}, {}, {}, {}};
  }

  // Use priority queues to track k lowest and k highest projections
  // For min heap (highest projections): use std::greater
  // For max heap (lowest projections): use std::less
  using ProjectionPair = std::pair<T, size_t>; // (projection, vertex_index)

  // Max heap for k lowest projections (top element is largest of the k lowest)
  std::priority_queue<ProjectionPair, std::vector<ProjectionPair>, std::less<>> minQueue;

  // Min heap for k highest projections (top element is smallest of the k highest)
  std::priority_queue<ProjectionPair, std::vector<ProjectionPair>, std::greater<>> maxQueue;

  // Populate the priority queues
  for (size_t i = 0; i < meshState_.posedMesh_->vertices.size(); ++i) {
    const T projection = upDirection_.dot(meshState_.posedMesh_->vertices[i]);

    // For k lowest projections
    if (minQueue.size() < k_) {
      minQueue.push({projection, i});
    } else if (projection < minQueue.top().first) {
      minQueue.pop();
      minQueue.push({projection, i});
    }

    // For k highest projections
    if (maxQueue.size() < k_) {
      maxQueue.push({projection, i});
    } else if (projection > maxQueue.top().first) {
      maxQueue.pop();
      maxQueue.push({projection, i});
    }
  }

  // Extract results from queues
  HeightResult result;
  result.minVertexIndices.reserve(k_);
  result.minVertexWeights.reserve(k_);
  result.maxVertexIndices.reserve(k_);
  result.maxVertexWeights.reserve(k_);

  T minProjectionSum = T(0);
  while (!minQueue.empty()) {
    const auto& pair = minQueue.top();
    result.minVertexIndices.push_back(pair.second);
    minProjectionSum += pair.first;
    minQueue.pop();
  }

  T maxProjectionSum = T(0);
  while (!maxQueue.empty()) {
    const auto& pair = maxQueue.top();
    result.maxVertexIndices.push_back(pair.second);
    maxProjectionSum += pair.first;
    maxQueue.pop();
  }

  // Compute uniform weights (1/k for each vertex)
  const T weight = T(1) / T(k_);
  result.minVertexWeights.resize(k_, weight);
  result.maxVertexWeights.resize(k_, weight);

  // Average height is the difference of average projections
  const T avgMinProjection = minProjectionSum / T(k_);
  const T avgMaxProjection = maxProjectionSum / T(k_);
  result.height = avgMaxProjection - avgMinProjection;

  return result;
}

template <typename T>
double HeightErrorFunctionT<T>::getError(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& /* state */,
    const MeshStateT<T>& /* meshState */) {
  MT_PROFILE_FUNCTION();

  // Create modified parameters with inactive parameters zeroed out
  const ModelParametersT<T> activeOnlyParams = applyActiveParameters(modelParameters);

  // Update skeleton state with active parameters only
  skeletonState_.set(
      this->parameterTransform_.template cast<T>().apply(activeOnlyParams), this->skeleton_, false);

  // Update internal mesh state using the modified parameters
  meshState_.update(activeOnlyParams, skeletonState_, character_);

  const auto heightResult = calculateHeight();
  const T heightDiff = heightResult.height - targetHeight_;

  return this->weight_ * heightDiff * heightDiff;
}

template <typename T>
double HeightErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& /* state */,
    const MeshStateT<T>& /* meshState */,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  // Create modified parameters with inactive parameters zeroed out
  const ModelParametersT<T> activeOnlyParams = applyActiveParameters(modelParameters);

  // Update skeleton state with active parameters only
  skeletonState_.set(
      this->parameterTransform_.template cast<T>().apply(activeOnlyParams), this->skeleton_, true);

  // Update internal mesh state
  meshState_.update(activeOnlyParams, skeletonState_, character_);

  const auto heightResult = calculateHeight();
  const T heightDiff = heightResult.height - targetHeight_;

  const T wgt = T(2) * heightDiff * this->weight_;

  // Average gradients over k highest points
  for (size_t i = 0; i < heightResult.maxVertexIndices.size(); ++i) {
    calculateVertexGradient(
        heightResult.maxVertexIndices[i],
        wgt * heightResult.maxVertexWeights[i] * upDirection_,
        gradient);
  }

  // Average gradients over k lowest points
  for (size_t i = 0; i < heightResult.minVertexIndices.size(); ++i) {
    calculateVertexGradient(
        heightResult.minVertexIndices[i],
        -wgt * heightResult.minVertexWeights[i] * upDirection_,
        gradient);
  }

  return this->weight_ * heightDiff * heightDiff;
}

template <typename T>
double HeightErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& /* state */,
    const MeshStateT<T>& /* meshState */,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      jacobian.cols() == static_cast<Eigen::Index>(this->parameterTransform_.transform.cols()));
  MT_CHECK(jacobian.rows() >= 1);
  MT_CHECK(residual.rows() >= 1);

  // Create modified parameters with inactive parameters zeroed out
  const ModelParametersT<T> activeOnlyParams = applyActiveParameters(modelParameters);

  // Update skeleton state with active parameters only
  skeletonState_.set(
      this->parameterTransform_.template cast<T>().apply(activeOnlyParams), this->skeleton_, true);

  // Update internal mesh state
  meshState_.update(activeOnlyParams, skeletonState_, character_);

  const auto heightResult = calculateHeight();
  const T heightDiff = heightResult.height - targetHeight_;

  const T wgt = std::sqrt(this->weight_);
  residual(0) = wgt * heightDiff;

  jacobian.row(0).setZero();
  auto jac_row = jacobian.row(0);

  // Average jacobians over k highest points
  for (size_t i = 0; i < heightResult.maxVertexIndices.size(); ++i) {
    calculateVertexJacobian(
        heightResult.maxVertexIndices[i],
        wgt * heightResult.maxVertexWeights[i] * upDirection_,
        jac_row);
  }

  // Average jacobians over k lowest points
  for (size_t i = 0; i < heightResult.minVertexIndices.size(); ++i) {
    calculateVertexJacobian(
        heightResult.minVertexIndices[i],
        -wgt * heightResult.minVertexWeights[i] * upDirection_,
        jac_row);
  }

  usedRows = 1;
  return wgt * wgt * heightDiff * heightDiff;
}

template <typename T>
size_t HeightErrorFunctionT<T>::getJacobianSize() const {
  return 1;
}

template <typename T>
template <typename Derived>
void HeightErrorFunctionT<T>::calculateVertexJacobian(
    size_t vertexIndex,
    const Eigen::Vector3<T>& jacobianDirection,
    Eigen::MatrixBase<Derived>& jacobian) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState_.restMesh_);

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *meshState_.restMesh_, skeletonState_, vertexIndex);

  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = skeletonState_.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // Calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        // Jacobian wrt translation:
        jacobian_jointParams_to_modelParams_filtered<T>(
            boneWeight * jacobianDirection.dot(jointState.getTranslationDerivative(d)),
            paramIndex + d,
            this->parameterTransform_,
            activeModelParams_,
            jacobian);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // Jacobian wrt rotation:
        jacobian_jointParams_to_modelParams_filtered<T>(
            boneWeight * jacobianDirection.dot(jointState.getRotationDerivative(d, posd)),
            paramIndex + 3 + d,
            this->parameterTransform_,
            activeModelParams_,
            jacobian);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // Jacobian wrt scale:
      jacobian_jointParams_to_modelParams_filtered<T>(
          boneWeight * jacobianDirection.dot(jointState.getScaleDerivative(posd)),
          paramIndex + 6,
          this->parameterTransform_,
          activeModelParams_,
          jacobian);
    }
  }

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
      calculateDWorldPos(vertexIndex, d_restPos, d_worldPos);

      jacobian(0, paramIdx) += jacobianDirection.dot(d_worldPos);
    }
  }

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
      calculateDWorldPos(vertexIndex, d_restPos, d_worldPos);

      jacobian(0, paramIdx) += jacobianDirection.dot(d_worldPos);
    }
  }
}

template <typename T>
void HeightErrorFunctionT<T>::calculateVertexGradient(
    size_t vertexIndex,
    const Eigen::Vector3<T>& gradientDirection,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState_.restMesh_);

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *meshState_.restMesh_, skeletonState_, vertexIndex);

  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = skeletonState_.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // Calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        // Gradient wrt translation:
        gradient_jointParams_to_modelParams_filtered<T>(
            boneWeight * gradientDirection.dot(jointState.getTranslationDerivative(d)),
            paramIndex + d,
            this->parameterTransform_,
            activeModelParams_,
            gradient);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // Gradient wrt rotation:
        gradient_jointParams_to_modelParams_filtered<T>(
            boneWeight * gradientDirection.dot(jointState.getRotationDerivative(d, posd)),
            paramIndex + 3 + d,
            this->parameterTransform_,
            activeModelParams_,
            gradient);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // Gradient wrt scale:
      gradient_jointParams_to_modelParams_filtered<T>(
          boneWeight * gradientDirection.dot(jointState.getScaleDerivative(posd)),
          paramIndex + 6,
          this->parameterTransform_,
          activeModelParams_,
          gradient);
    }
  }

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
      calculateDWorldPos(vertexIndex, d_restPos, d_worldPos);

      gradient[paramIdx] += gradientDirection.dot(d_worldPos);
    }
  }

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
      calculateDWorldPos(vertexIndex, d_restPos, d_worldPos);

      gradient[paramIdx] += gradientDirection.dot(d_worldPos);
    }
  }
}

template <typename T>
void HeightErrorFunctionT<T>::calculateDWorldPos(
    size_t vertexIndex,
    const Eigen::Vector3<T>& d_restPos,
    Eigen::Vector3<T>& d_worldPos) const {
  const auto& skinWeights = *character_.skinWeights;

  for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
    const auto w = skinWeights.weight(vertexIndex, i);
    const auto parentBone = skinWeights.index(vertexIndex, i);
    if (w > 0) {
      d_worldPos += w *
          (skeletonState_.jointState[parentBone].transform.toLinear() *
           (character_.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
    }
  }
}

template class HeightErrorFunctionT<float>;
template class HeightErrorFunctionT<double>;

} // namespace momentum
