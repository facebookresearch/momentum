/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/vertex_error_function.h"
#include "momentum/character_solver/skinning_weight_iterator.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/mesh.h"

#include <dispenso/parallel_for.h>

#include <numeric>

namespace momentum {

std::string_view toString(VertexConstraintType type) {
  static const std::array<std::string_view, 4> strings = {
      "Position", "Plane", "Normal", "SymmetricNormal"};
  return strings[static_cast<size_t>(type)];
}

template <typename T>
VertexErrorFunctionT<T>::VertexErrorFunctionT(
    const Character& character_in,
    VertexConstraintType type,
    uint32_t maxThreads)
    : SkeletonErrorFunctionT<T>(character_in.skeleton, character_in.parameterTransform),
      character_(character_in),
      constraintType_(type),
      maxThreads_(maxThreads) {
  MT_CHECK(static_cast<bool>(character_in.mesh));
  MT_CHECK(static_cast<bool>(character_in.skinWeights));
  MT_THROW_IF(
      character_in.faceExpressionBlendShape && (type != VertexConstraintType::Position),
      "Constraint type {} not implemented yet for face. Only Position type is supported.",
      toString(type));
}

template <typename T>
VertexErrorFunctionT<T>::~VertexErrorFunctionT() = default;

template <typename T>
void VertexErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
void VertexErrorFunctionT<T>::addConstraint(
    int vertexIndex,
    T weight,
    const Eigen::Vector3<T>& targetPosition,
    const Eigen::Vector3<T>& targetNormal) {
  MT_CHECK(vertexIndex >= 0 && ((size_t)vertexIndex) < character_.mesh->vertices.size());
  constraints_.push_back(VertexConstraintT<T>{vertexIndex, weight, targetPosition, targetNormal});
}

template <typename T>
double VertexErrorFunctionT<T>::getError(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState) {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState.posedMesh_);
  MT_CHECK_NOTNULL(meshState.restMesh_);

  // loop over all constraints and calculate the error
  double error = 0.0;

  if (constraintType_ == VertexConstraintType::Position) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      const VertexConstraintT<T>& constr = constraints_[i];

      const Eigen::Vector3<T> diff =
          meshState.posedMesh_->vertices[constr.vertexIndex] - constr.targetPosition;
      error += constr.weight * diff.squaredNorm() * kPositionWeight;
    }
  } else {
    const auto [sourceNormalWeight, targetNormalWeight] = computeNormalWeights();
    for (size_t i = 0; i < constraints_.size(); ++i) {
      const VertexConstraintT<T>& constr = constraints_[i];

      const Eigen::Vector3<T> sourceNormal = meshState.posedMesh_->normals[constr.vertexIndex];
      Eigen::Vector3<T> targetNormal = constr.targetNormal;
      if (sourceNormal.dot(constr.targetNormal) < 0) {
        targetNormal *= -1;
      }
      const Eigen::Vector3<T> normal =
          sourceNormalWeight * sourceNormal + targetNormalWeight * targetNormal;

      // calculate point->plane distance and error
      const Eigen::Vector3<T> diff =
          meshState.posedMesh_->vertices[constr.vertexIndex] - constr.targetPosition;
      const T dist = normal.dot(diff);
      error += constr.weight * dist * dist * kPlaneWeight;
    }
  }

  // return error
  return error * this->weight_;
}

template <typename T>
void VertexErrorFunctionT<T>::calculateDWorldPos(
    const SkeletonStateT<T>& state,
    const VertexConstraintT<T>& constr,
    const Eigen::Vector3<T>& d_restPos,
    Eigen::Vector3<T>& d_worldPos) const {
  const auto& skinWeights = *character_.skinWeights;

  for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
    const auto w = skinWeights.weight(constr.vertexIndex, i);
    const auto parentBone = skinWeights.index(constr.vertexIndex, i);
    if (w > 0) {
      d_worldPos += w *
          (state.jointState[parentBone].transform.toLinear() *
           (character_.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
    }
  }
}

template <typename T>
double VertexErrorFunctionT<T>::calculatePositionGradient(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    const VertexConstraintT<T>& constr,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  const Eigen::Vector3<T> diff =
      meshState.posedMesh_->vertices[constr.vertexIndex] - constr.targetPosition;

  // calculate the difference between target and position and error
  const T wgt = constr.weight * 2.0f * kPositionWeight * this->weight_;

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *meshState.restMesh_, state, constr.vertexIndex);

  // IN handle derivatives wrt jointParameters
  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        // Gradient wrt translation:
        gradient_jointParams_to_modelParams(
            boneWeight * diff.dot(jointState.getTranslationDerivative(d)) * wgt,
            paramIndex + d,
            this->parameterTransform_,
            gradient);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // Gradient wrt rotation:
        gradient_jointParams_to_modelParams(
            boneWeight * diff.dot(jointState.getRotationDerivative(d, posd)) * wgt,
            paramIndex + 3 + d,
            this->parameterTransform_,
            gradient);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // Gradient wrt scale:
      gradient_jointParams_to_modelParams(
          boneWeight * diff.dot(jointState.getScaleDerivative(posd)) * wgt,
          paramIndex + 6,
          this->parameterTransform_,
          gradient);
    }
  } // OUT handle derivatives wrt jointParameters

  // IN handle derivatives wrt blend shape parameters
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
              .template block<3, 1>(3 * constr.vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, constr, d_restPos, d_worldPos);

      gradient[paramIdx] += wgt * diff.dot(d_worldPos);
    }
  }
  // OUT handle derivatives wrt blend shape parameters

  // IN handle derivatives wrt face expression blend shape parameters
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
              .template block<3, 1>(3 * constr.vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, constr, d_restPos, d_worldPos);

      gradient[paramIdx] += wgt * diff.dot(d_worldPos);
    }
  }
  // OUT handle derivatives wrt face expression blend shape parameters

  return constr.weight * diff.squaredNorm() * kPositionWeight;
}

template <typename T>
double VertexErrorFunctionT<T>::calculatePositionJacobian(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    const VertexConstraintT<T>& constr,
    Ref<Eigen::MatrixX<T>> jac,
    Ref<Eigen::VectorX<T>> res) const {
  // calculate the difference between target and position and error
  const Eigen::Vector3<T> diff =
      meshState.posedMesh_->vertices[constr.vertexIndex] - constr.targetPosition;

  const T wgt = std::sqrt(constr.weight * kPositionWeight * this->weight_);

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *meshState.restMesh_, state, constr.vertexIndex);

  // IN handle derivatives wrt jointParameters
  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * jointState.getTranslationDerivative(d),
            paramIndex + d,
            this->parameterTransform_,
            jac);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * jointState.getRotationDerivative(d, posd),
            paramIndex + d + 3,
            this->parameterTransform_,
            jac);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      jacobian_jointParams_to_modelParams<T>(
          boneWeight * wgt * jointState.getScaleDerivative(posd),
          paramIndex + 6,
          this->parameterTransform_,
          jac);
    }
  }
  // OUT handle derivatives wrt jointParameters

  // IN handle derivatives wrt blend shape parameters
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
              .template block<3, 1>(3 * constr.vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, constr, d_restPos, d_worldPos);

      jac.col(paramIdx) += wgt * d_worldPos;
    }
  }
  // OUT handle derivatives wrt blend shape parameters

  // IN handle derivatives wrt face expression blend shape parameters
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
              .template block<3, 1>(3 * constr.vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, constr, d_restPos, d_worldPos);

      jac.col(paramIdx) += wgt * d_worldPos;
    }
  }
  // OUT handle derivatives wrt face expression blend shape parameters

  res = diff * wgt;

  return wgt * wgt * diff.squaredNorm();
}

template <typename T>
double VertexErrorFunctionT<T>::calculateNormalGradient(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    const VertexConstraintT<T>& constr,
    const T sourceNormalWeight,
    const T targetNormalWeight,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  const auto& skinWeights = *character_.skinWeights;

  const Eigen::Vector3<T> sourceNormal = meshState.posedMesh_->normals[constr.vertexIndex];
  Eigen::Vector3<T> targetNormal = constr.targetNormal;
  if (sourceNormal.dot(constr.targetNormal) < 0) {
    targetNormal *= -1;
  }
  const Eigen::Vector3<T> normal =
      sourceNormalWeight * sourceNormal + targetNormalWeight * targetNormal;

  const Eigen::Vector3<T> diff =
      meshState.posedMesh_->vertices[constr.vertexIndex] - constr.targetPosition;
  const T dist = diff.dot(normal);

  // calculate the difference between target and position and error
  const T wgt = constr.weight * 2.0f * kPositionWeight * this->weight_;

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *meshState.restMesh_, state, constr.vertexIndex);

  // IN handle derivatives wrt jointParameters
  while (!skinningIter.finished()) {
    const auto [jointIndex, boneWeight, pos] = skinningIter.next();

    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        // Gradient wrt translation:
        gradient_jointParams_to_modelParams(
            boneWeight * wgt * dist * normal.dot(jointState.getTranslationDerivative(d)),
            paramIndex + d,
            this->parameterTransform_,
            gradient);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // Gradient wrt rotation:
        const Eigen::Vector3<T> cross =
            -sourceNormal.cross(constr.targetPosition - jointState.translation());
        const T diff_src = jointState.rotationAxis.col(d).dot(cross);
        const T diff_tgt = targetNormal.dot(jointState.getRotationDerivative(d, posd));

        gradient_jointParams_to_modelParams(
            boneWeight * wgt * dist *
                (sourceNormalWeight * diff_src + targetNormalWeight * diff_tgt),
            paramIndex + 3 + d,
            this->parameterTransform_,
            gradient);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // Gradient wrt scale:
      gradient_jointParams_to_modelParams(
          boneWeight * wgt * dist * normal.dot(jointState.getScaleDerivative(posd)),
          paramIndex + 6,
          this->parameterTransform_,
          gradient);
    }
  } // OUT handle derivatives wrt jointParameters

  // IN handle derivatives wrt blend shape parameters:
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
              .template block<3, 1>(3 * constr.vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();

      for (uint32_t jWeight = 0; jWeight < kMaxSkinJoints; ++jWeight) {
        const auto w = skinWeights.weight(constr.vertexIndex, jWeight);
        const auto parentBone = skinWeights.index(constr.vertexIndex, jWeight);
        if (w > 0) {
          d_worldPos += w *
              (state.jointState[parentBone].transform.toLinear() *
               (character_.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      gradient[paramIdx] += wgt * dist * normal.dot(d_worldPos);
    }
  }
  // OUT handle derivatives wrt blend shape parameters:

  return constr.weight * dist * dist * kPlaneWeight;
}

template <typename T>
double VertexErrorFunctionT<T>::calculateNormalJacobian(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    const VertexConstraintT<T>& constr,
    const T sourceNormalWeight,
    const T targetNormalWeight,
    Ref<Eigen::MatrixX<T>> jac,
    T& res) const {
  const auto& skinWeights = *character_.skinWeights;

  const Eigen::Vector3<T> sourceNormal = meshState.posedMesh_->normals[constr.vertexIndex];
  Eigen::Vector3<T> targetNormal = constr.targetNormal;
  if (sourceNormal.dot(constr.targetNormal) < 0) {
    targetNormal *= -1;
  }
  const Eigen::Vector3<T> normal =
      sourceNormalWeight * sourceNormal + targetNormalWeight * targetNormal;

  // calculate the difference between target and position and error
  const Eigen::Vector3<T> diff =
      meshState.posedMesh_->vertices[constr.vertexIndex] - constr.targetPosition;
  const T dist = diff.dot(normal);

  const T wgt = std::sqrt(constr.weight * kPlaneWeight * this->weight_);

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *meshState.restMesh_, state, constr.vertexIndex);

  // IN handle derivatives wrt jointParameters
  while (!skinningIter.finished()) {
    const auto [jointIndex, boneWeight, pos] = skinningIter.next();

    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        // Jacobian wrt translation:
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * normal.dot(jointState.getTranslationDerivative(d)),
            paramIndex + d,
            this->parameterTransform_,
            jac);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // Jacobian wrt rotation:
        const Eigen::Vector3<T> cross =
            -sourceNormal.cross(constr.targetPosition - jointState.translation());
        const T diff_src = jointState.rotationAxis.col(d).dot(cross);
        const T diff_tgt = targetNormal.dot(jointState.getRotationDerivative(d, posd));

        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * (sourceNormalWeight * diff_src + targetNormalWeight * diff_tgt),
            paramIndex + d + 3,
            this->parameterTransform_,
            jac);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // Jacobian wrt scale:
      jacobian_jointParams_to_modelParams<T>(
          boneWeight * wgt * normal.dot(jointState.getScaleDerivative(posd)),
          paramIndex + 6,
          this->parameterTransform_,
          jac);
    }
  }
  // OUT handle derivatives wrt jointParameters

  // IN handle derivatives wrt blend shape parameters:
  for (Eigen::Index iBlendShape = 0;
       iBlendShape < this->parameterTransform_.blendShapeParameters.size();
       ++iBlendShape) {
    const auto paramIdx = this->parameterTransform_.blendShapeParameters[iBlendShape];
    if (paramIdx < 0) {
      continue;
    }

    const Eigen::Vector3<T> d_restPos =
        this->character_.blendShape->getShapeVectors()
            .template block<3, 1>(3 * constr.vertexIndex, iBlendShape, 3, 1)
            .template cast<T>();
    Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();

    for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
      const auto w = skinWeights.weight(constr.vertexIndex, i);
      const auto parentBone = skinWeights.index(constr.vertexIndex, i);
      if (w > 0) {
        d_worldPos += w *
            (state.jointState[parentBone].transform.toLinear() *
             (character_.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
      }
    }

    jac(0, paramIdx) += wgt * d_worldPos.dot(normal);
  }
  // OUT handle derivatives wrt blend shape parameters:

  res = dist * wgt;

  return wgt * wgt * dist * dist;
}

template <typename T>
std::pair<T, T> VertexErrorFunctionT<T>::computeNormalWeights() const {
  switch (constraintType_) {
    case VertexConstraintType::Plane:
      return {T(0), T(1)};
    case VertexConstraintType::Normal:
      return {T(1), T(0)};
    case VertexConstraintType::SymmetricNormal:
    default:
      return {T(0.5), T(0.5)};
  }
}

template <typename T>
double VertexErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_CHECK_NOTNULL(meshState.posedMesh_);
  MT_CHECK_NOTNULL(meshState.restMesh_);

  double error = 0;
  std::vector<std::tuple<double, VectorX<T>>> errorGradThread;

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;

  if (constraintType_ == VertexConstraintType::Position) {
    dispenso::parallel_for(
        errorGradThread,
        [&]() -> std::tuple<double, VectorX<T>> {
          return {0.0, VectorX<T>::Zero(modelParameters.size())};
        },
        0,
        constraints_.size(),
        [&](std::tuple<double, VectorX<T>>& errorGradLocal, const size_t iCons) {
          double& errorLocal = std::get<0>(errorGradLocal);
          auto& gradLocal = std::get<1>(errorGradLocal);
          errorLocal += calculatePositionGradient(
              modelParameters, state, meshState, constraints_[iCons], gradLocal);
        },
        dispensoOptions);
  } else {
    T sourceNormalWeight;
    T targetNormalWeight;
    std::tie(sourceNormalWeight, targetNormalWeight) = computeNormalWeights();

    dispenso::parallel_for(
        errorGradThread,
        [&]() -> std::tuple<double, VectorX<T>> {
          return {0.0, VectorX<T>::Zero(modelParameters.size())};
        },
        0,
        constraints_.size(),
        [&](std::tuple<double, VectorX<T>>& errorGradLocal, const size_t iCons) {
          double& errorLocal = std::get<0>(errorGradLocal);
          auto& gradLocal = std::get<1>(errorGradLocal);
          errorLocal += calculateNormalGradient(
              modelParameters,
              state,
              meshState,
              constraints_[iCons],
              sourceNormalWeight,
              targetNormalWeight,
              gradLocal);
        },
        dispensoOptions);
  }

  if (!errorGradThread.empty()) {
    errorGradThread[0] = std::accumulate(
        errorGradThread.begin() + 1,
        errorGradThread.end(),
        errorGradThread[0],
        [](const auto& a, const auto& b) -> std::tuple<double, VectorX<T>> {
          return {std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b)};
        });

    // finalize the gradient
    gradient += std::get<1>(errorGradThread[0]);
    error = std::get<0>(errorGradThread[0]);
  }

  return this->weight_ * error;
}

template <typename T>
double VertexErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_CHECK(
      jacobian.cols() == static_cast<Eigen::Index>(this->parameterTransform_.transform.cols()));
  MT_CHECK(jacobian.rows() >= (Eigen::Index)(1 * constraints_.size()));
  MT_CHECK(residual.rows() >= (Eigen::Index)(1 * constraints_.size()));

  MT_CHECK_NOTNULL(meshState.posedMesh_);
  MT_CHECK_NOTNULL(meshState.restMesh_);

  double error = 0;
  std::vector<double> errorThread;

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;

  if (constraintType_ == VertexConstraintType::Position) {
    MT_PROFILE_EVENT("VertexErrorFunction - position jacobians");

    dispenso::parallel_for(
        errorThread,
        [&]() -> double { return 0.0; },
        0,
        constraints_.size(),
        [&](double& errorLocal, const size_t iCons) {
          errorLocal += calculatePositionJacobian(
              modelParameters,
              state,
              meshState,
              constraints_[iCons],
              jacobian.block(3 * iCons, 0, 3, modelParameters.size()),
              residual.middleRows(3 * iCons, 3));
        },
        dispensoOptions);
    usedRows = 3 * constraints_.size();
  } else {
    MT_PROFILE_EVENT("VertexErrorFunction - normal jacobians");
    T sourceNormalWeight;
    T targetNormalWeight;
    std::tie(sourceNormalWeight, targetNormalWeight) = computeNormalWeights();

    dispenso::parallel_for(
        errorThread,
        [&]() -> double { return 0.0; },
        0,
        constraints_.size(),
        [&](double& errorLocal, const size_t iCons) {
          errorLocal += calculateNormalJacobian(
              modelParameters,
              state,
              meshState,
              constraints_[iCons],
              sourceNormalWeight,
              targetNormalWeight,
              jacobian.block(iCons, 0, 1, modelParameters.size()),
              residual(iCons));
        },
        dispensoOptions);

    usedRows = constraints_.size();
  }

  if (!errorThread.empty()) {
    error = std::accumulate(errorThread.begin() + 1, errorThread.end(), errorThread[0]);
  }

  return error;
}

template <typename T>
size_t VertexErrorFunctionT<T>::getJacobianSize() const {
  switch (constraintType_) {
    case VertexConstraintType::Position:
      return 3 * constraints_.size();
    case VertexConstraintType::Normal:
    case VertexConstraintType::Plane:
    case VertexConstraintType::SymmetricNormal:
    default:
      return constraints_.size();
  }
}

template class VertexErrorFunctionT<float>;
template class VertexErrorFunctionT<double>;

} // namespace momentum
