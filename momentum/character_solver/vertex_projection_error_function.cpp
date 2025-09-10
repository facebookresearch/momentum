/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/vertex_projection_error_function.h"
#include "momentum/character_solver/skinning_weight_iterator.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/mesh.h"
#include "momentum/math/utility.h"

#include <dispenso/parallel_for.h>

#include <numeric>

namespace momentum {

template <typename T>
VertexProjectionErrorFunctionT<T>::VertexProjectionErrorFunctionT(
    const Character& character_in,
    size_t maxThreads)
    : SkeletonErrorFunctionT<T>(character_in.skeleton, character_in.parameterTransform),
      character_(character_in),
      maxThreads_(maxThreads) {
  MT_CHECK(static_cast<bool>(character_in.mesh));
  MT_CHECK(static_cast<bool>(character_in.skinWeights));
  this->neutralMesh_ = std::make_unique<MeshT<T>>(character_in.mesh->template cast<T>());
  this->restMesh_ = std::make_unique<MeshT<T>>(character_in.mesh->template cast<T>());
  this->posedMesh_ = std::make_unique<MeshT<T>>(character_in.mesh->template cast<T>());
}

template <typename T>
VertexProjectionErrorFunctionT<T>::~VertexProjectionErrorFunctionT() = default;

template <typename T>
void VertexProjectionErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
void VertexProjectionErrorFunctionT<T>::addConstraint(
    int vertexIndex,
    T weight,
    const Eigen::Vector2<T>& targetPosition,
    const Eigen::Matrix<T, 3, 4>& projection) {
  MT_CHECK(vertexIndex >= 0 && ((size_t)vertexIndex) < character_.mesh->vertices.size());
  constraints_.push_back(
      VertexProjectionConstraintT<T>{vertexIndex, weight, targetPosition, projection});
}

template <typename T>
double VertexProjectionErrorFunctionT<T>::getError(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state) {
  MT_PROFILE_FUNCTION();

  updateMeshes(modelParameters, state);

  // loop over all constraints and calculate the error
  double error = 0.0;

  for (size_t i = 0; i < constraints_.size(); ++i) {
    const VertexProjectionConstraintT<T>& constr = constraints_[i];

    const Eigen::Vector3<T> p_projected =
        constr.projection * this->posedMesh_->vertices[constr.vertexIndex].homogeneous();

    // Behind camera:
    if (p_projected.z() < _nearClip) {
      continue;
    }

    const Eigen::Vector2<T> diff =
        p_projected.hnormalized().template head<2>() - constr.targetPosition;
    error += constr.weight * diff.squaredNorm();
  }

  // return error
  return error * this->weight_;
}

template <typename T>
void VertexProjectionErrorFunctionT<T>::calculateDWorldPos(
    const SkeletonStateT<T>& state,
    const VertexProjectionConstraintT<T>& constr,
    const Eigen::Vector3<T>& d_restPos,
    Eigen::Vector3<T>& d_worldPos) const {
  const auto& skinWeights = *character_.skinWeights;

  for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
    const auto w = skinWeights.weight(constr.vertexIndex, i);
    const auto parentBone = skinWeights.index(constr.vertexIndex, i);
    if (w > 0) {
      d_worldPos += w *
          (state.jointState[parentBone].transform.toLinear() *
           (character_.inverseBindPose[parentBone].toLinear().template cast<T>() * d_restPos));
    }
  }
}

template <typename T>
double VertexProjectionErrorFunctionT<T>::calculateGradient(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const VertexProjectionConstraintT<T>& constr,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  const Eigen::Vector3<T>& p_world_cm = this->posedMesh_->vertices[constr.vertexIndex];
  const Eigen::Vector3<T> p_projected_cm = constr.projection * p_world_cm.homogeneous();

  // Behind camera:
  if (p_projected_cm.z() < _nearClip) {
    return 0.0;
  }

  const Eigen::Vector2<T> p_res = p_projected_cm.hnormalized() - constr.targetPosition;
  T error = constr.weight * this->weight_ * p_res.squaredNorm();

  // calculate the variables needed to calculate the gradient
  const T wgt = constr.weight * 2.0f * this->weight_;
  const T z = p_projected_cm(2);
  const T z_sqr = sqr(z);
  const T x_zz = p_projected_cm(0) / z_sqr;
  const T y_zz = p_projected_cm(1) / z_sqr;

  auto gradientFunc = [&](const Eigen::Vector3<T>& d_p_world_cm) {
    const Eigen::Vector3<T> d_p_projected =
        constr.projection.template topLeftCorner<3, 3>() * d_p_world_cm;
    const T& dx = d_p_projected(0);
    const T& dy = d_p_projected(1);
    const T& dz = d_p_projected(2);

    const Eigen::Vector2<T> d_p_res(dx / z - x_zz * dz, dy / z - y_zz * dz);
    const T gradFull = d_p_res.dot(p_res);
    return gradFull;
  };

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *this->restMesh_, state, constr.vertexIndex);

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
            boneWeight * gradientFunc(jointState.getTranslationDerivative(d)) * wgt,
            paramIndex + d,
            this->parameterTransform_,
            gradient);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // Gradient wrt rotation:
        gradient_jointParams_to_modelParams(
            boneWeight * gradientFunc(jointState.getRotationDerivative(d, posd)) * wgt,
            paramIndex + 3 + d,
            this->parameterTransform_,
            gradient);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // Gradient wrt scale:
      gradient_jointParams_to_modelParams(
          boneWeight * gradientFunc(jointState.getScaleDerivative(posd)) * wgt,
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

      gradient[paramIdx] += wgt * gradientFunc(d_worldPos);
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

      gradient[paramIdx] += wgt * gradientFunc(d_worldPos);
    }
  }
  // OUT handle derivatives wrt face expression blend shape parameters

  return error;
}

template <typename T>
double VertexProjectionErrorFunctionT<T>::calculateJacobian(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const VertexProjectionConstraintT<T>& constr,
    Ref<Eigen::MatrixX<T>> jac,
    Ref<Eigen::VectorX<T>> res) const {
  const Eigen::Vector3<T>& p_world_cm = this->posedMesh_->vertices[constr.vertexIndex];
  const Eigen::Vector3<T> p_projected_cm = constr.projection * p_world_cm.homogeneous();

  // Behind camera:
  if (p_projected_cm.z() < _nearClip) {
    return 0.0;
  }

  const Eigen::Vector2<T> p_res = p_projected_cm.hnormalized() - constr.targetPosition;
  T error = constr.weight * this->weight_ * p_res.squaredNorm();

  // calculate the variables needed to calculate the gradient
  const T wgt = std::sqrt(constr.weight * this->weight_);
  const T z = p_projected_cm(2);
  const T z_sqr = sqr(z);
  const T x_zz = p_projected_cm(0) / z_sqr;
  const T y_zz = p_projected_cm(1) / z_sqr;

  auto jacobianFunc = [&](const Eigen::Vector3<T>& d_p_world_cm) {
    const Eigen::Vector3<T> d_p_projected =
        constr.projection.template topLeftCorner<3, 3>() * d_p_world_cm;
    const T& dx = d_p_projected(0);
    const T& dy = d_p_projected(1);
    const T& dz = d_p_projected(2);

    Eigen::Vector2<T> d_p_res(dx / z - x_zz * dz, dy / z - y_zz * dz);
    return d_p_res;
  };

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *this->restMesh_, state, constr.vertexIndex);

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
            boneWeight * wgt * jacobianFunc(jointState.getTranslationDerivative(d)),
            paramIndex + d,
            this->parameterTransform_,
            jac);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * jacobianFunc(jointState.getRotationDerivative(d, posd)),
            paramIndex + d + 3,
            this->parameterTransform_,
            jac);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      jacobian_jointParams_to_modelParams<T>(
          boneWeight * wgt * jacobianFunc(jointState.getScaleDerivative(posd)),
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

      jac.col(paramIdx) += wgt * jacobianFunc(d_worldPos);
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

      jac.col(paramIdx) += wgt * jacobianFunc(d_worldPos);
    }
  }
  // OUT handle derivatives wrt face expression blend shape parameters

  res = p_res * wgt;

  return error;
}

template <typename T>
double VertexProjectionErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  updateMeshes(modelParameters, state);

  double error = 0;
  std::vector<std::tuple<double, VectorX<T>>> errorGradThread;

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;

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
        errorLocal += calculateGradient(modelParameters, state, constraints_[iCons], gradLocal);
      },
      dispensoOptions);

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

  return error;
}

template <typename T>
double VertexProjectionErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_CHECK(
      jacobian.cols() == static_cast<Eigen::Index>(this->parameterTransform_.transform.cols()));
  MT_CHECK(jacobian.rows() >= (Eigen::Index)(1 * constraints_.size()));
  MT_CHECK(residual.rows() >= (Eigen::Index)(1 * constraints_.size()));

  updateMeshes(modelParameters, state);

  double error = 0;
  std::vector<double> errorThread;

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads_;

  dispenso::parallel_for(
      errorThread,
      [&]() -> double { return 0.0; },
      0,
      constraints_.size(),
      [&](double& errorLocal, const size_t iCons) {
        errorLocal += calculateJacobian(
            modelParameters,
            state,
            constraints_[iCons],
            jacobian.block(2 * iCons, 0, 2, modelParameters.size()),
            residual.middleRows(2 * iCons, 2));
      },
      dispensoOptions);
  usedRows = 2 * constraints_.size();

  if (!errorThread.empty()) {
    error = std::accumulate(errorThread.begin() + 1, errorThread.end(), errorThread[0]);
  }

  return error;
}

template <typename T>
size_t VertexProjectionErrorFunctionT<T>::getJacobianSize() const {
  return 2 * constraints_.size();
}

template <typename T>
void VertexProjectionErrorFunctionT<T>::updateMeshes(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state) {
  MT_PROFILE_FUNCTION();

  bool doUpdateNormals = false;
  if (this->character_.blendShape) {
    const BlendWeightsT<T> blendWeights =
        extractBlendWeights(this->parameterTransform_, modelParameters);
    this->character_.blendShape->computeShape(blendWeights, this->restMesh_->vertices);
    doUpdateNormals = true;
  }
  if (this->character_.faceExpressionBlendShape) {
    if (!this->character_.blendShape) {
      // Set restMesh back to neutral, removing potential previous expressions.
      // Note that if the character comes with (shape) blendShape, the previous if block already
      // takes care of this step.
      Eigen::Map<Eigen::VectorX<T>> outputVec(
          &this->restMesh_->vertices[0][0], this->restMesh_->vertices.size() * 3);
      const Eigen::Map<Eigen::VectorX<T>> baseVec(
          &this->neutralMesh_->vertices[0][0], this->neutralMesh_->vertices.size() * 3);
      outputVec = baseVec.template cast<T>();
    }
    const BlendWeightsT<T> faceExpressionBlendWeights =
        extractFaceExpressionBlendWeights(this->parameterTransform_, modelParameters);
    this->character_.faceExpressionBlendShape->applyDeltas(
        faceExpressionBlendWeights, this->restMesh_->vertices);
    doUpdateNormals = true;
  }
  if (doUpdateNormals) {
    this->restMesh_->updateNormals();
  }

  // Manually cast the transform list since TransformT doesn't have a Scalar typedef
  TransformListT<T> castedInverseBindPose;
  castedInverseBindPose.reserve(character_.inverseBindPose.size());
  for (const auto& transform : character_.inverseBindPose) {
    castedInverseBindPose.push_back(transform.template cast<T>());
  }

  applySSD(
      castedInverseBindPose,
      *this->character_.skinWeights,
      *this->restMesh_,
      state,
      *this->posedMesh_);
  // TODO should we call updateNormals() here too or trust the ones from skinning?
}

template class VertexProjectionErrorFunctionT<float>;
template class VertexProjectionErrorFunctionT<double>;

} // namespace momentum
