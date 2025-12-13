/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/point_triangle_vertex_error_function.h"
#include "momentum/character_solver/skinning_weight_iterator.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/mesh.h"

#include <iostream>
#include <memory>
#include <numeric>

namespace momentum {

template <typename T>
PointTriangleVertexErrorFunctionT<T>::PointTriangleVertexErrorFunctionT(
    const Character& character_in,
    VertexConstraintType type)
    : SkeletonErrorFunctionT<T>(character_in.skeleton, character_in.parameterTransform),
      character_(character_in),
      constraintType_(type) {
  MT_CHECK(static_cast<bool>(character_in.mesh));
  MT_CHECK(static_cast<bool>(character_in.skinWeights));
  MT_THROW_IF(
      character_in.faceExpressionBlendShape && (type != VertexConstraintType::Position),
      "Constraint type {} not implemented yet for face. Only Position type is supported.",
      toString(type));
}

template <typename T>
size_t PointTriangleVertexErrorFunctionT<T>::getNumVertices() const {
  return this->character_.mesh->vertices.size();
}

template <typename T>
PointTriangleVertexErrorFunctionT<T>::~PointTriangleVertexErrorFunctionT() = default;

template <typename T>
void PointTriangleVertexErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
void PointTriangleVertexErrorFunctionT<T>::addConstraint(
    int vertexIndex,
    const Eigen::Vector3i& triangleIndices,
    const Eigen::Vector3<T>& triangleBaryCoords,
    float depth,
    T weight) {
  MT_CHECK(vertexIndex >= 0 && ((size_t)vertexIndex) < character_.mesh->vertices.size());
  constraints_.push_back(
      PointTriangleVertexConstraintT<T>{
          vertexIndex, triangleIndices, triangleBaryCoords, depth, weight});
}

template <typename T>
Eigen::Vector3<T> computeTargetBaryPosition(
    const MeshT<T>& mesh,
    const PointTriangleVertexConstraintT<T>& c) {
  return c.tgtTriangleBaryCoords[0] * mesh.vertices[c.tgtTriangleIndices[0]] +
      c.tgtTriangleBaryCoords[1] * mesh.vertices[c.tgtTriangleIndices[1]] +
      c.tgtTriangleBaryCoords[2] * mesh.vertices[c.tgtTriangleIndices[2]];
}

template <typename T>
Eigen::Vector3<T> computeTargetTriangleNormal(
    const MeshT<T>& mesh,
    const PointTriangleVertexConstraintT<T>& c) {
  return (mesh.vertices[c.tgtTriangleIndices[1]] - mesh.vertices[c.tgtTriangleIndices[0]])
      .cross(mesh.vertices[c.tgtTriangleIndices[2]] - mesh.vertices[c.tgtTriangleIndices[0]])
      .normalized();
}

template <typename T>
Eigen::Vector3<T> computeTargetPosition(
    const MeshT<T>& mesh,
    const PointTriangleVertexConstraintT<T>& c) {
  return computeTargetBaryPosition(mesh, c) + c.depth * computeTargetTriangleNormal(mesh, c);
}

template <typename T>
double PointTriangleVertexErrorFunctionT<T>::getError(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState) {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState.posedMesh_);

  // loop over all constraints and calculate the error
  double error = 0.0;

  if (constraintType_ == VertexConstraintType::Position) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      const PointTriangleVertexConstraintT<T>& constr = constraints_[i];

      const Eigen::Vector3<T> srcPoint = meshState.posedMesh_->vertices[constr.srcVertexIndex];
      const Eigen::Vector3<T> tgtPoint = computeTargetPosition(*meshState.posedMesh_, constr);
      const Eigen::Vector3<T> diff = srcPoint - tgtPoint;
      error += constr.weight * diff.squaredNorm() * kPositionWeight;
    }
  } else {
    const auto [sourceNormalWeight, targetNormalWeight] = computeNormalWeights();
    for (size_t i = 0; i < constraints_.size(); ++i) {
      const PointTriangleVertexConstraintT<T>& constr = constraints_[i];

      const Eigen::Vector3<T> srcPoint = meshState.posedMesh_->vertices[constr.srcVertexIndex];

      const Eigen::Vector3<T> sourceNormal = meshState.posedMesh_->normals[constr.srcVertexIndex];
      Eigen::Vector3<T> targetNormal = computeTargetTriangleNormal(*meshState.posedMesh_, constr);

      const Eigen::Vector3<T> tgtPoint =
          computeTargetBaryPosition(*meshState.posedMesh_, constr) + constr.depth * targetNormal;

      if (sourceNormal.dot(targetNormal) < 0) {
        targetNormal *= -1;
      }
      const Eigen::Vector3<T> normal =
          sourceNormalWeight * sourceNormal + targetNormalWeight * targetNormal;

      // calculate point->plane distance and error
      const Eigen::Vector3<T> diff = srcPoint - tgtPoint;
      const T dist = normal.dot(diff);
      error += constr.weight * dist * dist * kPlaneWeight;
    }
  }

  // return error
  return error * this->weight_;
}

template <typename T>
void gradient_jointParams_to_modelParams(
    const T grad_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  // explicitly multiply with the parameter transform to generate parameter space gradients
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    gradient[parameterTransform.transform.innerIndexPtr()[index]] +=
        grad_jointParams * parameterTransform.transform.valuePtr()[index];
  }
}

template <typename T>
void jacobian_jointParams_to_modelParams(
    const Eigen::Ref<const Eigen::VectorX<T>> jacobian_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian) {
  // explicitly multiply with the parameter transform to generate parameter space gradients
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    jacobian.col(parameterTransform.transform.innerIndexPtr()[index]) +=
        jacobian_jointParams * parameterTransform.transform.valuePtr()[index];
  }
}

template <typename T>
void jacobian_jointParams_to_modelParams(
    const T jacobian_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian) {
  // explicitly multiply with the parameter transform to generate parameter space gradients
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    jacobian(0, parameterTransform.transform.innerIndexPtr()[index]) +=
        jacobian_jointParams * parameterTransform.transform.valuePtr()[index];
  }
}

template <typename T>
Eigen::Vector3<T> calculateDWorldPos(
    const momentum::Character& character,
    const SkeletonStateT<T>& state,
    const int vertexIndex,
    const Eigen::Vector3<T>& d_restPos) {
  const auto& skinWeights = *character.skinWeights;

  Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
  for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
    const auto w = skinWeights.weight(vertexIndex, i);
    const auto parentBone = skinWeights.index(vertexIndex, i);
    if (w > 0) {
      d_worldPos += w *
          (state.jointState[parentBone].transform.toLinear() *
           (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
    }
  }

  return d_worldPos;
}

template <typename T>
double PointTriangleVertexErrorFunctionT<T>::calculatePositionGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    const PointTriangleVertexConstraintT<T>& constr,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  const Eigen::Vector3<T> srcPos = meshState.posedMesh_->vertices[constr.srcVertexIndex];
  const Eigen::Vector3<T> tgtPos = computeTargetPosition(*meshState.posedMesh_, constr);

  const Eigen::Vector3<T> diff = srcPos - tgtPos;

  // calculate the difference between target and position and error
  const T wgt = constr.weight * 2.0f * kPositionWeight * this->weight_;

  const auto d_targetPos_d_tgtTriVertexPos =
      compute_d_targetPos_d_vertexPos(constr, *meshState.posedMesh_);

  // To simplify things we'll have one loop that goes through all 3 target triangle vertices _and_
  // the source vertex.
  //   iter 0-3: target triangle vertices
  //   iter 4: source vertex
  for (int iTriVert = 0; iTriVert < 4; ++iTriVert) {
    const auto vertexIndex =
        iTriVert < 3 ? constr.tgtTriangleIndices[iTriVert] : constr.srcVertexIndex;
    // Flip sign for the target vertex since the actual error is (srcPoint - tgtPoint)
    const Eigen::Matrix3<T> d_targetPos_d_vertexIndex = iTriVert < 3
        ? Eigen::Matrix3<T>(-d_targetPos_d_tgtTriVertexPos[iTriVert])
        : Eigen::Matrix3<T>::Identity();

    SkinningWeightIteratorT<T> skinningIter(
        this->character_, *meshState.restMesh_, state, vertexIndex);

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
              boneWeight *
                  diff.dot(d_targetPos_d_vertexIndex * jointState.getTranslationDerivative(d)) *
                  wgt,
              paramIndex + d,
              this->parameterTransform_,
              gradient);
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          // Gradient wrt rotation:
          gradient_jointParams_to_modelParams(
              boneWeight *
                  diff.dot(d_targetPos_d_vertexIndex * jointState.getRotationDerivative(d, posd)) *
                  wgt,
              paramIndex + 3 + d,
              this->parameterTransform_,
              gradient);
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        // Gradient wrt scale:
        gradient_jointParams_to_modelParams(
            boneWeight * diff.dot(d_targetPos_d_vertexIndex * jointState.getScaleDerivative(posd)) *
                wgt,
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
                .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
                .template cast<T>();
        Eigen::Vector3<T> d_worldPos =
            calculateDWorldPos(character_, state, vertexIndex, d_restPos);

        gradient[paramIdx] += wgt * diff.dot(d_targetPos_d_vertexIndex * d_worldPos);
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
                .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
                .template cast<T>();
        Eigen::Vector3<T> d_worldPos =
            calculateDWorldPos(character_, state, vertexIndex, d_restPos);

        gradient[paramIdx] += wgt * diff.dot(d_targetPos_d_vertexIndex * d_worldPos);
      }
    }
    // OUT handle derivatives wrt face expression blend shape parameters
  }

  return constr.weight * diff.squaredNorm() * kPositionWeight;
}

template <typename T>
std::array<Eigen::Matrix3<T>, 3> compute_d_targetNormal_d_vertexPos(
    const PointTriangleVertexConstraintT<T>& cons,
    const MeshT<T>& mesh) {
  std::array<Eigen::Matrix3<T>, 3> result;
  for (auto& r : result) {
    r.setZero();
  }

  const std::array<Eigen::Vector3<T>, 3> tgtTrianglePositions = {
      mesh.vertices[cons.tgtTriangleIndices[0]],
      mesh.vertices[cons.tgtTriangleIndices[1]],
      mesh.vertices[cons.tgtTriangleIndices[2]]};

  Eigen::Vector3<T> n = (tgtTrianglePositions[1] - tgtTrianglePositions[0])
                            .cross(tgtTrianglePositions[2] - tgtTrianglePositions[0]);
  T n_norm = n.norm();
  if (n_norm < 1e-6) {
    // ignore normal gradients
    return result;
  }
  n /= n_norm;

  const T area_times_2 = n_norm;

  for (int k = 0; k < 3; ++k) {
    // https://www.cs.cmu.edu/~kmcrane/Projects/Other/TriangleMeshDerivativesCheatSheet.pdf
    const Eigen::Vector3<T> e =
        tgtTrianglePositions[(k + 2) % 3] - tgtTrianglePositions[(k + 1) % 3];
    result[k] += (e.cross(n) * n.transpose()) / area_times_2;
  }

  return result;
}

template <typename T>
std::array<Eigen::Matrix3<T>, 3> compute_d_targetPos_d_vertexPos(
    const PointTriangleVertexConstraintT<T>& cons,
    const MeshT<T>& mesh) {
  std::array<Eigen::Matrix3<T>, 3> result;
  result[0] = Eigen::Matrix3<T>::Identity() * cons.tgtTriangleBaryCoords[0];
  result[1] = Eigen::Matrix3<T>::Identity() * cons.tgtTriangleBaryCoords[1];
  result[2] = Eigen::Matrix3<T>::Identity() * cons.tgtTriangleBaryCoords[2];

  const auto dNormal = compute_d_targetNormal_d_vertexPos(cons, mesh);
  for (int k = 0; k < 3; ++k) {
    result[k] += cons.depth * dNormal[k];
  }

  return result;
}

template <typename T>
double PointTriangleVertexErrorFunctionT<T>::calculatePositionJacobian(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    const PointTriangleVertexConstraintT<T>& constr,
    Ref<Eigen::MatrixX<T>> jac,
    Ref<Eigen::VectorX<T>> res) const {
  // calculate the difference between target and position and error
  const Eigen::Vector3<T> srcPos = meshState.posedMesh_->vertices[constr.srcVertexIndex];
  const Eigen::Vector3<T> tgtPos = computeTargetPosition(*meshState.posedMesh_, constr);

  // The derivative of the target position wrt the triangle vertices:
  //   p_tgt = bary_0 * p_0 + bary_1 * p_1 + bary_2 * p_2 + depth * n

  const Eigen::Vector3<T> diff = meshState.posedMesh_->vertices[constr.srcVertexIndex] - tgtPos;

  const T wgt = std::sqrt(constr.weight * kPositionWeight * this->weight_);

  const auto d_targetPos_d_tgtTriVertexPos =
      compute_d_targetPos_d_vertexPos(constr, *meshState.posedMesh_);

// Verify derivative:
#if 0
  {
    const int kTriVert = 0;
    auto meshTmp = *this->posedMesh_;
    const float eps = 1e-5;
    const Eigen::Vector3<T> srcPosOrig = meshTmp.vertices[constr.tgtTriangleIndices[kTriVert]];
    Eigen::Matrix3<T> diff_est = Eigen::Matrix3<T>::Zero();
    for (int k = 0; k < 3; ++k) {
      meshTmp.vertices[constr.tgtTriangleIndices[kTriVert]] =
          srcPosOrig + eps * Eigen::Vector3<T>::Unit(k);
      const Eigen::Vector3<T> tgtPosPlus = computeTargetPosition(meshTmp, constr);
      meshTmp.vertices[constr.tgtTriangleIndices[kTriVert]] = srcPosOrig;

      const Eigen::Vector3<T> tgtPosPlusDiff = (tgtPosPlus - tgtPos) / eps;
      diff_est.col(k) = tgtPosPlusDiff;
    }

    const T derivativeMismatch = (d_targetPos_d_srcPos[kTriVert] - diff_est).norm();
    if (derivativeMismatch > 1e-6) {
      MT_LOGI("Target position derivative mismatch detected: {}", derivativeMismatch);
      MT_LOGI("  d_targetPos_d_srcPos (computed): \n{}", d_targetPos_d_srcPos[kTriVert]);
      MT_LOGI("  d_targetPos_d_srcPos (estimated): \n{}", diff_est);
    }
  }
#endif

  // To simplify things we'll have one loop that goes through all 3 target triangle vertices _and_
  // the source vertex.
  //   iter 0-3: target triangle vertices
  //   iter 4: source vertex
  for (int iTriVert = 0; iTriVert < 4; ++iTriVert) {
    const auto vertexIndex =
        iTriVert < 3 ? constr.tgtTriangleIndices[iTriVert] : constr.srcVertexIndex;
    // Flip sign for the target vertex since the actual error is (srcPoint - tgtPoint)
    const Eigen::Matrix3<T> d_targetPos_d_vertexPos = iTriVert < 3
        ? Eigen::Matrix3<T>(-d_targetPos_d_tgtTriVertexPos[iTriVert])
        : Eigen::Matrix3<T>::Identity();

    SkinningWeightIteratorT<T> skinningIter(
        this->character_, *meshState.restMesh_, state, vertexIndex);

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
              boneWeight * wgt * d_targetPos_d_vertexPos * jointState.getTranslationDerivative(d),
              paramIndex + d,
              this->parameterTransform_,
              jac);
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          jacobian_jointParams_to_modelParams<T>(
              boneWeight * wgt * d_targetPos_d_vertexPos *
                  jointState.getRotationDerivative(d, posd),
              paramIndex + d + 3,
              this->parameterTransform_,
              jac);
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * d_targetPos_d_vertexPos * jointState.getScaleDerivative(posd),
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
                .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
                .template cast<T>();
        Eigen::Vector3<T> d_worldPos =
            calculateDWorldPos(this->character_, state, vertexIndex, d_restPos);

        jac.col(paramIdx) += wgt * d_targetPos_d_vertexPos * d_worldPos;
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
                .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
                .template cast<T>();
        Eigen::Vector3<T> d_worldPos =
            calculateDWorldPos(this->character_, state, vertexIndex, d_restPos);

        jac.col(paramIdx) += wgt * d_targetPos_d_vertexPos * d_worldPos;
      }
    }
  }
  // OUT handle derivatives wrt face expression blend shape parameters
  res = diff * wgt;

  return wgt * wgt * diff.squaredNorm();
}

template <typename T>
double PointTriangleVertexErrorFunctionT<T>::calculateNormalGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    const PointTriangleVertexConstraintT<T>& constr,
    const T sourceNormalWeight,
    const T targetNormalWeight,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  const Eigen::Vector3<T> sourceNormal = meshState.posedMesh_->normals[constr.srcVertexIndex];
  Eigen::Vector3<T> targetNormal = computeTargetTriangleNormal(*meshState.posedMesh_, constr);
  const Eigen::Vector3<T> tgtPosition = computeTargetPosition(*meshState.posedMesh_, constr);

  T targetNormalSign = 1;
  if (sourceNormal.dot(targetNormal) < 0) {
    targetNormal *= -1;
    targetNormalSign = -1;
  }
  const Eigen::Vector3<T> normal =
      sourceNormalWeight * sourceNormal + targetNormalWeight * targetNormal;

  // calculate the difference between target and position and error
  const Eigen::Vector3<T> diff =
      meshState.posedMesh_->vertices[constr.srcVertexIndex] - tgtPosition;
  const T dist = diff.dot(normal);

  const T wgt = constr.weight * 2.0f * kPlaneWeight * this->weight_;

  const auto d_targetPos_d_tgtTriVertexPos =
      compute_d_targetPos_d_vertexPos(constr, *meshState.posedMesh_);
  const auto d_targetNormal_d_tgtTriVertexPos =
      compute_d_targetNormal_d_vertexPos(constr, *meshState.posedMesh_);

  for (int iTriVert = 0; iTriVert < 4; ++iTriVert) {
    const auto vertexIndex =
        iTriVert < 3 ? constr.tgtTriangleIndices[iTriVert] : constr.srcVertexIndex;
    // Flip sign for the target vertex since the actual error is (srcPoint - tgtPoint)
    const Eigen::Matrix3<T> d_targetPos_d_vertexPos = iTriVert < 3
        ? Eigen::Matrix3<T>(-d_targetPos_d_tgtTriVertexPos[iTriVert])
        : Eigen::Matrix3<T>::Identity();

    SkinningWeightIteratorT<T> skinningIter(
        this->character_, *meshState.restMesh_, state, vertexIndex);

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
          gradient_jointParams_to_modelParams<T>(
              boneWeight * dist *
                  normal.dot(d_targetPos_d_vertexPos * jointState.getTranslationDerivative(d)) *
                  wgt,
              paramIndex + d,
              this->parameterTransform_,
              gradient);
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          // Error is (p_src - p_tgt) . (w_src * n_src + w_tgt * n_tgt)
          // Full derivative will be:
          //   (d/dTheta (p_src - p_tgt)) . (w_src * n_src + w_tgt * n_tgt) +
          //             (p_src - p_tgt) . (w_src * dn_src/dTheta) +
          //             (p_src - p_tgt) . (w_tgt * dn_tgt/dTheta))

          // Jacobian wrt rotation:
          T diff_total =
              normal.dot(d_targetPos_d_vertexPos * jointState.getRotationDerivative(d, posd));

          if (iTriVert < 3) {
            const Eigen::Vector3<T> diff_tgt_normal = targetNormalSign *
                d_targetNormal_d_tgtTriVertexPos[iTriVert] *
                jointState.getRotationDerivative(d, posd);
            diff_total += targetNormalWeight * diff.dot(diff_tgt_normal);
          } else {
            // Jacobian wrt rotation:
            const Eigen::Vector3<T> diff_src_normal =
                jointState.getRotationDerivative(d, sourceNormal);
            diff_total += sourceNormalWeight * diff.dot(diff_src_normal);
          }

          gradient_jointParams_to_modelParams<T>(
              boneWeight * wgt * diff_total * dist,
              paramIndex + d + 3,
              this->parameterTransform_,
              gradient);
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        // Jacobian wrt scale:
        gradient_jointParams_to_modelParams<T>(
            boneWeight * wgt *
                normal.dot(d_targetPos_d_vertexPos * jointState.getScaleDerivative(posd)) * dist,
            paramIndex + 6,
            this->parameterTransform_,
            gradient);
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
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = calculateDWorldPos(character_, state, vertexIndex, d_restPos);

      gradient(paramIdx) += wgt * dist * normal.dot(d_targetPos_d_vertexPos * d_worldPos);
      if (iTriVert < 3) {
        const Eigen::Vector3<T> diff_tgt_normal =
            d_targetNormal_d_tgtTriVertexPos[iTriVert] * d_worldPos;
        gradient(paramIdx) += targetNormalSign * wgt * dist * diff.dot(diff_tgt_normal);
      }
    }
    // OUT handle derivatives wrt blend shape parameters:
  }

  return constr.weight * dist * dist * kPlaneWeight;
}

template <typename T>
double PointTriangleVertexErrorFunctionT<T>::calculateNormalJacobian(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    const PointTriangleVertexConstraintT<T>& constr,
    const T sourceNormalWeight,
    const T targetNormalWeight,
    Ref<Eigen::MatrixX<T>> jac,
    T& res) const {
  const Eigen::Vector3<T> sourceNormal = meshState.posedMesh_->normals[constr.srcVertexIndex];
  Eigen::Vector3<T> targetNormal = computeTargetTriangleNormal(*meshState.posedMesh_, constr);
  const Eigen::Vector3<T> tgtPosition = computeTargetPosition(*meshState.posedMesh_, constr);

  T targetNormalSign = 1;
  if (sourceNormal.dot(targetNormal) < 0) {
    targetNormal *= -1;
    targetNormalSign = -1;
  }
  const Eigen::Vector3<T> normal =
      sourceNormalWeight * sourceNormal + targetNormalWeight * targetNormal;

  // calculate the difference between target and position and error
  const Eigen::Vector3<T> diff =
      meshState.posedMesh_->vertices[constr.srcVertexIndex] - tgtPosition;
  const T dist = diff.dot(normal);

  const T wgt = std::sqrt(constr.weight * kPositionWeight * this->weight_);

  const auto d_targetPos_d_tgtTriVertexPos =
      compute_d_targetPos_d_vertexPos(constr, *meshState.posedMesh_);
  const auto d_targetNormal_d_tgtTriVertexPos =
      compute_d_targetNormal_d_vertexPos(constr, *meshState.posedMesh_);

  for (int iTriVert = 0; iTriVert < 4; ++iTriVert) {
    const auto vertexIndex =
        iTriVert < 3 ? constr.tgtTriangleIndices[iTriVert] : constr.srcVertexIndex;
    // Flip sign for the target vertex since the actual error is (srcPoint - tgtPoint)
    const Eigen::Matrix3<T> d_targetPos_d_vertexPos = iTriVert < 3
        ? Eigen::Matrix3<T>(-d_targetPos_d_tgtTriVertexPos[iTriVert])
        : Eigen::Matrix3<T>::Identity();

    SkinningWeightIteratorT<T> skinningIter(
        this->character_, *meshState.restMesh_, state, vertexIndex);

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
              boneWeight * wgt *
                  normal.dot(d_targetPos_d_vertexPos * jointState.getTranslationDerivative(d)),
              paramIndex + d,
              this->parameterTransform_,
              jac);
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          // Error is (p_src - p_tgt) . (w_src * n_src + w_tgt * n_tgt)
          // Full derivative will be:
          //   (d/dTheta (p_src - p_tgt)) . (w_src * n_src + w_tgt * n_tgt) +
          //             (p_src - p_tgt) . (w_src * dn_src/dTheta) +
          //             (p_src - p_tgt) . (w_tgt * dn_tgt/dTheta))

          // Jacobian wrt rotation:
          T diff_total =
              normal.dot(d_targetPos_d_vertexPos * jointState.getRotationDerivative(d, posd));

          if (iTriVert < 3) {
            const Eigen::Vector3<T> diff_tgt_normal = targetNormalSign *
                d_targetNormal_d_tgtTriVertexPos[iTriVert] *
                jointState.getRotationDerivative(d, posd);
            diff_total += targetNormalWeight * diff.dot(diff_tgt_normal);
          } else {
            // Jacobian wrt rotation:
            const Eigen::Vector3<T> diff_src_normal =
                jointState.getRotationDerivative(d, sourceNormal);
            diff_total += sourceNormalWeight * diff.dot(diff_src_normal);
          }

          jacobian_jointParams_to_modelParams<T>(
              boneWeight * wgt * diff_total, paramIndex + d + 3, this->parameterTransform_, jac);
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        // Jacobian wrt scale:
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt *
                normal.dot(d_targetPos_d_vertexPos * jointState.getScaleDerivative(posd)),
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
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = calculateDWorldPos(character_, state, vertexIndex, d_restPos);

      jac(0, paramIdx) += wgt * normal.dot(d_targetPos_d_vertexPos * d_worldPos);
      if (iTriVert < 3) {
        const Eigen::Vector3<T> diff_tgt_normal =
            d_targetNormal_d_tgtTriVertexPos[iTriVert] * d_worldPos;
        jac(0, paramIdx) += targetNormalSign * wgt * diff.dot(diff_tgt_normal);
      }
    }
    // OUT handle derivatives wrt blend shape parameters:
  }

  res = dist * wgt;

  return wgt * wgt * dist * dist;
}

template <typename T>
std::pair<T, T> PointTriangleVertexErrorFunctionT<T>::computeNormalWeights() const {
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
double PointTriangleVertexErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_CHECK_NOTNULL(meshState.posedMesh_);

  double error = 0;

  if (constraintType_ == VertexConstraintType::Position) {
    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      error += calculatePositionGradient(
          modelParameters, state, meshState, constraints_[iCons], gradient);
    }
  } else {
    T sourceNormalWeight;
    T targetNormalWeight;
    std::tie(sourceNormalWeight, targetNormalWeight) = computeNormalWeights();

    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      error += calculateNormalGradient(
          modelParameters,
          state,
          meshState,
          constraints_[iCons],
          sourceNormalWeight,
          targetNormalWeight,
          gradient);
    }
  }

  return this->weight_ * error;
}

template <typename T>
double PointTriangleVertexErrorFunctionT<T>::getJacobian(
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

  double error = 0;

  if (constraintType_ == VertexConstraintType::Position) {
    MT_PROFILE_EVENT("VertexErrorFunction - position jacobians");

    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      error += calculatePositionJacobian(
          modelParameters,
          state,
          meshState,
          constraints_[iCons],
          jacobian.block(3 * iCons, 0, 3, modelParameters.size()),
          residual.middleRows(3 * iCons, 3));
    }
    usedRows = 3 * constraints_.size();
  } else {
    MT_PROFILE_EVENT("VertexErrorFunction - normal jacobians");
    T sourceNormalWeight;
    T targetNormalWeight;
    std::tie(sourceNormalWeight, targetNormalWeight) = computeNormalWeights();

    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      error += calculateNormalJacobian(
          modelParameters,
          state,
          meshState,
          constraints_[iCons],
          sourceNormalWeight,
          targetNormalWeight,
          jacobian.block(iCons, 0, 1, modelParameters.size()),
          residual(iCons));
    }

    usedRows = constraints_.size();
  }

  return error;
}

template <typename T>
size_t PointTriangleVertexErrorFunctionT<T>::getJacobianSize() const {
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

template class PointTriangleVertexErrorFunctionT<float>;
template class PointTriangleVertexErrorFunctionT<double>;

} // namespace momentum
