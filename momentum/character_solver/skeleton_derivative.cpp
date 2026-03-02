/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/skeleton_derivative.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/character_solver/skinning_weight_iterator.h"
#include "momentum/common/checks.h"

#include <array>
#include <limits>

namespace momentum {

template <typename T>
SkeletonDerivativeT<T>::SkeletonDerivativeT(
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform,
    const VectorX<bool>& activeJointParams,
    const ParameterSet& enabledParameters)
    : skeleton_(skeleton),
      parameterTransform_(parameterTransform),
      activeJointParams_(activeJointParams),
      enabledParameters_(enabledParameters) {}

template <typename T>
template <size_t FuncDim>
void SkeletonDerivativeT<T>::accumulateJointGradient(
    size_t jointIndex,
    std::span<const Eigen::Vector3<T>> worldVecs,
    std::span<const Eigen::Matrix<T, FuncDim, 3>> dfdvs,
    std::span<const uint8_t> isPoints,
    const Eigen::Matrix<T, FuncDim, 1>& weightedResidual,
    const JointStateListT<T>& jointStates,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_CHECK(worldVecs.size() == dfdvs.size());
  MT_CHECK(worldVecs.size() == isPoints.size());

  const size_t numVecs = worldVecs.size();
  if (numVecs == 0) {
    return;
  }

  // Check if all derivatives are zero
  bool allZero = true;
  for (size_t i = 0; i < numVecs; ++i) {
    if (!dfdvs[i].isZero()) {
      allZero = false;
      break;
    }
  }
  if (allZero) {
    return;
  }

  // ---- Combined fast path for 1 JointPoint + N JointDirections ----
  // When multiple contributions share the same joint (e.g. CameraProjection with
  // 1 point + 3 directions), we combine them into a single hierarchy walk using
  // the triple product identity: wR . (axis x offset) = axis . (offset x wR).
  // Precompute C = pointWorldVec x pointWR + sum_dirs dirWorldVec[k] x dirWR[k],
  // then at each joint j: rotVal = rotAxis[d] . (C + pointWR x t_j).
  {
    size_t pointIdx = SIZE_MAX;
    bool canUseCombined = (numVecs > 1);
    if (canUseCombined) {
      for (size_t i = 0; i < numVecs; ++i) {
        if (isPoints[i]) {
          if (pointIdx != SIZE_MAX) {
            canUseCombined = false;
            break;
          }
          pointIdx = i;
        }
      }
      if (pointIdx == SIZE_MAX) {
        canUseCombined = false;
      }
    }

    if (canUseCombined) {
      const Eigen::Vector3<T> pointWR = dfdvs[pointIdx].transpose() * weightedResidual;

      // C = pointWorldVec x pointWR + sum_dirs dirWorldVec[k] x dirWR[k]
      Eigen::Vector3<T> C = worldVecs[pointIdx].cross(pointWR);
      for (size_t i = 0; i < numVecs; ++i) {
        if (i == pointIdx || dfdvs[i].isZero()) {
          continue;
        }
        const Eigen::Vector3<T> dirWR = dfdvs[i].transpose() * weightedResidual;
        C += worldVecs[i].cross(dirWR);
      }

      size_t jntIndex = jointIndex;
      while (jntIndex != kInvalidIndex) {
        MT_CHECK(jntIndex < skeleton_.joints.size());

        const auto& jntState = jointStates[jntIndex];
        const size_t jntParamIndex = jntIndex * kParametersPerJoint;

        // Translation DOFs: only the JointPoint contributes
        for (size_t d = 0; d < 3; ++d) {
          if (!activeJointParams_[jntParamIndex + d]) {
            continue;
          }
          const T val = pointWR.dot(jntState.getTranslationDerivative(d));
          for (auto index = parameterTransform_.transform.outerIndexPtr()[jntParamIndex + d];
               index < parameterTransform_.transform.outerIndexPtr()[jntParamIndex + d + 1];
               ++index) {
            if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
              gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * parameterTransform_.transform.valuePtr()[index];
            }
          }
        }

        // Rotation DOFs: val = rotAxis[d] . (C + pointWR x t_j)
        const Eigen::Vector3<T> Cj = C + pointWR.cross(jntState.translation());
        for (size_t d = 0; d < 3; ++d) {
          if (!activeJointParams_[jntParamIndex + 3 + d]) {
            continue;
          }
          const T val = jntState.rotationAxis.col(d).dot(Cj);
          for (auto index = parameterTransform_.transform.outerIndexPtr()[jntParamIndex + 3 + d];
               index < parameterTransform_.transform.outerIndexPtr()[jntParamIndex + 3 + d + 1];
               ++index) {
            if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
              gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * parameterTransform_.transform.valuePtr()[index];
            }
          }
        }

        // Scale DOF: only the JointPoint contributes
        if (activeJointParams_[jntParamIndex + 6]) {
          const Eigen::Vector3<T> offset = worldVecs[pointIdx] - jntState.translation();
          const T val = pointWR.dot(jntState.getScaleDerivative(offset));
          for (auto index = parameterTransform_.transform.outerIndexPtr()[jntParamIndex + 6];
               index < parameterTransform_.transform.outerIndexPtr()[jntParamIndex + 6 + 1];
               ++index) {
            if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
              gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * parameterTransform_.transform.valuePtr()[index];
            }
          }
        }

        jntIndex = skeleton_.joints[jntIndex].parent;
      }
      return;
    }
  }

  // Walk the joint hierarchy from this joint to the root
  size_t jntIndex = jointIndex;
  while (jntIndex != kInvalidIndex) {
    MT_CHECK(jntIndex < skeleton_.joints.size());

    const auto& jntState = jointStates[jntIndex];
    const size_t jntParamIndex = jntIndex * kParametersPerJoint;

    for (size_t iVec = 0; iVec < numVecs; ++iVec) {
      if (dfdvs[iVec].isZero()) {
        continue;
      }

      // wR = dfdv^T * weightedResidual: converts matmul+dot to 3D dot
      const Eigen::Vector3<T> wR = dfdvs[iVec].transpose() * weightedResidual;

      // Compute offset for rotation/scale derivatives
      // For points (w=1): offset = worldVec - translation
      // For directions (w=0): offset = worldVec
      Eigen::Vector3<T> offset;
      if (isPoints[iVec]) {
        offset.noalias() = worldVecs[iVec] - jntState.translation();
      } else {
        offset = worldVecs[iVec];
      }

      // Translational DOFs -- only affect points (w=1)
      if (isPoints[iVec]) {
        for (size_t d = 0; d < 3; ++d) {
          if (!activeJointParams_[jntParamIndex + d]) {
            continue;
          }

          const T val = wR.dot(jntState.getTranslationDerivative(d));

          // Use the existing utility to map joint params to model params
          for (auto index = parameterTransform_.transform.outerIndexPtr()[jntParamIndex + d];
               index < parameterTransform_.transform.outerIndexPtr()[jntParamIndex + d + 1];
               ++index) {
            if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
              gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * parameterTransform_.transform.valuePtr()[index];
            }
          }
        }
      }

      // Rotational DOFs -- affect both points and directions
      for (size_t d = 0; d < 3; ++d) {
        if (!activeJointParams_[jntParamIndex + 3 + d]) {
          continue;
        }

        const T val = wR.dot(jntState.getRotationDerivative(d, offset));

        for (auto index = parameterTransform_.transform.outerIndexPtr()[jntParamIndex + 3 + d];
             index < parameterTransform_.transform.outerIndexPtr()[jntParamIndex + 3 + d + 1];
             ++index) {
          if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
            gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * parameterTransform_.transform.valuePtr()[index];
          }
        }
      }

      // Scale DOF -- only affects points (w=1)
      if (isPoints[iVec]) {
        if (activeJointParams_[jntParamIndex + 6]) {
          const T val = wR.dot(jntState.getScaleDerivative(offset));

          for (auto index = parameterTransform_.transform.outerIndexPtr()[jntParamIndex + 6];
               index < parameterTransform_.transform.outerIndexPtr()[jntParamIndex + 6 + 1];
               ++index) {
            if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
              gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * parameterTransform_.transform.valuePtr()[index];
            }
          }
        }
      }
    }

    // Move to parent joint
    jntIndex = skeleton_.joints[jntIndex].parent;
  }
}

template <typename T>
template <size_t FuncDim>
void SkeletonDerivativeT<T>::accumulateJointJacobian(
    size_t jointIndex,
    std::span<const Eigen::Vector3<T>> worldVecs,
    std::span<const Eigen::Matrix<T, FuncDim, 3>> dfdvs,
    std::span<const uint8_t> isPoints,
    T scale,
    const JointStateListT<T>& jointStates,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    size_t rowIndex) const {
  MT_CHECK(worldVecs.size() == dfdvs.size());
  MT_CHECK(worldVecs.size() == isPoints.size());

  const size_t numVecs = worldVecs.size();
  if (numVecs == 0) {
    return;
  }

  // Check if all derivatives are zero
  bool allZero = true;
  for (size_t i = 0; i < numVecs; ++i) {
    if (!dfdvs[i].isZero()) {
      allZero = false;
      break;
    }
  }
  if (allZero) {
    return;
  }

  // Get the jacobian submatrix for this constraint
  // Use runtime-sized middleRows to handle FuncDim=1 Eigen block type issues
  auto jac = jacobian.middleRows(rowIndex, FuncDim);

  // Walk the joint hierarchy from this joint to the root
  size_t jntIndex = jointIndex;
  while (jntIndex != kInvalidIndex) {
    MT_CHECK(jntIndex < skeleton_.joints.size());

    const auto& jntState = jointStates[jntIndex];
    const size_t jntParamIndex = jntIndex * kParametersPerJoint;

    for (size_t iVec = 0; iVec < numVecs; ++iVec) {
      if (dfdvs[iVec].isZero()) {
        continue;
      }

      // Compute offset for rotation/scale derivatives
      Eigen::Vector3<T> offset;
      if (isPoints[iVec]) {
        offset.noalias() = worldVecs[iVec] - jntState.translation();
      } else {
        offset = worldVecs[iVec];
      }

      // Translational DOFs -- only affect points (w=1)
      if (isPoints[iVec]) {
        for (size_t d = 0; d < 3; ++d) {
          if (!activeJointParams_[jntParamIndex + d]) {
            continue;
          }

          // Compute jacobian contribution: scale * dfdv * derivative
          const Eigen::Matrix<T, FuncDim, 1> jc =
              scale * dfdvs[iVec] * jntState.getTranslationDerivative(d);

          for (auto index = parameterTransform_.transform.outerIndexPtr()[jntParamIndex + d];
               index < parameterTransform_.transform.outerIndexPtr()[jntParamIndex + d + 1];
               ++index) {
            if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
              jac.col(parameterTransform_.transform.innerIndexPtr()[index]).noalias() +=
                  jc * parameterTransform_.transform.valuePtr()[index];
            }
          }
        }
      }

      // Rotational DOFs -- affect both points and directions
      for (size_t d = 0; d < 3; ++d) {
        if (!activeJointParams_[jntParamIndex + 3 + d]) {
          continue;
        }

        const Eigen::Matrix<T, FuncDim, 1> jc =
            scale * dfdvs[iVec] * jntState.getRotationDerivative(d, offset);

        for (auto index = parameterTransform_.transform.outerIndexPtr()[jntParamIndex + 3 + d];
             index < parameterTransform_.transform.outerIndexPtr()[jntParamIndex + 3 + d + 1];
             ++index) {
          if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
            jac.col(parameterTransform_.transform.innerIndexPtr()[index]).noalias() +=
                jc * parameterTransform_.transform.valuePtr()[index];
          }
        }
      }

      // Scale DOF -- only affects points (w=1)
      if (isPoints[iVec]) {
        if (activeJointParams_[jntParamIndex + 6]) {
          const Eigen::Matrix<T, FuncDim, 1> jc =
              scale * dfdvs[iVec] * jntState.getScaleDerivative(offset);

          for (auto index = parameterTransform_.transform.outerIndexPtr()[jntParamIndex + 6];
               index < parameterTransform_.transform.outerIndexPtr()[jntParamIndex + 6 + 1];
               ++index) {
            if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
              jac.col(parameterTransform_.transform.innerIndexPtr()[index]).noalias() +=
                  jc * parameterTransform_.transform.valuePtr()[index];
            }
          }
        }
      }
    }

    // Move to parent joint
    jntIndex = skeleton_.joints[jntIndex].parent;
  }
}

template <typename T>
template <size_t FuncDim>
void SkeletonDerivativeT<T>::accumulateVertexGradient(
    size_t vertexIndex,
    const Eigen::Vector3<T>& /* worldPos */,
    const Eigen::Matrix<T, FuncDim, 3>& dfdv,
    const Eigen::Matrix<T, FuncDim, 1>& weightedResidual,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  // Skip if derivative is zero
  if (dfdv.isZero()) {
    return;
  }

  MT_CHECK(meshState.restMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);
  MT_CHECK(!character.inverseBindPose.empty());

  // Use SkinningWeightIterator to walk through influencing bones
  SkinningWeightIteratorT<T> skinningIter(
      character, *meshState.restMesh_, skeletonState, vertexIndex);

  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    MT_CHECK(jointIndex < skeleton_.joints.size());

    const auto& jointState = skeletonState.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> offset = pos - jointState.translation();

    // Translation DOFs
    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + d]) {
        continue;
      }

      const T val =
          boneWeight * weightedResidual.dot(dfdv * jointState.getTranslationDerivative(d));

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }

    // Rotation DOFs
    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + 3 + d]) {
        continue;
      }

      const T val =
          boneWeight * weightedResidual.dot(dfdv * jointState.getRotationDerivative(d, offset));

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }

    // Scale DOF
    if (activeJointParams_[paramIndex + 6]) {
      const T val = boneWeight * weightedResidual.dot(dfdv * jointState.getScaleDerivative(offset));

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }
  }

  // Handle blend shape parameters
  if (character.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.blendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      // Transform d_restPos through skinning
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      gradient[paramIdx] += weightedResidual.dot(dfdv * d_worldPos);
    }
  }

  // Handle face expression blend shape parameters
  if (character.faceExpressionBlendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.faceExpressionParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.faceExpressionParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.faceExpressionBlendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      // Transform d_restPos through skinning
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      gradient[paramIdx] += weightedResidual.dot(dfdv * d_worldPos);
    }
  }
}

template <typename T>
template <size_t FuncDim>
void SkeletonDerivativeT<T>::accumulateVertexJacobian(
    size_t vertexIndex,
    const Eigen::Vector3<T>& /* worldPos */,
    const Eigen::Matrix<T, FuncDim, 3>& dfdv,
    T scale,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    size_t rowIndex) const {
  // Skip if derivative is zero
  if (dfdv.isZero()) {
    return;
  }

  MT_CHECK(meshState.restMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);
  MT_CHECK(!character.inverseBindPose.empty());

  // Get the jacobian submatrix for this constraint
  auto jac = jacobian.middleRows(rowIndex, FuncDim);

  // Use SkinningWeightIterator to walk through influencing bones
  SkinningWeightIteratorT<T> skinningIter(
      character, *meshState.restMesh_, skeletonState, vertexIndex);

  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    MT_CHECK(jointIndex < skeleton_.joints.size());

    const auto& jointState = skeletonState.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> offset = pos - jointState.translation();
    const T scaledBoneWeight = scale * boneWeight;

    // Translation DOFs
    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + d]) {
        continue;
      }

      const Eigen::Matrix<T, FuncDim, 1> jc =
          scaledBoneWeight * dfdv * jointState.getTranslationDerivative(d);

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          jac.col(parameterTransform_.transform.innerIndexPtr()[index]).noalias() +=
              jc * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }

    // Rotation DOFs
    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + 3 + d]) {
        continue;
      }

      const Eigen::Matrix<T, FuncDim, 1> jc =
          scaledBoneWeight * dfdv * jointState.getRotationDerivative(d, offset);

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          jac.col(parameterTransform_.transform.innerIndexPtr()[index]).noalias() +=
              jc * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }

    // Scale DOF
    if (activeJointParams_[paramIndex + 6]) {
      const Eigen::Matrix<T, FuncDim, 1> jc =
          scaledBoneWeight * dfdv * jointState.getScaleDerivative(offset);

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          jac.col(parameterTransform_.transform.innerIndexPtr()[index]).noalias() +=
              jc * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }
  }

  // Handle blend shape parameters
  if (character.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.blendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      // Transform d_restPos through skinning
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      jac.col(paramIdx).noalias() += scale * dfdv * d_worldPos;
    }
  }

  // Handle face expression blend shape parameters
  if (character.faceExpressionBlendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.faceExpressionParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.faceExpressionParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.faceExpressionBlendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      // Transform d_restPos through skinning
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      jac.col(paramIdx).noalias() += scale * dfdv * d_worldPos;
    }
  }
}

template <typename T>
void SkeletonDerivativeT<T>::accumulateVertexGradientIdentityDfdv(
    size_t vertexIndex,
    const Eigen::Vector3<T>& weightedResidual,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_CHECK(meshState.restMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);
  MT_CHECK(!character.inverseBindPose.empty());

  SkinningWeightIteratorT<T> skinningIter(
      character, *meshState.restMesh_, skeletonState, vertexIndex);

  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    MT_CHECK(jointIndex < skeleton_.joints.size());

    const auto& jointState = skeletonState.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> offset = pos - jointState.translation();

    // Translation DOFs (dfdv=I eliminates matrix multiply)
    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + d]) {
        continue;
      }

      const T val = boneWeight * weightedResidual.dot(jointState.getTranslationDerivative(d));

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }

    // Rotation DOFs (dfdv=I eliminates matrix multiply)
    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + 3 + d]) {
        continue;
      }

      const T val = boneWeight * weightedResidual.dot(jointState.getRotationDerivative(d, offset));

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }

    // Scale DOF (dfdv=I eliminates matrix multiply)
    if (activeJointParams_[paramIndex + 6]) {
      const T val = boneWeight * weightedResidual.dot(jointState.getScaleDerivative(offset));

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }
  }

  // Handle blend shape parameters (dfdv=I eliminates matrix multiply)
  if (character.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.blendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      gradient[paramIdx] += weightedResidual.dot(d_worldPos);
    }
  }

  // Handle face expression blend shape parameters (dfdv=I eliminates matrix multiply)
  if (character.faceExpressionBlendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.faceExpressionParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.faceExpressionParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.faceExpressionBlendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      gradient[paramIdx] += weightedResidual.dot(d_worldPos);
    }
  }
}

template <typename T>
void SkeletonDerivativeT<T>::accumulateVertexJacobianIdentityDfdv(
    size_t vertexIndex,
    T scale,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    size_t rowIndex) const {
  MT_CHECK(meshState.restMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);
  MT_CHECK(!character.inverseBindPose.empty());

  constexpr size_t FuncDim = 3;
  auto jac = jacobian.middleRows(rowIndex, FuncDim);

  SkinningWeightIteratorT<T> skinningIter(
      character, *meshState.restMesh_, skeletonState, vertexIndex);

  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    MT_CHECK(jointIndex < skeleton_.joints.size());

    const auto& jointState = skeletonState.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> offset = pos - jointState.translation();
    const T scaledBoneWeight = scale * boneWeight;

    // Translation DOFs (dfdv=I eliminates matrix multiply)
    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + d]) {
        continue;
      }

      const Eigen::Vector3<T> jc = scaledBoneWeight * jointState.getTranslationDerivative(d);

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          jac.col(parameterTransform_.transform.innerIndexPtr()[index]).noalias() +=
              jc * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }

    // Rotation DOFs (dfdv=I eliminates matrix multiply)
    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + 3 + d]) {
        continue;
      }

      const Eigen::Vector3<T> jc = scaledBoneWeight * jointState.getRotationDerivative(d, offset);

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          jac.col(parameterTransform_.transform.innerIndexPtr()[index]).noalias() +=
              jc * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }

    // Scale DOF (dfdv=I eliminates matrix multiply)
    if (activeJointParams_[paramIndex + 6]) {
      const Eigen::Vector3<T> jc = scaledBoneWeight * jointState.getScaleDerivative(offset);

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          jac.col(parameterTransform_.transform.innerIndexPtr()[index]).noalias() +=
              jc * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }
  }

  // Handle blend shape parameters (dfdv=I eliminates matrix multiply)
  if (character.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.blendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      jac.col(paramIdx).noalias() += scale * d_worldPos;
    }
  }

  // Handle face expression blend shape parameters (dfdv=I eliminates matrix multiply)
  if (character.faceExpressionBlendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.faceExpressionParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.faceExpressionParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.faceExpressionBlendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      jac.col(paramIdx).noalias() += scale * d_worldPos;
    }
  }
}

// Explicit template instantiations for float
template class SkeletonDerivativeT<float>;
template void SkeletonDerivativeT<float>::accumulateJointGradient<1>(
    size_t,
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Matrix<float, 1, 3>>,
    std::span<const uint8_t>,
    const Eigen::Matrix<float, 1, 1>&,
    const JointStateListT<float>&,
    Eigen::Ref<Eigen::VectorX<float>>) const;
template void SkeletonDerivativeT<float>::accumulateJointGradient<2>(
    size_t,
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Matrix<float, 2, 3>>,
    std::span<const uint8_t>,
    const Eigen::Matrix<float, 2, 1>&,
    const JointStateListT<float>&,
    Eigen::Ref<Eigen::VectorX<float>>) const;
template void SkeletonDerivativeT<float>::accumulateJointGradient<3>(
    size_t,
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Matrix<float, 3, 3>>,
    std::span<const uint8_t>,
    const Eigen::Matrix<float, 3, 1>&,
    const JointStateListT<float>&,
    Eigen::Ref<Eigen::VectorX<float>>) const;
template void SkeletonDerivativeT<float>::accumulateJointGradient<9>(
    size_t,
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Matrix<float, 9, 3>>,
    std::span<const uint8_t>,
    const Eigen::Matrix<float, 9, 1>&,
    const JointStateListT<float>&,
    Eigen::Ref<Eigen::VectorX<float>>) const;

template void SkeletonDerivativeT<float>::accumulateJointJacobian<1>(
    size_t,
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Matrix<float, 1, 3>>,
    std::span<const uint8_t>,
    float,
    const JointStateListT<float>&,
    Eigen::Ref<Eigen::MatrixX<float>>,
    size_t) const;
template void SkeletonDerivativeT<float>::accumulateJointJacobian<2>(
    size_t,
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Matrix<float, 2, 3>>,
    std::span<const uint8_t>,
    float,
    const JointStateListT<float>&,
    Eigen::Ref<Eigen::MatrixX<float>>,
    size_t) const;
template void SkeletonDerivativeT<float>::accumulateJointJacobian<3>(
    size_t,
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Matrix<float, 3, 3>>,
    std::span<const uint8_t>,
    float,
    const JointStateListT<float>&,
    Eigen::Ref<Eigen::MatrixX<float>>,
    size_t) const;
template void SkeletonDerivativeT<float>::accumulateJointJacobian<9>(
    size_t,
    std::span<const Eigen::Vector3<float>>,
    std::span<const Eigen::Matrix<float, 9, 3>>,
    std::span<const uint8_t>,
    float,
    const JointStateListT<float>&,
    Eigen::Ref<Eigen::MatrixX<float>>,
    size_t) const;

// Explicit template instantiations for double
template class SkeletonDerivativeT<double>;
template void SkeletonDerivativeT<double>::accumulateJointGradient<1>(
    size_t,
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Matrix<double, 1, 3>>,
    std::span<const uint8_t>,
    const Eigen::Matrix<double, 1, 1>&,
    const JointStateListT<double>&,
    Eigen::Ref<Eigen::VectorX<double>>) const;
template void SkeletonDerivativeT<double>::accumulateJointGradient<2>(
    size_t,
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Matrix<double, 2, 3>>,
    std::span<const uint8_t>,
    const Eigen::Matrix<double, 2, 1>&,
    const JointStateListT<double>&,
    Eigen::Ref<Eigen::VectorX<double>>) const;
template void SkeletonDerivativeT<double>::accumulateJointGradient<3>(
    size_t,
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Matrix<double, 3, 3>>,
    std::span<const uint8_t>,
    const Eigen::Matrix<double, 3, 1>&,
    const JointStateListT<double>&,
    Eigen::Ref<Eigen::VectorX<double>>) const;
template void SkeletonDerivativeT<double>::accumulateJointGradient<9>(
    size_t,
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Matrix<double, 9, 3>>,
    std::span<const uint8_t>,
    const Eigen::Matrix<double, 9, 1>&,
    const JointStateListT<double>&,
    Eigen::Ref<Eigen::VectorX<double>>) const;

template void SkeletonDerivativeT<double>::accumulateJointJacobian<1>(
    size_t,
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Matrix<double, 1, 3>>,
    std::span<const uint8_t>,
    double,
    const JointStateListT<double>&,
    Eigen::Ref<Eigen::MatrixX<double>>,
    size_t) const;
template void SkeletonDerivativeT<double>::accumulateJointJacobian<2>(
    size_t,
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Matrix<double, 2, 3>>,
    std::span<const uint8_t>,
    double,
    const JointStateListT<double>&,
    Eigen::Ref<Eigen::MatrixX<double>>,
    size_t) const;
template void SkeletonDerivativeT<double>::accumulateJointJacobian<3>(
    size_t,
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Matrix<double, 3, 3>>,
    std::span<const uint8_t>,
    double,
    const JointStateListT<double>&,
    Eigen::Ref<Eigen::MatrixX<double>>,
    size_t) const;
template void SkeletonDerivativeT<double>::accumulateJointJacobian<9>(
    size_t,
    std::span<const Eigen::Vector3<double>>,
    std::span<const Eigen::Matrix<double, 9, 3>>,
    std::span<const uint8_t>,
    double,
    const JointStateListT<double>&,
    Eigen::Ref<Eigen::MatrixX<double>>,
    size_t) const;

// Explicit template instantiations for accumulateVertexGradient (float)

// accumulateVertexNormalCorrectionGradient implementation
template <typename T>
void SkeletonDerivativeT<T>::accumulateVertexNormalCorrectionGradient(
    size_t vertexIndex,
    const Eigen::Vector3<T>& sourceNormal,
    const Eigen::Vector3<T>& targetPosition,
    T scale,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  // This computes the CORRECTION to the standard vertex gradient for
  // the source mesh normal rotation. The standard accumulateVertexGradient
  // treats the mixed normal as constant, but the sourceNormal rotates with
  // each bone. This correction fixes the rotation derivative.
  //
  // Formula: For each bone (jointIndex, boneWeight, pos):
  //   cross = (targetPosition - pos) × sourceNormal
  //   correction += boneWeight * scale * rotationAxis[d] · cross
  //
  // WHERE pos is the skinned rest position from SkinningWeightIterator,
  // NOT jointState.translation(). See VERTEX_REFACTORING_PLAN.md for
  // the complete derivation using scalar triple product identity.
  MT_CHECK(meshState.restMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);
  MT_CHECK(!character.inverseBindPose.empty());

  SkinningWeightIteratorT<T> skinningIter(
      character, *meshState.restMesh_, skeletonState, vertexIndex);

  while (!skinningIter.finished()) {
    auto [jointIndex, boneWeight, pos] = skinningIter.next();
    MT_CHECK(jointIndex < skeleton_.joints.size());

    const auto& jointState = skeletonState.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;

    // CRITICAL: use (targetPosition - pos), NOT (targetPosition - jointTranslation)
    const Eigen::Vector3<T> cross = (targetPosition - pos).cross(sourceNormal);

    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + 3 + d]) {
        continue;
      }
      const T val = boneWeight * scale * jointState.rotationAxis.col(d).dot(cross);
      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }
  }
}

// accumulateVertexNormalCorrectionJacobian implementation
template <typename T>
void SkeletonDerivativeT<T>::accumulateVertexNormalCorrectionJacobian(
    size_t vertexIndex,
    const Eigen::Vector3<T>& sourceNormal,
    const Eigen::Vector3<T>& targetPosition,
    T scale,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    size_t rowIndex) const {
  MT_CHECK(meshState.restMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);
  MT_CHECK(!character.inverseBindPose.empty());

  SkinningWeightIteratorT<T> skinningIter(
      character, *meshState.restMesh_, skeletonState, vertexIndex);

  while (!skinningIter.finished()) {
    auto [jointIndex, boneWeight, pos] = skinningIter.next();
    MT_CHECK(jointIndex < skeleton_.joints.size());

    const auto& jointState = skeletonState.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;

    const Eigen::Vector3<T> cross = (targetPosition - pos).cross(sourceNormal);

    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + 3 + d]) {
        continue;
      }
      const T val = boneWeight * scale * jointState.rotationAxis.col(d).dot(cross);
      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          jacobian(rowIndex, parameterTransform_.transform.innerIndexPtr()[index]) +=
              val * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }
  }
}

// accumulateVertexNormalBlendShapeGradient implementation
template <typename T>
void SkeletonDerivativeT<T>::accumulateVertexNormalBlendShapeGradient(
    size_t vertexIndex,
    const Eigen::Vector3<T>& leverArm,
    T scale,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  if (!character.blendShape && !character.faceExpressionBlendShape) {
    return;
  }

  MT_CHECK(meshState.posedMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);

  const auto& faces = meshState.posedMesh_->faces;

  // Find faces adjacent to this vertex via linear scan (fallback path)
  std::vector<size_t> adjacentFaces;
  for (size_t fIdx = 0; fIdx < faces.size(); ++fIdx) {
    const auto& face = faces[fIdx];
    if (face[0] == static_cast<int>(vertexIndex) || face[1] == static_cast<int>(vertexIndex) ||
        face[2] == static_cast<int>(vertexIndex)) {
      adjacentFaces.push_back(fIdx);
    }
  }

  accumulateVertexNormalBlendShapeGradient(
      vertexIndex, leverArm, scale, adjacentFaces, skeletonState, meshState, character, gradient);
}

// accumulateVertexNormalBlendShapeGradient with pre-built adjacency
template <typename T>
void SkeletonDerivativeT<T>::accumulateVertexNormalBlendShapeGradient(
    size_t vertexIndex,
    const Eigen::Vector3<T>& leverArm,
    T scale,
    std::span<const size_t> adjacentFaces,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  if (!character.blendShape && !character.faceExpressionBlendShape) {
    return;
  }

  MT_CHECK(meshState.posedMesh_ != nullptr);
  MT_CHECK(meshState.restMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);
  MT_CHECK(!character.inverseBindPose.empty());

  const auto& faces = meshState.posedMesh_->faces;
  const auto& restVertices = meshState.restMesh_->vertices;

  if (adjacentFaces.empty()) {
    return;
  }

  // The forward path computes posed normals as:
  //   restNormal = normalize(Σ_f restFaceNormal_f)        [updateNormals on rest mesh]
  //   posedNormal = normalize(M_v * restNormal)            [applySSD skins + normalizes]
  //
  // Since normalize(M_v * normalize(s)) = normalize(M_v * s), the derivative simplifies to:
  //   d(posedNormal)/d(s) = [(I - n̂*n̂^T) / |M_v * s|] * M_v
  // where s = restSum = unnormalized sum of rest face normals, n̂ = posedNormal.

  // Step 1: Compute rest-space edge vectors and unnormalized sum of face normals
  Eigen::Vector3<T> restSum = Eigen::Vector3<T>::Zero();
  struct FaceEdges {
    Eigen::Vector3<T> e1;
    Eigen::Vector3<T> e2;
    Eigen::Vector3i indices;
  };
  std::vector<FaceEdges> faceEdges(adjacentFaces.size());

  for (size_t i = 0; i < adjacentFaces.size(); ++i) {
    const auto& face = faces[adjacentFaces[i]];
    const Eigen::Vector3<T> v0 = restVertices[face[0]].template cast<T>();
    const Eigen::Vector3<T> v1 = restVertices[face[1]].template cast<T>();
    const Eigen::Vector3<T> v2 = restVertices[face[2]].template cast<T>();
    faceEdges[i].e1 = v1 - v0;
    faceEdges[i].e2 = v2 - v0;
    faceEdges[i].indices = face;
    restSum += faceEdges[i].e1.cross(faceEdges[i].e2);
  }

  if (restSum.squaredNorm() < std::numeric_limits<T>::epsilon()) {
    return;
  }

  // Step 2: Compute skinning rotation M_v for the constraint vertex
  const auto& skinWeights = *character.skinWeights;
  Eigen::Matrix3<T> Mv;
  Mv.setZero();
  for (uint32_t jWeight = 0; jWeight < kMaxSkinJoints; ++jWeight) {
    const auto w = skinWeights.weight(vertexIndex, jWeight);
    if (w > T(0)) {
      const auto parentBone = skinWeights.index(vertexIndex, jWeight);
      Mv.noalias() += w *
          (skeletonState.jointState[parentBone].transform.toLinear() *
           character.inverseBindPose[parentBone].linear().template cast<T>());
    }
  }

  // Step 3: Compute posed unnormalized = M_v * restSum and normalization derivative
  const Eigen::Vector3<T> posedUnnorm = Mv * restSum;
  const T posedNorm = posedUnnorm.norm();
  if (posedNorm < T(1e-12)) {
    return;
  }
  const Eigen::Vector3<T> posedNhat = posedUnnorm / posedNorm;

  // d(posedNormal)/d(restSum) = [(I - n̂*n̂^T) / |M_v * restSum|] * M_v
  const Eigen::Matrix3<T> dnormMv =
      ((Eigen::Matrix3<T>::Identity() - posedNhat * posedNhat.transpose()) / posedNorm) * Mv;

  // Process standard blend shapes
  if (character.blendShape) {
    const auto& shapeVectors = character.blendShape->getShapeVectors();
    for (Eigen::Index iBS = 0; iBS < parameterTransform_.blendShapeParameters.size(); ++iBS) {
      const auto paramIdx = parameterTransform_.blendShapeParameters[iBS];
      if (paramIdx < 0) {
        continue;
      }

      Eigen::Vector3<T> dRestSumDBs = Eigen::Vector3<T>::Zero();

      for (size_t i = 0; i < adjacentFaces.size(); ++i) {
        const auto& fe = faceEdges[i];

        // In rest space, d(restVertex)/d(bsWeight) = shapeVector (no skinning)
        const Eigen::Vector3<T> sv0 =
            shapeVectors.template block<3, 1>(3 * fe.indices[0], iBS, 3, 1).template cast<T>();
        const Eigen::Vector3<T> sv1 =
            shapeVectors.template block<3, 1>(3 * fe.indices[1], iBS, 3, 1).template cast<T>();
        const Eigen::Vector3<T> sv2 =
            shapeVectors.template block<3, 1>(3 * fe.indices[2], iBS, 3, 1).template cast<T>();

        const Eigen::Vector3<T> de1 = sv1 - sv0;
        const Eigen::Vector3<T> de2 = sv2 - sv0;

        // d(restFaceNormal)/d(bs) = de1 × e2 + e1 × de2
        dRestSumDBs += de1.cross(fe.e2) + fe.e1.cross(de2);
      }

      const Eigen::Vector3<T> dNormalDBs = dnormMv * dRestSumDBs;
      gradient[paramIdx] += scale * leverArm.dot(dNormalDBs);
    }
  }

  // Process face expression blend shapes
  if (character.faceExpressionBlendShape) {
    const auto& shapeVectors = character.faceExpressionBlendShape->getShapeVectors();
    for (Eigen::Index iBS = 0; iBS < parameterTransform_.faceExpressionParameters.size(); ++iBS) {
      const auto paramIdx = parameterTransform_.faceExpressionParameters[iBS];
      if (paramIdx < 0) {
        continue;
      }

      Eigen::Vector3<T> dRestSumDBs = Eigen::Vector3<T>::Zero();

      for (size_t i = 0; i < adjacentFaces.size(); ++i) {
        const auto& fe = faceEdges[i];

        // In rest space, d(restVertex)/d(bsWeight) = shapeVector (no skinning)
        const Eigen::Vector3<T> sv0 =
            shapeVectors.template block<3, 1>(3 * fe.indices[0], iBS, 3, 1).template cast<T>();
        const Eigen::Vector3<T> sv1 =
            shapeVectors.template block<3, 1>(3 * fe.indices[1], iBS, 3, 1).template cast<T>();
        const Eigen::Vector3<T> sv2 =
            shapeVectors.template block<3, 1>(3 * fe.indices[2], iBS, 3, 1).template cast<T>();

        const Eigen::Vector3<T> de1 = sv1 - sv0;
        const Eigen::Vector3<T> de2 = sv2 - sv0;

        dRestSumDBs += de1.cross(fe.e2) + fe.e1.cross(de2);
      }

      const Eigen::Vector3<T> dNormalDBs = dnormMv * dRestSumDBs;
      gradient[paramIdx] += scale * leverArm.dot(dNormalDBs);
    }
  }
}

// accumulateVertexNormalBlendShapeJacobian implementation
template <typename T>
void SkeletonDerivativeT<T>::accumulateVertexNormalBlendShapeJacobian(
    size_t vertexIndex,
    const Eigen::Vector3<T>& leverArm,
    T scale,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    size_t rowIndex) const {
  if (!character.blendShape && !character.faceExpressionBlendShape) {
    return;
  }

  MT_CHECK(meshState.posedMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);

  const auto& faces = meshState.posedMesh_->faces;

  // Find faces adjacent to this vertex via linear scan (fallback path)
  std::vector<size_t> adjacentFaces;
  for (size_t fIdx = 0; fIdx < faces.size(); ++fIdx) {
    const auto& face = faces[fIdx];
    if (face[0] == static_cast<int>(vertexIndex) || face[1] == static_cast<int>(vertexIndex) ||
        face[2] == static_cast<int>(vertexIndex)) {
      adjacentFaces.push_back(fIdx);
    }
  }

  accumulateVertexNormalBlendShapeJacobian(
      vertexIndex,
      leverArm,
      scale,
      adjacentFaces,
      skeletonState,
      meshState,
      character,
      jacobian,
      rowIndex);
}

// accumulateVertexNormalBlendShapeJacobian with pre-built adjacency
template <typename T>
void SkeletonDerivativeT<T>::accumulateVertexNormalBlendShapeJacobian(
    size_t vertexIndex,
    const Eigen::Vector3<T>& leverArm,
    T scale,
    std::span<const size_t> adjacentFaces,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    size_t rowIndex) const {
  if (!character.blendShape && !character.faceExpressionBlendShape) {
    return;
  }

  MT_CHECK(meshState.posedMesh_ != nullptr);
  MT_CHECK(meshState.restMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);
  MT_CHECK(!character.inverseBindPose.empty());

  const auto& faces = meshState.posedMesh_->faces;
  const auto& restVertices = meshState.restMesh_->vertices;

  if (adjacentFaces.empty()) {
    return;
  }

  // The forward path computes posed normals as:
  //   restNormal = normalize(Σ_f restFaceNormal_f)        [updateNormals on rest mesh]
  //   posedNormal = normalize(M_v * restNormal)            [applySSD skins + normalizes]
  //
  // Since normalize(M_v * normalize(s)) = normalize(M_v * s), the derivative simplifies to:
  //   d(posedNormal)/d(s) = [(I - n̂*n̂^T) / |M_v * s|] * M_v
  // where s = restSum, n̂ = posedNormal.

  // Step 1: Compute rest-space edge vectors and unnormalized sum
  Eigen::Vector3<T> restSum = Eigen::Vector3<T>::Zero();
  struct FaceEdges {
    Eigen::Vector3<T> e1;
    Eigen::Vector3<T> e2;
    Eigen::Vector3i indices;
  };
  std::vector<FaceEdges> faceEdges(adjacentFaces.size());

  for (size_t i = 0; i < adjacentFaces.size(); ++i) {
    const auto& face = faces[adjacentFaces[i]];
    const Eigen::Vector3<T> v0 = restVertices[face[0]].template cast<T>();
    const Eigen::Vector3<T> v1 = restVertices[face[1]].template cast<T>();
    const Eigen::Vector3<T> v2 = restVertices[face[2]].template cast<T>();
    faceEdges[i].e1 = v1 - v0;
    faceEdges[i].e2 = v2 - v0;
    faceEdges[i].indices = face;
    restSum += faceEdges[i].e1.cross(faceEdges[i].e2);
  }

  if (restSum.squaredNorm() < std::numeric_limits<T>::epsilon()) {
    return;
  }

  // Step 2: Compute skinning rotation M_v for the constraint vertex
  const auto& skinWeights = *character.skinWeights;
  Eigen::Matrix3<T> Mv;
  Mv.setZero();
  for (uint32_t jWeight = 0; jWeight < kMaxSkinJoints; ++jWeight) {
    const auto w = skinWeights.weight(vertexIndex, jWeight);
    if (w > T(0)) {
      const auto parentBone = skinWeights.index(vertexIndex, jWeight);
      Mv.noalias() += w *
          (skeletonState.jointState[parentBone].transform.toLinear() *
           character.inverseBindPose[parentBone].linear().template cast<T>());
    }
  }

  // Step 3: Compute posed unnormalized = M_v * restSum and normalization derivative
  const Eigen::Vector3<T> posedUnnorm = Mv * restSum;
  const T posedNorm = posedUnnorm.norm();
  if (posedNorm < T(1e-12)) {
    return;
  }
  const Eigen::Vector3<T> posedNhat = posedUnnorm / posedNorm;

  // d(posedNormal)/d(restSum) = [(I - n̂*n̂^T) / |M_v * restSum|] * M_v
  const Eigen::Matrix3<T> dnormMv =
      ((Eigen::Matrix3<T>::Identity() - posedNhat * posedNhat.transpose()) / posedNorm) * Mv;

  // Precompute leverArm^T * dnormMv for efficiency
  const Eigen::RowVector3<T> leverDnormMv = leverArm.transpose() * dnormMv;

  // Process standard blend shapes
  if (character.blendShape) {
    const auto& shapeVectors = character.blendShape->getShapeVectors();
    for (Eigen::Index iBS = 0; iBS < parameterTransform_.blendShapeParameters.size(); ++iBS) {
      const auto paramIdx = parameterTransform_.blendShapeParameters[iBS];
      if (paramIdx < 0) {
        continue;
      }

      Eigen::Vector3<T> dRestSumDBs = Eigen::Vector3<T>::Zero();

      for (size_t i = 0; i < adjacentFaces.size(); ++i) {
        const auto& fe = faceEdges[i];

        const Eigen::Vector3<T> sv0 =
            shapeVectors.template block<3, 1>(3 * fe.indices[0], iBS, 3, 1).template cast<T>();
        const Eigen::Vector3<T> sv1 =
            shapeVectors.template block<3, 1>(3 * fe.indices[1], iBS, 3, 1).template cast<T>();
        const Eigen::Vector3<T> sv2 =
            shapeVectors.template block<3, 1>(3 * fe.indices[2], iBS, 3, 1).template cast<T>();

        const Eigen::Vector3<T> de1 = sv1 - sv0;
        const Eigen::Vector3<T> de2 = sv2 - sv0;

        dRestSumDBs += de1.cross(fe.e2) + fe.e1.cross(de2);
      }

      jacobian(rowIndex, paramIdx) += scale * leverDnormMv.dot(dRestSumDBs);
    }
  }

  // Process face expression blend shapes
  if (character.faceExpressionBlendShape) {
    const auto& shapeVectors = character.faceExpressionBlendShape->getShapeVectors();
    for (Eigen::Index iBS = 0; iBS < parameterTransform_.faceExpressionParameters.size(); ++iBS) {
      const auto paramIdx = parameterTransform_.faceExpressionParameters[iBS];
      if (paramIdx < 0) {
        continue;
      }

      Eigen::Vector3<T> dRestSumDBs = Eigen::Vector3<T>::Zero();

      for (size_t i = 0; i < adjacentFaces.size(); ++i) {
        const auto& fe = faceEdges[i];

        const Eigen::Vector3<T> sv0 =
            shapeVectors.template block<3, 1>(3 * fe.indices[0], iBS, 3, 1).template cast<T>();
        const Eigen::Vector3<T> sv1 =
            shapeVectors.template block<3, 1>(3 * fe.indices[1], iBS, 3, 1).template cast<T>();
        const Eigen::Vector3<T> sv2 =
            shapeVectors.template block<3, 1>(3 * fe.indices[2], iBS, 3, 1).template cast<T>();

        const Eigen::Vector3<T> de1 = sv1 - sv0;
        const Eigen::Vector3<T> de2 = sv2 - sv0;

        dRestSumDBs += de1.cross(fe.e2) + fe.e1.cross(de2);
      }

      jacobian(rowIndex, paramIdx) += scale * leverDnormMv.dot(dRestSumDBs);
    }
  }
}

// accumulateVertexGradientWithNormalCorrection implementation
template <typename T>
template <size_t FuncDim>
void SkeletonDerivativeT<T>::accumulateVertexGradientWithNormalCorrection(
    size_t vertexIndex,
    const Eigen::Vector3<T>& /* worldPos */,
    const Eigen::Matrix<T, FuncDim, 3>& dfdv,
    const Eigen::Matrix<T, FuncDim, 1>& weightedResidual,
    const Eigen::Vector3<T>& sourceNormal,
    const Eigen::Vector3<T>& targetPosition,
    T normalCorrectionScale,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  static_assert(FuncDim == 1, "Normal correction assumes FuncDim == 1 (scalar residual)");
  // Skip if derivative is zero (normal correction still needs to run, so only
  // skip both if dfdv is zero AND normalCorrectionScale is zero)
  const bool hasDfdv = !dfdv.isZero();
  if (!hasDfdv && normalCorrectionScale == T(0)) {
    return;
  }

  MT_CHECK(meshState.restMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);
  MT_CHECK(!character.inverseBindPose.empty());

  // Single skinning walk combining vertex gradient + normal correction
  SkinningWeightIteratorT<T> skinningIter(
      character, *meshState.restMesh_, skeletonState, vertexIndex);

  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    MT_CHECK(jointIndex < skeleton_.joints.size());

    const auto& jointState = skeletonState.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> offset = pos - jointState.translation();

    // Translation DOFs (from standard vertex gradient only)
    if (hasDfdv) {
      for (size_t d = 0; d < 3; ++d) {
        if (!activeJointParams_[paramIndex + d]) {
          continue;
        }

        const T val =
            boneWeight * weightedResidual.dot(dfdv * jointState.getTranslationDerivative(d));

        for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
             index < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
             ++index) {
          if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
            gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
    }

    // Rotation DOFs: COMBINED standard vertex + normal correction
    // Normal correction cross product (computed once per bone)
    const Eigen::Vector3<T> cross = (targetPosition - pos).cross(sourceNormal);

    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + 3 + d]) {
        continue;
      }

      T val = T(0);
      if (hasDfdv) {
        val +=
            boneWeight * weightedResidual.dot(dfdv * jointState.getRotationDerivative(d, offset));
      }
      val += boneWeight * normalCorrectionScale * jointState.rotationAxis.col(d).dot(cross);

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }

    // Scale DOF (from standard vertex gradient only)
    if (hasDfdv) {
      if (activeJointParams_[paramIndex + 6]) {
        const T val =
            boneWeight * weightedResidual.dot(dfdv * jointState.getScaleDerivative(offset));

        for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
             index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
             ++index) {
          if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
            gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
    }
  }

  if (!hasDfdv) {
    return;
  }

  // Handle blend shape parameters (from standard vertex gradient)
  if (character.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.blendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      gradient[paramIdx] += weightedResidual.dot(dfdv * d_worldPos);
    }
  }

  // Handle face expression blend shape parameters (from standard vertex gradient)
  if (character.faceExpressionBlendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.faceExpressionParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.faceExpressionParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.faceExpressionBlendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      gradient[paramIdx] += weightedResidual.dot(dfdv * d_worldPos);
    }
  }
}

// accumulateVertexJacobianWithNormalCorrection implementation
template <typename T>
template <size_t FuncDim>
void SkeletonDerivativeT<T>::accumulateVertexJacobianWithNormalCorrection(
    size_t vertexIndex,
    const Eigen::Vector3<T>& /* worldPos */,
    const Eigen::Matrix<T, FuncDim, 3>& dfdv,
    T scale,
    const Eigen::Vector3<T>& sourceNormal,
    const Eigen::Vector3<T>& targetPosition,
    T normalCorrectionScale,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    size_t rowIndex) const {
  static_assert(FuncDim == 1, "Normal correction assumes FuncDim == 1 (scalar residual)");
  const bool hasDfdv = !dfdv.isZero();
  if (!hasDfdv && normalCorrectionScale == T(0)) {
    return;
  }

  MT_CHECK(meshState.restMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);
  MT_CHECK(!character.inverseBindPose.empty());

  auto jac = jacobian.middleRows(rowIndex, FuncDim);

  // Single skinning walk combining vertex Jacobian + normal correction
  SkinningWeightIteratorT<T> skinningIter(
      character, *meshState.restMesh_, skeletonState, vertexIndex);

  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    MT_CHECK(jointIndex < skeleton_.joints.size());

    const auto& jointState = skeletonState.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> offset = pos - jointState.translation();
    const T scaledBoneWeight = scale * boneWeight;

    // Translation DOFs (from standard vertex Jacobian only)
    if (hasDfdv) {
      for (size_t d = 0; d < 3; ++d) {
        if (!activeJointParams_[paramIndex + d]) {
          continue;
        }

        const Eigen::Matrix<T, FuncDim, 1> jc =
            scaledBoneWeight * dfdv * jointState.getTranslationDerivative(d);

        for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
             index < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
             ++index) {
          if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
            jac.col(parameterTransform_.transform.innerIndexPtr()[index]).noalias() +=
                jc * parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
    }

    // Rotation DOFs: COMBINED standard vertex + normal correction
    const Eigen::Vector3<T> cross = (targetPosition - pos).cross(sourceNormal);

    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + 3 + d]) {
        continue;
      }

      Eigen::Matrix<T, FuncDim, 1> jc;
      if (hasDfdv) {
        jc = scaledBoneWeight * dfdv * jointState.getRotationDerivative(d, offset);
      } else {
        jc.setZero();
      }
      const T normalVal =
          boneWeight * normalCorrectionScale * jointState.rotationAxis.col(d).dot(cross);
      jc[0] += normalVal;

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          jac.col(parameterTransform_.transform.innerIndexPtr()[index]).noalias() +=
              jc * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }

    // Scale DOF (from standard vertex Jacobian only)
    if (hasDfdv) {
      if (activeJointParams_[paramIndex + 6]) {
        const Eigen::Matrix<T, FuncDim, 1> jc =
            scaledBoneWeight * dfdv * jointState.getScaleDerivative(offset);

        for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
             index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
             ++index) {
          if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
            jac.col(parameterTransform_.transform.innerIndexPtr()[index]).noalias() +=
                jc * parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
    }
  }

  if (!hasDfdv) {
    return;
  }

  // Handle blend shape parameters (from standard vertex Jacobian)
  if (character.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.blendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      jac.col(paramIdx).noalias() += scale * dfdv * d_worldPos;
    }
  }

  // Handle face expression blend shape parameters (from standard vertex Jacobian)
  if (character.faceExpressionBlendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.faceExpressionParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.faceExpressionParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.faceExpressionBlendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      jac.col(paramIdx).noalias() += scale * dfdv * d_worldPos;
    }
  }
}

// accumulateVertexGradientWithNormalCorrectionDirect implementation
// Scalar-specialized for FuncDim=1: uses Vector3.dot() instead of Matrix<1,3> * Vector3
template <typename T>
void SkeletonDerivativeT<T>::accumulateVertexGradientWithNormalCorrectionDirect(
    size_t vertexIndex,
    const Eigen::Vector3<T>& /* worldPos */,
    const Eigen::Vector3<T>& normal,
    T weightedResidual,
    const Eigen::Vector3<T>& sourceNormal,
    const Eigen::Vector3<T>& targetPosition,
    T normalCorrectionScale,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  const bool hasNormal = !normal.isZero();
  if (!hasNormal && normalCorrectionScale == T(0)) {
    return;
  }

  MT_CHECK(meshState.restMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);
  MT_CHECK(!character.inverseBindPose.empty());

  SkinningWeightIteratorT<T> skinningIter(
      character, *meshState.restMesh_, skeletonState, vertexIndex);

  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    MT_CHECK(jointIndex < skeleton_.joints.size());

    const auto& jointState = skeletonState.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> offset = pos - jointState.translation();

    // Translation DOFs (scalar dot product instead of Matrix<1,3> * Vector3)
    if (hasNormal) {
      for (size_t d = 0; d < 3; ++d) {
        if (!activeJointParams_[paramIndex + d]) {
          continue;
        }

        const T val =
            boneWeight * weightedResidual * normal.dot(jointState.getTranslationDerivative(d));

        for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
             index < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
             ++index) {
          if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
            gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
    }

    // Rotation DOFs: COMBINED standard vertex + normal correction
    const Eigen::Vector3<T> cross = (targetPosition - pos).cross(sourceNormal);

    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + 3 + d]) {
        continue;
      }

      T val = T(0);
      if (hasNormal) {
        val +=
            boneWeight * weightedResidual * normal.dot(jointState.getRotationDerivative(d, offset));
      }
      val += boneWeight * normalCorrectionScale * jointState.rotationAxis.col(d).dot(cross);

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
              val * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }

    // Scale DOF (scalar dot product instead of Matrix<1,3> * Vector3)
    if (hasNormal) {
      if (activeJointParams_[paramIndex + 6]) {
        const T val =
            boneWeight * weightedResidual * normal.dot(jointState.getScaleDerivative(offset));

        for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
             index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
             ++index) {
          if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
            gradient[parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
    }
  }

  if (!hasNormal) {
    return;
  }

  // Handle blend shape parameters (scalar dot product)
  if (character.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.blendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      gradient[paramIdx] += weightedResidual * normal.dot(d_worldPos);
    }
  }

  // Handle face expression blend shape parameters (scalar dot product)
  if (character.faceExpressionBlendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.faceExpressionParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.faceExpressionParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.faceExpressionBlendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      gradient[paramIdx] += weightedResidual * normal.dot(d_worldPos);
    }
  }
}

// accumulateVertexJacobianWithNormalCorrectionDirect implementation
// Scalar-specialized for FuncDim=1: uses Vector3.dot() instead of Matrix<1,3> * Vector3
template <typename T>
void SkeletonDerivativeT<T>::accumulateVertexJacobianWithNormalCorrectionDirect(
    size_t vertexIndex,
    const Eigen::Vector3<T>& /* worldPos */,
    const Eigen::Vector3<T>& normal,
    T scale,
    const Eigen::Vector3<T>& sourceNormal,
    const Eigen::Vector3<T>& targetPosition,
    T normalCorrectionScale,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    const Character& character,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    size_t rowIndex) const {
  const bool hasNormal = !normal.isZero();
  if (!hasNormal && normalCorrectionScale == T(0)) {
    return;
  }

  MT_CHECK(meshState.restMesh_ != nullptr);
  MT_CHECK(character.skinWeights != nullptr);
  MT_CHECK(!character.inverseBindPose.empty());

  SkinningWeightIteratorT<T> skinningIter(
      character, *meshState.restMesh_, skeletonState, vertexIndex);

  while (!skinningIter.finished()) {
    size_t jointIndex = 0;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    MT_CHECK(jointIndex < skeleton_.joints.size());

    const auto& jointState = skeletonState.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> offset = pos - jointState.translation();
    const T scaledBoneWeight = scale * boneWeight;

    // Translation DOFs (scalar dot product instead of Matrix<1,3> * Vector3)
    if (hasNormal) {
      for (size_t d = 0; d < 3; ++d) {
        if (!activeJointParams_[paramIndex + d]) {
          continue;
        }

        const T jc = scaledBoneWeight * normal.dot(jointState.getTranslationDerivative(d));

        for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
             index < parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
             ++index) {
          if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
            jacobian(rowIndex, parameterTransform_.transform.innerIndexPtr()[index]) +=
                jc * parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
    }

    // Rotation DOFs: COMBINED standard vertex + normal correction
    const Eigen::Vector3<T> cross = (targetPosition - pos).cross(sourceNormal);

    for (size_t d = 0; d < 3; ++d) {
      if (!activeJointParams_[paramIndex + 3 + d]) {
        continue;
      }

      T jc = T(0);
      if (hasNormal) {
        jc += scaledBoneWeight * normal.dot(jointState.getRotationDerivative(d, offset));
      }
      jc += boneWeight * normalCorrectionScale * jointState.rotationAxis.col(d).dot(cross);

      for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
           index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
           ++index) {
        if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
          jacobian(rowIndex, parameterTransform_.transform.innerIndexPtr()[index]) +=
              jc * parameterTransform_.transform.valuePtr()[index];
        }
      }
    }

    // Scale DOF (scalar dot product instead of Matrix<1,3> * Vector3)
    if (hasNormal) {
      if (activeJointParams_[paramIndex + 6]) {
        const T jc = scaledBoneWeight * normal.dot(jointState.getScaleDerivative(offset));

        for (auto index = parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
             index < parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
             ++index) {
          if (enabledParameters_.test(parameterTransform_.transform.innerIndexPtr()[index])) {
            jacobian(rowIndex, parameterTransform_.transform.innerIndexPtr()[index]) +=
                jc * parameterTransform_.transform.valuePtr()[index];
          }
        }
      }
    }
  }

  if (!hasNormal) {
    return;
  }

  // Handle blend shape parameters (scalar dot product)
  if (character.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.blendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      jacobian(rowIndex, paramIdx) += scale * normal.dot(d_worldPos);
    }
  }

  // Handle face expression blend shape parameters (scalar dot product)
  if (character.faceExpressionBlendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < parameterTransform_.faceExpressionParameters.size();
         ++iBlendShape) {
      const auto paramIdx = parameterTransform_.faceExpressionParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          character.faceExpressionBlendShape->getShapeVectors()
              .template block<3, 1>(3 * vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();

      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      const auto& skinWeights = *character.skinWeights;
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          d_worldPos += w *
              (skeletonState.jointState[parentBone].transform.toLinear() *
               (character.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      jacobian(rowIndex, paramIdx) += scale * normal.dot(d_worldPos);
    }
  }
}

// Explicit template instantiations for accumulateVertexGradientWithNormalCorrectionDirect
template void SkeletonDerivativeT<float>::accumulateVertexGradientWithNormalCorrectionDirect(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Vector3<float>&,
    float,
    const Eigen::Vector3<float>&,
    const Eigen::Vector3<float>&,
    float,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::VectorX<float>>) const;
template void SkeletonDerivativeT<double>::accumulateVertexGradientWithNormalCorrectionDirect(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Vector3<double>&,
    double,
    const Eigen::Vector3<double>&,
    const Eigen::Vector3<double>&,
    double,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::VectorX<double>>) const;

// Explicit template instantiations for accumulateVertexJacobianWithNormalCorrectionDirect
template void SkeletonDerivativeT<float>::accumulateVertexJacobianWithNormalCorrectionDirect(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Vector3<float>&,
    float,
    const Eigen::Vector3<float>&,
    const Eigen::Vector3<float>&,
    float,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixX<float>>,
    size_t) const;
template void SkeletonDerivativeT<double>::accumulateVertexJacobianWithNormalCorrectionDirect(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Vector3<double>&,
    double,
    const Eigen::Vector3<double>&,
    const Eigen::Vector3<double>&,
    double,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixX<double>>,
    size_t) const;

// Explicit template instantiations for accumulateVertexGradientWithNormalCorrection
template void SkeletonDerivativeT<float>::accumulateVertexGradientWithNormalCorrection<1>(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Matrix<float, 1, 3>&,
    const Eigen::Matrix<float, 1, 1>&,
    const Eigen::Vector3<float>&,
    const Eigen::Vector3<float>&,
    float,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::VectorX<float>>) const;
template void SkeletonDerivativeT<double>::accumulateVertexGradientWithNormalCorrection<1>(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Matrix<double, 1, 3>&,
    const Eigen::Matrix<double, 1, 1>&,
    const Eigen::Vector3<double>&,
    const Eigen::Vector3<double>&,
    double,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::VectorX<double>>) const;

// Explicit template instantiations for accumulateVertexJacobianWithNormalCorrection
template void SkeletonDerivativeT<float>::accumulateVertexJacobianWithNormalCorrection<1>(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Matrix<float, 1, 3>&,
    float,
    const Eigen::Vector3<float>&,
    const Eigen::Vector3<float>&,
    float,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixX<float>>,
    size_t) const;
template void SkeletonDerivativeT<double>::accumulateVertexJacobianWithNormalCorrection<1>(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Matrix<double, 1, 3>&,
    double,
    const Eigen::Vector3<double>&,
    const Eigen::Vector3<double>&,
    double,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixX<double>>,
    size_t) const;

// Explicit template instantiations for accumulateVertexNormalCorrection
template void SkeletonDerivativeT<float>::accumulateVertexNormalCorrectionGradient(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Vector3<float>&,
    float,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::VectorXf>) const;
template void SkeletonDerivativeT<double>::accumulateVertexNormalCorrectionGradient(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Vector3<double>&,
    double,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::VectorXd>) const;
template void SkeletonDerivativeT<float>::accumulateVertexNormalCorrectionJacobian(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Vector3<float>&,
    float,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixXf>,
    size_t) const;
template void SkeletonDerivativeT<double>::accumulateVertexNormalCorrectionJacobian(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Vector3<double>&,
    double,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixXd>,
    size_t) const;

// Explicit template instantiations for accumulateVertexNormalBlendShape
template void SkeletonDerivativeT<float>::accumulateVertexNormalBlendShapeGradient(
    size_t,
    const Eigen::Vector3<float>&,
    float,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::VectorXf>) const;
template void SkeletonDerivativeT<double>::accumulateVertexNormalBlendShapeGradient(
    size_t,
    const Eigen::Vector3<double>&,
    double,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::VectorXd>) const;
template void SkeletonDerivativeT<float>::accumulateVertexNormalBlendShapeGradient(
    size_t,
    const Eigen::Vector3<float>&,
    float,
    std::span<const size_t>,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::VectorXf>) const;
template void SkeletonDerivativeT<double>::accumulateVertexNormalBlendShapeGradient(
    size_t,
    const Eigen::Vector3<double>&,
    double,
    std::span<const size_t>,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::VectorXd>) const;
template void SkeletonDerivativeT<float>::accumulateVertexNormalBlendShapeJacobian(
    size_t,
    const Eigen::Vector3<float>&,
    float,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixXf>,
    size_t) const;
template void SkeletonDerivativeT<double>::accumulateVertexNormalBlendShapeJacobian(
    size_t,
    const Eigen::Vector3<double>&,
    double,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixXd>,
    size_t) const;
template void SkeletonDerivativeT<float>::accumulateVertexNormalBlendShapeJacobian(
    size_t,
    const Eigen::Vector3<float>&,
    float,
    std::span<const size_t>,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixXf>,
    size_t) const;
template void SkeletonDerivativeT<double>::accumulateVertexNormalBlendShapeJacobian(
    size_t,
    const Eigen::Vector3<double>&,
    double,
    std::span<const size_t>,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixXd>,
    size_t) const;

template void SkeletonDerivativeT<float>::accumulateVertexGradient<1>(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Matrix<float, 1, 3>&,
    const Eigen::Matrix<float, 1, 1>&,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::VectorX<float>>) const;
template void SkeletonDerivativeT<float>::accumulateVertexGradient<2>(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Matrix<float, 2, 3>&,
    const Eigen::Matrix<float, 2, 1>&,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::VectorX<float>>) const;
template void SkeletonDerivativeT<float>::accumulateVertexGradient<3>(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Matrix<float, 3, 3>&,
    const Eigen::Matrix<float, 3, 1>&,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::VectorX<float>>) const;
template void SkeletonDerivativeT<float>::accumulateVertexGradient<9>(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Matrix<float, 9, 3>&,
    const Eigen::Matrix<float, 9, 1>&,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::VectorX<float>>) const;

// Explicit template instantiations for accumulateVertexJacobian (float)
template void SkeletonDerivativeT<float>::accumulateVertexJacobian<1>(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Matrix<float, 1, 3>&,
    float,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixX<float>>,
    size_t) const;
template void SkeletonDerivativeT<float>::accumulateVertexJacobian<2>(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Matrix<float, 2, 3>&,
    float,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixX<float>>,
    size_t) const;
template void SkeletonDerivativeT<float>::accumulateVertexJacobian<3>(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Matrix<float, 3, 3>&,
    float,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixX<float>>,
    size_t) const;
template void SkeletonDerivativeT<float>::accumulateVertexJacobian<9>(
    size_t,
    const Eigen::Vector3<float>&,
    const Eigen::Matrix<float, 9, 3>&,
    float,
    const SkeletonStateT<float>&,
    const MeshStateT<float>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixX<float>>,
    size_t) const;

// Explicit template instantiations for accumulateVertexGradient (double)
template void SkeletonDerivativeT<double>::accumulateVertexGradient<1>(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Matrix<double, 1, 3>&,
    const Eigen::Matrix<double, 1, 1>&,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::VectorX<double>>) const;
template void SkeletonDerivativeT<double>::accumulateVertexGradient<2>(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Matrix<double, 2, 3>&,
    const Eigen::Matrix<double, 2, 1>&,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::VectorX<double>>) const;
template void SkeletonDerivativeT<double>::accumulateVertexGradient<3>(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Matrix<double, 3, 3>&,
    const Eigen::Matrix<double, 3, 1>&,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::VectorX<double>>) const;
template void SkeletonDerivativeT<double>::accumulateVertexGradient<9>(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Matrix<double, 9, 3>&,
    const Eigen::Matrix<double, 9, 1>&,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::VectorX<double>>) const;

// Explicit template instantiations for accumulateVertexJacobian (double)
template void SkeletonDerivativeT<double>::accumulateVertexJacobian<1>(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Matrix<double, 1, 3>&,
    double,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixX<double>>,
    size_t) const;
template void SkeletonDerivativeT<double>::accumulateVertexJacobian<2>(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Matrix<double, 2, 3>&,
    double,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixX<double>>,
    size_t) const;
template void SkeletonDerivativeT<double>::accumulateVertexJacobian<3>(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Matrix<double, 3, 3>&,
    double,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixX<double>>,
    size_t) const;
template void SkeletonDerivativeT<double>::accumulateVertexJacobian<9>(
    size_t,
    const Eigen::Vector3<double>&,
    const Eigen::Matrix<double, 9, 3>&,
    double,
    const SkeletonStateT<double>&,
    const MeshStateT<double>&,
    const Character&,
    Eigen::Ref<Eigen::MatrixX<double>>,
    size_t) const;
} // namespace momentum
