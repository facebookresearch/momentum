/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/skinned_locator_triangle_error_function.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/character_solver/skinning_weight_iterator.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/mesh.h"

#include <axel/math/PointTriangleProjectionDefinitions.h>

#include <memory>

namespace momentum {

template <typename T>
SkinnedLocatorTriangleErrorFunctionT<T>::SkinnedLocatorTriangleErrorFunctionT(
    const Character& character_in,
    VertexConstraintType type)
    : SkeletonErrorFunctionT<T>(character_in.skeleton, character_in.parameterTransform),
      character_(character_in),
      constraintType_(type) {
  MT_CHECK(static_cast<bool>(character_in.mesh));
  MT_THROW_IF(
      type != VertexConstraintType::Position && type != VertexConstraintType::Plane,
      "SkinnedLocatorTriangleErrorFunction only supports Position and Plane constraint types");
}

template <typename T>
SkinnedLocatorTriangleErrorFunctionT<T>::~SkinnedLocatorTriangleErrorFunctionT() = default;

template <typename T>
void SkinnedLocatorTriangleErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
void SkinnedLocatorTriangleErrorFunctionT<T>::setConstraints(
    const std::vector<SkinnedLocatorTriangleConstraintT<T>>& constraints) {
  constraints_ = constraints;
}

template <typename T>
void SkinnedLocatorTriangleErrorFunctionT<T>::addConstraint(
    int locatorIndex,
    const Eigen::Vector3i& triangleIndices,
    const Eigen::Vector3<T>& triangleBaryCoords,
    float depth,
    T weight) {
  MT_CHECK(locatorIndex >= 0 && ((size_t)locatorIndex) < character_.skinnedLocators.size());
  constraints_.push_back(
      SkinnedLocatorTriangleConstraintT<T>{
          locatorIndex, triangleIndices, triangleBaryCoords, depth, weight, {}});
}

template <typename T>
Eigen::Vector3<T> computeTargetBaryPosition(
    const MeshT<T>& mesh,
    const SkinnedLocatorTriangleConstraintT<T>& c) {
  return c.tgtTriangleBaryCoords[0] * mesh.vertices[c.tgtTriangleIndices[0]] +
      c.tgtTriangleBaryCoords[1] * mesh.vertices[c.tgtTriangleIndices[1]] +
      c.tgtTriangleBaryCoords[2] * mesh.vertices[c.tgtTriangleIndices[2]];
}

template <typename T>
Eigen::Vector3<T> computeTargetTriangleNormal(
    const MeshT<T>& mesh,
    const SkinnedLocatorTriangleConstraintT<T>& c) {
  return (mesh.vertices[c.tgtTriangleIndices[1]] - mesh.vertices[c.tgtTriangleIndices[0]])
      .cross(mesh.vertices[c.tgtTriangleIndices[2]] - mesh.vertices[c.tgtTriangleIndices[0]])
      .normalized();
}

template <typename T>
Eigen::Vector3<T> computeTargetPosition(
    const MeshT<T>& mesh,
    const SkinnedLocatorTriangleConstraintT<T>& c) {
  return computeTargetBaryPosition(mesh, c) + c.depth * computeTargetTriangleNormal(mesh, c);
}

namespace {

// Helper struct for reprojection result
template <typename T>
struct ReprojectionResult {
  Eigen::Vector3i triangleIndices;
  Eigen::Vector3<T> baryCoords;
  Eigen::Vector3<T> closestPoint;
};

// Find the closest candidate triangle and compute reprojection
template <typename T>
ReprojectionResult<T> findClosestCandidateTriangle(
    const MeshT<T>& mesh,
    const Eigen::Vector3<T>& locatorPos,
    const std::vector<CandidateTriangle>& candidates,
    T depth) {
  ReprojectionResult<T> result;
  T minDistSq = std::numeric_limits<T>::max();

  for (const auto& candidate : candidates) {
    const Eigen::Vector3<T>& v0 = mesh.vertices[candidate.vertexIndices[0]];
    const Eigen::Vector3<T>& v1 = mesh.vertices[candidate.vertexIndices[1]];
    const Eigen::Vector3<T>& v2 = mesh.vertices[candidate.vertexIndices[2]];

    Eigen::Vector3<T> projPoint;
    Eigen::Vector3<T> bary;
    axel::projectOnTriangle(locatorPos, v0, v1, v2, projPoint, &bary);

    // Compute normal and apply depth offset
    Eigen::Vector3<T> normal = (v1 - v0).cross(v2 - v0);
    const T normalNorm = normal.norm();
    if (normalNorm > T(1e-8)) {
      normal /= normalNorm;
    } else {
      normal.setZero();
    }
    const Eigen::Vector3<T> targetPoint = projPoint + depth * normal;

    const T distSq = (locatorPos - targetPoint).squaredNorm();
    if (distSq < minDistSq) {
      minDistSq = distSq;
      result.triangleIndices = candidate.vertexIndices;
      result.baryCoords = bary;
      result.closestPoint = targetPoint;
    }
  }

  return result;
}

// Helper to get effective constraint after potential reprojection
template <typename T>
SkinnedLocatorTriangleConstraintT<T> getEffectiveConstraint(
    const MeshT<T>& mesh,
    const SkinnedLocatorTriangleConstraintT<T>& constr,
    const Eigen::Vector3<T>& locatorPos) {
  if (constr.candidateTriangles.empty()) {
    return constr;
  }

  // Find closest candidate triangle
  ReprojectionResult<T> reproj =
      findClosestCandidateTriangle(mesh, locatorPos, constr.candidateTriangles, constr.depth);

  // Create effective constraint with reprojected triangle
  SkinnedLocatorTriangleConstraintT<T> effective;
  effective.locatorIndex = constr.locatorIndex;
  effective.tgtTriangleIndices = reproj.triangleIndices;
  effective.tgtTriangleBaryCoords = reproj.baryCoords;
  effective.depth = constr.depth;
  effective.weight = constr.weight;
  // Don't copy candidateTriangles - effective constraint is for single evaluation
  return effective;
}

} // namespace

template <typename T>
Eigen::Vector3<T> SkinnedLocatorTriangleErrorFunctionT<T>::getLocatorRestPosition(
    const ModelParametersT<T>& modelParams,
    int locatorIndex) const {
  MT_CHECK(locatorIndex >= 0 && locatorIndex < static_cast<int>(character_.skinnedLocators.size()));
  const auto& locator = character_.skinnedLocators[locatorIndex];

  Vector3<T> result = locator.position.template cast<T>();
  int locatorParameterIndex = getSkinnedLocatorParameterIndex(locatorIndex);
  if (locatorParameterIndex >= 0 && locatorParameterIndex + 2 < modelParams.size()) {
    result += modelParams.v.template segment<3>(locatorParameterIndex);
  }

  return result;
}

template <typename T>
std::array<Eigen::Matrix3<T>, 3> compute_d_targetNormal_d_vertexPos(
    const SkinnedLocatorTriangleConstraintT<T>& cons,
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
    const SkinnedLocatorTriangleConstraintT<T>& cons,
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
double SkinnedLocatorTriangleErrorFunctionT<T>::getError(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& /* state */,
    const MeshStateT<T>& meshState) {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState.restMesh_);

  double error = 0.0;

  if (constraintType_ == VertexConstraintType::Position) {
    for (const auto& constr : constraints_) {
      // Get the skinned locator position
      const Eigen::Vector3<T> locatorRestPos =
          getLocatorRestPosition(modelParameters, constr.locatorIndex);

      // Get effective constraint (potentially reprojected onto closest candidate triangle)
      const SkinnedLocatorTriangleConstraintT<T> effectiveConstr =
          getEffectiveConstraint(*meshState.restMesh_, constr, locatorRestPos);

      // Get the target position on the triangle
      const Eigen::Vector3<T> tgtPoint =
          computeTargetPosition(*meshState.restMesh_, effectiveConstr);

      // Calculate position error
      const Eigen::Vector3<T> diff = locatorRestPos - tgtPoint;
      error += constr.weight * diff.squaredNorm();
    }
  } else {
    MT_CHECK(constraintType_ == VertexConstraintType::Plane);
    for (const auto& constr : constraints_) {
      // Get the skinned locator position
      const Eigen::Vector3<T> locatorRestPos =
          getLocatorRestPosition(modelParameters, constr.locatorIndex);

      // Get effective constraint (potentially reprojected onto closest candidate triangle)
      const SkinnedLocatorTriangleConstraintT<T> effectiveConstr =
          getEffectiveConstraint(*meshState.restMesh_, constr, locatorRestPos);

      // Get the target position and normal
      const Eigen::Vector3<T> targetNormal =
          computeTargetTriangleNormal(*meshState.restMesh_, effectiveConstr);
      const Eigen::Vector3<T> tgtPoint =
          computeTargetBaryPosition(*meshState.restMesh_, effectiveConstr) +
          effectiveConstr.depth * targetNormal;
      // Calculate plane error (projection onto normal)
      const T dist = targetNormal.dot(locatorRestPos - tgtPoint);
      error += constr.weight * dist * dist;
    }
  }

  return error * this->weight_;
}

template <typename T>
double SkinnedLocatorTriangleErrorFunctionT<T>::calculatePositionGradient(
    const ModelParametersT<T>& modelParameters,
    const MeshStateT<T>& meshState,
    const SkinnedLocatorTriangleConstraintT<T>& constr,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  // Get the skinned locator position
  const Eigen::Vector3<T> locatorRestPos =
      getLocatorRestPosition(modelParameters, constr.locatorIndex);

  // Get the target position on the triangle
  const Eigen::Vector3<T> tgtPoint = computeTargetPosition(*meshState.restMesh_, constr);

  // Calculate position error
  const Eigen::Vector3<T> diff = locatorRestPos - tgtPoint;

  // Calculate gradient weight
  const T wgt = 2.0f * constr.weight * this->weight_;

  // Get the derivative of the target position with respect to the triangle vertices
  const auto d_targetPos_d_tgtTriVertexPos =
      compute_d_targetPos_d_vertexPos(constr, *meshState.restMesh_);

  // Apply gradient for locator parameter if it exists
  int paramIndex = getSkinnedLocatorParameterIndex(constr.locatorIndex);
  if (paramIndex >= 0 && paramIndex + 2 < modelParameters.size()) {
    // Each skinned locator has 3 parameters (x, y, z)
    for (int d = 0; d < 3; ++d) {
      Eigen::Vector3<T> d_restPos = Eigen::Vector3<T>::Zero();
      d_restPos[d] = 1.0;

      gradient(paramIndex + d) += diff.dot(d_restPos) * wgt;
    }
  }

  // IN handle derivatives wrt blend shape parameters
  if (this->character_.blendShape) {
    for (int iTriVert = 0; iTriVert < 3; ++iTriVert) {
      const int vertexIndex = constr.tgtTriangleIndices[iTriVert];
      const Eigen::Matrix3<T>& d_targetPos_d_vertexPos = d_targetPos_d_tgtTriVertexPos[iTriVert];
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

        gradient[paramIdx] += -wgt * diff.dot(d_targetPos_d_vertexPos * d_restPos);
      }
    }
  }
  // OUT handle derivatives wrt blend shape parameters

  return constr.weight * diff.squaredNorm();
}

template <typename T>
double SkinnedLocatorTriangleErrorFunctionT<T>::calculatePlaneGradient(
    const ModelParametersT<T>& modelParameters,
    const MeshStateT<T>& meshState,
    const SkinnedLocatorTriangleConstraintT<T>& constr,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  // Get the skinned locator position
  const Eigen::Vector3<T> locatorRestPos =
      getLocatorRestPosition(modelParameters, constr.locatorIndex);

  // Get the target position and normal
  const Eigen::Vector3<T> targetNormal = computeTargetTriangleNormal(*meshState.restMesh_, constr);
  const Eigen::Vector3<T> tgtPoint =
      computeTargetBaryPosition(*meshState.restMesh_, constr) + constr.depth * targetNormal;

  // Calculate plane error (projection onto normal)
  const Vector3<T> diff = locatorRestPos - tgtPoint;
  const T dist = targetNormal.dot(diff);
  const T wgt = 2.0f * constr.weight * this->weight_;

  // Get the derivative of the target position with respect to the triangle vertices
  const auto d_targetPos_d_tgtTriVertexPos =
      compute_d_targetPos_d_vertexPos(constr, *meshState.restMesh_);
  const auto d_targetNormal_d_tgtTriVertexPos =
      compute_d_targetNormal_d_vertexPos(constr, *meshState.restMesh_);

  // Apply gradient for locator parameter if it exists
  int paramIndex = getSkinnedLocatorParameterIndex(constr.locatorIndex);
  if (paramIndex >= 0) {
    // Each skinned locator has 3 parameters (x, y, z)
    for (int k = 0; k < 3; ++k) {
      Eigen::Vector3<T> d_restPos = Eigen::Vector3<T>::Zero();
      d_restPos[k] = 1.0;
      gradient(paramIndex + k) += wgt * d_restPos.dot(targetNormal) * dist;
    }
  }

  // IN handle derivatives wrt blend shape parameters:
  for (int kTriVert = 0; kTriVert < 3; ++kTriVert) {
    const int vertexIndex = constr.tgtTriangleIndices[kTriVert];
    const Eigen::Matrix3<T>& d_targetPos_d_vertexPos = d_targetPos_d_tgtTriVertexPos[kTriVert];
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

      gradient(paramIdx) -= wgt * dist * targetNormal.dot(d_targetPos_d_vertexPos * d_restPos);
      const Eigen::Vector3<T> diff_tgt_normal =
          d_targetNormal_d_tgtTriVertexPos[kTriVert] * d_restPos;
      gradient(paramIdx) += wgt * dist * diff.dot(diff_tgt_normal);
    }
  }
  // OUT handle derivatives wrt blend shape parameters:

  return constr.weight * dist * dist;
}

template <typename T>
double SkinnedLocatorTriangleErrorFunctionT<T>::calculatePositionJacobian(
    const ModelParametersT<T>& modelParameters,
    const MeshStateT<T>& meshState,
    const SkinnedLocatorTriangleConstraintT<T>& constr,
    Ref<Eigen::MatrixX<T>> jac,
    Ref<Eigen::VectorX<T>> res) const {
  // Get the skinned locator position
  const Eigen::Vector3<T> locatorRestPos =
      getLocatorRestPosition(modelParameters, constr.locatorIndex);

  // Get the target position on the triangle
  const Eigen::Vector3<T> tgtPoint = computeTargetPosition(*meshState.restMesh_, constr);

  // Calculate position error
  const Eigen::Vector3<T> diff = locatorRestPos - tgtPoint;

  // Set residual
  const T wgt = std::sqrt(constr.weight * this->weight_);
  res = wgt * diff;

  // Get the derivative of the target position with respect to the triangle vertices
  const auto d_targetPos_d_tgtTriVertexPos =
      compute_d_targetPos_d_vertexPos(constr, *meshState.restMesh_);

  // Apply Jacobian for locator parameter if it exists
  int paramIndex = getSkinnedLocatorParameterIndex(constr.locatorIndex);
  if (paramIndex >= 0) {
    for (int k = 0; k < 3; ++k) {
      Eigen::Vector3<T> d_restPos = Eigen::Vector3<T>::Zero();
      d_restPos[k] = 1.0;
      jac.template block<3, 1>(0, paramIndex + k) += wgt * d_restPos;
    }
  }

  // IN handle derivatives wrt blend shape parameters:
  for (int iTriVert = 0; iTriVert < 3; ++iTriVert) {
    const int vertexIndex = constr.tgtTriangleIndices[iTriVert];
    const Eigen::Matrix3<T>& d_targetPos_d_vertexPos = d_targetPos_d_tgtTriVertexPos[iTriVert];

    // loop over blend shape parameters:
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

      jac.col(paramIdx) -= wgt * d_targetPos_d_vertexPos * d_restPos;
    }
  }
  // OUT handle derivatives wrt blend shape parameters:

  return wgt * wgt * diff.squaredNorm();
}

template <typename T>
double SkinnedLocatorTriangleErrorFunctionT<T>::calculatePlaneJacobian(
    const ModelParametersT<T>& modelParameters,
    const MeshStateT<T>& meshState,
    const SkinnedLocatorTriangleConstraintT<T>& constr,
    Ref<Eigen::MatrixX<T>> jac,
    T& res) const {
  // Get the skinned locator position
  const Eigen::Vector3<T> locatorRestPos =
      getLocatorRestPosition(modelParameters, constr.locatorIndex);

  // Get the target position and normal
  const Eigen::Vector3<T> targetNormal = computeTargetTriangleNormal(*meshState.restMesh_, constr);
  const Eigen::Vector3<T> tgtPoint =
      computeTargetBaryPosition(*meshState.restMesh_, constr) + constr.depth * targetNormal;

  // Calculate plane error (projection onto normal)
  const Eigen::Vector3<T> diff = locatorRestPos - tgtPoint;
  const T dist = targetNormal.dot(diff);

  const T wgt = std::sqrt(constr.weight * this->weight_);
  res = wgt * dist;

  // Get the derivative of the target position with respect to the triangle vertices
  const auto d_targetPos_d_tgtTriVertexPos =
      compute_d_targetPos_d_vertexPos(constr, *meshState.restMesh_);
  const auto d_targetNormal_d_tgtTriVertexPos =
      compute_d_targetNormal_d_vertexPos(constr, *meshState.restMesh_);

  // Apply Jacobian for locator parameter if it exists
  int paramIndex = getSkinnedLocatorParameterIndex(constr.locatorIndex);
  if (paramIndex >= 0) {
    for (int k = 0; k < 3; ++k) {
      Eigen::Vector3<T> d_restPos = Eigen::Vector3<T>::Zero();
      d_restPos[k] = 1.0;
      jac(0, paramIndex + k) += wgt * d_restPos.dot(targetNormal);
    }
  }

  // IN handle derivatives wrt blend shape parameters:
  for (int kTriVert = 0; kTriVert < 3; ++kTriVert) {
    const int vertexIndex = constr.tgtTriangleIndices[kTriVert];
    const Eigen::Matrix3<T>& d_targetPos_d_vertexPos = d_targetPos_d_tgtTriVertexPos[kTriVert];
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

      jac(0, paramIdx) -= wgt * targetNormal.dot(d_targetPos_d_vertexPos * d_restPos);
      const Eigen::Vector3<T> diff_tgt_normal =
          d_targetNormal_d_tgtTriVertexPos[kTriVert] * d_restPos;
      jac(0, paramIdx) += wgt * diff.dot(diff_tgt_normal);
    }
  }
  // OUT handle derivatives wrt blend shape parameters:

  return wgt * wgt * dist * dist;
}

template <typename T>
double SkinnedLocatorTriangleErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& /* state */,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_CHECK_NOTNULL(meshState.restMesh_);

  double error = 0;

  if (constraintType_ == VertexConstraintType::Position) {
    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      const SkinnedLocatorTriangleConstraintT<T>& constr = constraints_[iCons];
      // Get locator position for reprojection
      const Eigen::Vector3<T> locatorRestPos =
          getLocatorRestPosition(modelParameters, constr.locatorIndex);
      // Get effective constraint (potentially reprojected)
      const SkinnedLocatorTriangleConstraintT<T> effectiveConstr =
          getEffectiveConstraint(*meshState.restMesh_, constr, locatorRestPos);
      error += calculatePositionGradient(modelParameters, meshState, effectiveConstr, gradient);
    }
  } else {
    MT_CHECK(constraintType_ == VertexConstraintType::Plane);
    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      const SkinnedLocatorTriangleConstraintT<T>& constr = constraints_[iCons];
      // Get locator position for reprojection
      const Eigen::Vector3<T> locatorRestPos =
          getLocatorRestPosition(modelParameters, constr.locatorIndex);
      // Get effective constraint (potentially reprojected)
      const SkinnedLocatorTriangleConstraintT<T> effectiveConstr =
          getEffectiveConstraint(*meshState.restMesh_, constr, locatorRestPos);
      error += calculatePlaneGradient(modelParameters, meshState, effectiveConstr, gradient);
    }
  }

  return this->weight_ * error;
}

template <typename T>
double SkinnedLocatorTriangleErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& /* state */,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_CHECK(
      jacobian.cols() == static_cast<Eigen::Index>(this->parameterTransform_.transform.cols()));
  MT_CHECK(jacobian.rows() >= (Eigen::Index)(1 * constraints_.size()));
  MT_CHECK(residual.rows() >= (Eigen::Index)(1 * constraints_.size()));

  MT_CHECK(
      meshState.restMesh_, "MeshState must have rest mesh for SkinnedLocatorTriangleErrorFunction");

  double error = 0;

  if (constraintType_ == VertexConstraintType::Position) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      const SkinnedLocatorTriangleConstraintT<T>& constr = constraints_[i];
      // Get locator position for reprojection
      const Eigen::Vector3<T> locatorRestPos =
          getLocatorRestPosition(modelParameters, constr.locatorIndex);
      // Get effective constraint (potentially reprojected)
      const SkinnedLocatorTriangleConstraintT<T> effectiveConstr =
          getEffectiveConstraint(*meshState.restMesh_, constr, locatorRestPos);
      error += calculatePositionJacobian(
          modelParameters,
          meshState,
          effectiveConstr,
          jacobian.block(3 * i, 0, 3, modelParameters.size()),
          residual.middleRows(3 * i, 3));
    }
    usedRows = 3 * constraints_.size();
  } else {
    MT_CHECK(constraintType_ == VertexConstraintType::Plane);
    for (size_t i = 0; i < constraints_.size(); ++i) {
      const SkinnedLocatorTriangleConstraintT<T>& constr = constraints_[i];
      // Get locator position for reprojection
      const Eigen::Vector3<T> locatorRestPos =
          getLocatorRestPosition(modelParameters, constr.locatorIndex);
      // Get effective constraint (potentially reprojected)
      const SkinnedLocatorTriangleConstraintT<T> effectiveConstr =
          getEffectiveConstraint(*meshState.restMesh_, constr, locatorRestPos);
      T residualValue;
      error += calculatePlaneJacobian(
          modelParameters,
          meshState,
          effectiveConstr,
          jacobian.block(i, 0, 1, modelParameters.size()),
          residualValue);
      residual(i) = residualValue;
    }
    usedRows = constraints_.size();
  }

  return error;
}

template <typename T>
size_t SkinnedLocatorTriangleErrorFunctionT<T>::getJacobianSize() const {
  if (constraintType_ == VertexConstraintType::Position) {
    return constraints_.size() * 3;
  } else if (constraintType_ == VertexConstraintType::Plane) {
    return constraints_.size();
  }
  return 0;
}

} // namespace momentum

// Explicit template instantiations
template class momentum::SkinnedLocatorTriangleErrorFunctionT<float>;
template class momentum::SkinnedLocatorTriangleErrorFunctionT<double>;
