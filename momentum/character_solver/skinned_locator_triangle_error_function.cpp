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
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/mesh.h"

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

  this->neutralMesh_ = std::make_unique<MeshT<T>>(character_in.mesh->template cast<T>());
  this->restMesh_ = std::make_unique<MeshT<T>>(character_in.mesh->template cast<T>());
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
          locatorIndex, triangleIndices, triangleBaryCoords, depth, weight});
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
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& /* meshState */) {
  MT_PROFILE_FUNCTION();

  updateMeshes(modelParameters);

  // loop over all constraints and calculate the error
  double error = 0.0;

  if (constraintType_ == VertexConstraintType::Position) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      const SkinnedLocatorTriangleConstraintT<T>& constr = constraints_[i];

      // Get the skinned locator position
      const Eigen::Vector3<T> locatorRestPos =
          getLocatorRestPosition(modelParameters, constr.locatorIndex);

      // Get the target position on the triangle
      const Eigen::Vector3<T> tgtPoint = computeTargetPosition(*this->restMesh_, constr);

      // Calculate position error
      const Eigen::Vector3<T> diff = locatorRestPos - tgtPoint;
      error += constr.weight * diff.squaredNorm();
    }
  } else if (constraintType_ == VertexConstraintType::Plane) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      const SkinnedLocatorTriangleConstraintT<T>& constr = constraints_[i];

      // Get the skinned locator position
      const Eigen::Vector3<T> locatorRestPos =
          getLocatorRestPosition(modelParameters, constr.locatorIndex);

      // Get the target position and normal
      const Eigen::Vector3<T> targetNormal = computeTargetTriangleNormal(*this->restMesh_, constr);
      const Eigen::Vector3<T> tgtPoint =
          computeTargetBaryPosition(*this->restMesh_, constr) + constr.depth * targetNormal;
      // Calculate plane error (projection onto normal)
      const T dist = targetNormal.dot(locatorRestPos - tgtPoint);
      error += constr.weight * dist * dist;
    }
  }

  // return error
  return error * this->weight_;
}

template <typename T>
double SkinnedLocatorTriangleErrorFunctionT<T>::calculatePositionGradient(
    const ModelParametersT<T>& modelParameters,
    const SkinnedLocatorTriangleConstraintT<T>& constr,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  // Get the skinned locator position
  const Eigen::Vector3<T> locatorRestPos =
      getLocatorRestPosition(modelParameters, constr.locatorIndex);

  // Get the target position on the triangle
  const Eigen::Vector3<T> tgtPoint = computeTargetPosition(*this->restMesh_, constr);

  // Calculate position error
  const Eigen::Vector3<T> diff = locatorRestPos - tgtPoint;

  // Calculate gradient weight
  const T wgt = 2.0f * constr.weight * this->weight_;

  // Get the derivative of the target position with respect to the triangle vertices
  const auto d_targetPos_d_tgtTriVertexPos =
      compute_d_targetPos_d_vertexPos(constr, *this->restMesh_);

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
    const SkinnedLocatorTriangleConstraintT<T>& constr,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  // Get the skinned locator position
  const Eigen::Vector3<T> locatorRestPos =
      getLocatorRestPosition(modelParameters, constr.locatorIndex);

  // Get the target position and normal
  const Eigen::Vector3<T> targetNormal = computeTargetTriangleNormal(*this->restMesh_, constr);
  const Eigen::Vector3<T> tgtPoint =
      computeTargetBaryPosition(*this->restMesh_, constr) + constr.depth * targetNormal;

  // Calculate plane error (projection onto normal)
  const Vector3<T> diff = locatorRestPos - tgtPoint;
  const T dist = targetNormal.dot(diff);
  const T wgt = 2.0f * constr.weight * this->weight_;

  // Get the derivative of the target position with respect to the triangle vertices
  const auto d_targetPos_d_tgtTriVertexPos =
      compute_d_targetPos_d_vertexPos(constr, *this->restMesh_);
  const auto d_targetNormal_d_tgtTriVertexPos =
      compute_d_targetNormal_d_vertexPos(constr, *this->restMesh_);

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
    const SkinnedLocatorTriangleConstraintT<T>& constr,
    Ref<Eigen::MatrixX<T>> jac,
    Ref<Eigen::VectorX<T>> res) const {
  // Get the skinned locator position
  const Eigen::Vector3<T> locatorRestPos =
      getLocatorRestPosition(modelParameters, constr.locatorIndex);

  // Get the target position on the triangle
  const Eigen::Vector3<T> tgtPoint = computeTargetPosition(*this->restMesh_, constr);

  // Calculate position error
  const Eigen::Vector3<T> diff = locatorRestPos - tgtPoint;

  // Set residual
  const T wgt = std::sqrt(constr.weight * this->weight_);
  res = wgt * diff;

  // Get the derivative of the target position with respect to the triangle vertices
  const auto d_targetPos_d_tgtTriVertexPos =
      compute_d_targetPos_d_vertexPos(constr, *this->restMesh_);

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
    const SkinnedLocatorTriangleConstraintT<T>& constr,
    Ref<Eigen::MatrixX<T>> jac,
    T& res) const {
  // Get the skinned locator position
  const Eigen::Vector3<T> locatorRestPos =
      getLocatorRestPosition(modelParameters, constr.locatorIndex);

  // Get the target position and normal
  const Eigen::Vector3<T> targetNormal = computeTargetTriangleNormal(*this->restMesh_, constr);
  const Eigen::Vector3<T> tgtPoint =
      computeTargetBaryPosition(*this->restMesh_, constr) + constr.depth * targetNormal;

  // Calculate plane error (projection onto normal)
  const Eigen::Vector3<T> diff = locatorRestPos - tgtPoint;
  const T dist = targetNormal.dot(diff);

  // Set residual
  const T wgt = std::sqrt(constr.weight * this->weight_);
  res = wgt * dist;

  // Get the derivative of the target position with respect to the triangle vertices
  const auto d_targetPos_d_tgtTriVertexPos =
      compute_d_targetPos_d_vertexPos(constr, *this->restMesh_);
  const auto d_targetNormal_d_tgtTriVertexPos =
      compute_d_targetNormal_d_vertexPos(constr, *this->restMesh_);

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
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& /* meshState */,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  updateMeshes(modelParameters);

  double error = 0.0;

  for (const auto& constraint : constraints_) {
    if (constraintType_ == VertexConstraintType::Position) {
      error += calculatePositionGradient(modelParameters, constraint, gradient);
    } else if (constraintType_ == VertexConstraintType::Plane) {
      error += calculatePlaneGradient(modelParameters, constraint, gradient);
    }
  }

  return error * this->weight_;
}

template <typename T>
double SkinnedLocatorTriangleErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& /* meshState */,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_CHECK(
      jacobian.cols() == static_cast<Eigen::Index>(this->parameterTransform_.transform.cols()));

  updateMeshes(modelParameters);

  double error = 0.0;
  usedRows = 0;

  for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
    if (constraintType_ == VertexConstraintType::Position) {
      error += calculatePositionJacobian(
          modelParameters,
          constraints_[iCons],
          jacobian.block(usedRows, 0, 3, modelParameters.size()),
          residual.middleRows(usedRows, 3));
      usedRows += 3;
    } else if (constraintType_ == VertexConstraintType::Plane) {
      T res;
      error += calculatePlaneJacobian(
          modelParameters,
          constraints_[iCons],
          jacobian.block(usedRows, 0, 1, modelParameters.size()),
          res);
      residual[usedRows] = res;
      usedRows += 1;
    }
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

template <typename T>
void SkinnedLocatorTriangleErrorFunctionT<T>::updateMeshes(
    const ModelParametersT<T>& modelParameters) {
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
}

// Explicit template instantiations
template class SkinnedLocatorTriangleErrorFunctionT<float>;
template class SkinnedLocatorTriangleErrorFunctionT<double>;

} // namespace momentum
