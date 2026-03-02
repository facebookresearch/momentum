/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/vertex_sdf_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_solver/error_function_utils.h"
#include "momentum/character_solver/skinning_weight_iterator.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/mesh.h"
#include "momentum/math/utility.h"

namespace momentum {

namespace {

/// Finds the common ancestor joint between a skinned vertex and an SDF collider parent.
inline size_t findCommonAncestorForVertex(
    const SkinWeights& skinWeights,
    const Skeleton& skeleton,
    size_t vertexIndex,
    size_t sdfParent) {
  if (sdfParent == kInvalidIndex) {
    return kInvalidIndex;
  }

  size_t minSkinnedJoint = kInvalidIndex;
  for (uint32_t k = 0; k < kMaxSkinJoints; ++k) {
    const auto w = skinWeights.weight(vertexIndex, k);
    if (w > 0) {
      const auto joint = skinWeights.index(vertexIndex, k);
      if (minSkinnedJoint == kInvalidIndex || joint < minSkinnedJoint) {
        minSkinnedJoint = joint;
      }
    }
  }

  if (minSkinnedJoint == kInvalidIndex) {
    return kInvalidIndex;
  }

  return skeleton.commonAncestor(minSkinnedJoint, sdfParent);
}

} // namespace

template <typename T>
VertexSDFErrorFunctionT<T>::VertexSDFErrorFunctionT(
    const Character& character,
    const SDFColliderT<float>& sdfCollider)
    : SkeletonErrorFunctionT<T>(character.skeleton, character.parameterTransform),
      character_(character),
      sdfCollider_(sdfCollider) {
  MT_CHECK_NOTNULL(character_.mesh, "Character must have a mesh for vertex SDF constraints");
  MT_CHECK_NOTNULL(
      character_.skinWeights, "Character must have skin weights for vertex SDF constraints");
  MT_CHECK(sdfCollider_.isValid(), "SDF collider must be valid");
}

template <typename T>
void VertexSDFErrorFunctionT<T>::addConstraint(const VertexSDFConstraintT<T>& constraint) {
  MT_CHECK(
      constraint.vertexIndex >= 0 &&
          constraint.vertexIndex < static_cast<int>(character_.mesh->vertices.size()),
      "Invalid vertex index: {} (mesh has {} vertices)",
      constraint.vertexIndex,
      character_.mesh->vertices.size());
  constraints_.push_back(constraint);
}

template <typename T>
void VertexSDFErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
const std::vector<VertexSDFConstraintT<T>>& VertexSDFErrorFunctionT<T>::getConstraints() const {
  return constraints_;
}

template <typename T>
size_t VertexSDFErrorFunctionT<T>::getNumConstraints() const {
  return constraints_.size();
}

template <typename T>
TransformT<T> VertexSDFErrorFunctionT<T>::computeWorldToSDFTransform(
    const SkeletonStateT<T>& state) const {
  TransformT<T> parentTransform;
  if (sdfCollider_.parent != kInvalidIndex) {
    const size_t parentJoint = sdfCollider_.parent;
    MT_CHECK(parentJoint < state.jointState.size(), "Invalid parent joint index: {}", parentJoint);
    parentTransform = state.jointState[parentJoint].transform;
  }

  auto sdfWorldTransform = parentTransform * sdfCollider_.transformation.cast<T>();
  return sdfWorldTransform.inverse();
}

template <typename T>
T VertexSDFErrorFunctionT<T>::sampleSDFDistanceWithTransform(
    const Eigen::Vector3<T>& worldPos,
    const TransformT<T>& worldToSdfTransform) const {
  const Vector3<T> sdfLocalPos = worldToSdfTransform * worldPos;
  const float distance = sdfCollider_.sdf->sample(sdfLocalPos);
  return static_cast<T>(distance);
}

template <typename T>
std::pair<T, Eigen::Vector3<T>>
VertexSDFErrorFunctionT<T>::sampleSDFDistanceAndGradientWithTransform(
    const Eigen::Vector3<T>& worldPos,
    const TransformT<T>& worldToSdfTransform,
    const TransformT<T>& sdfToWorldTransform) const {
  const Vector3<T> sdfLocalPos = worldToSdfTransform * worldPos;
  const auto [localDistance, localGradient] = sdfCollider_.sdf->sampleWithGradient(sdfLocalPos);

  const T distance = static_cast<T>(localDistance);

  const Eigen::Vector3<T> worldGradient =
      sdfToWorldTransform.rotation * localGradient.template cast<T>();

  return std::make_pair(distance, worldGradient);
}

template <typename T>
void VertexSDFErrorFunctionT<T>::accumulateJointGradient(
    const JointStateT<T>& jointState,
    size_t paramIndex,
    const Eigen::Vector3<T>& posd,
    const Eigen::Vector3<T>& direction,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  for (size_t d = 0; d < 3; d++) {
    if (this->activeJointParams_[paramIndex + d]) {
      gradient_jointParams_to_modelParams(
          direction.dot(jointState.getTranslationDerivative(d)),
          paramIndex + d,
          this->parameterTransform_,
          gradient);
    }
    if (this->activeJointParams_[paramIndex + 3 + d]) {
      gradient_jointParams_to_modelParams(
          direction.dot(jointState.getRotationDerivative(d, posd)),
          paramIndex + 3 + d,
          this->parameterTransform_,
          gradient);
    }
  }
  if (this->activeJointParams_[paramIndex + 6]) {
    gradient_jointParams_to_modelParams(
        direction.dot(jointState.getScaleDerivative(posd)),
        paramIndex + 6,
        this->parameterTransform_,
        gradient);
  }
}

template <typename T>
void VertexSDFErrorFunctionT<T>::accumulateJointJacobian(
    const JointStateT<T>& jointState,
    size_t paramIndex,
    const Eigen::Vector3<T>& posd,
    const Eigen::Vector3<T>& direction,
    Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) const {
  for (size_t d = 0; d < 3; d++) {
    if (this->activeJointParams_[paramIndex + d]) {
      jacobian_jointParams_to_modelParams<T>(
          direction.dot(jointState.getTranslationDerivative(d)),
          paramIndex + d,
          this->parameterTransform_,
          jacobianRow);
    }
    if (this->activeJointParams_[paramIndex + 3 + d]) {
      jacobian_jointParams_to_modelParams<T>(
          direction.dot(jointState.getRotationDerivative(d, posd)),
          paramIndex + 3 + d,
          this->parameterTransform_,
          jacobianRow);
    }
  }
  if (this->activeJointParams_[paramIndex + 6]) {
    jacobian_jointParams_to_modelParams<T>(
        direction.dot(jointState.getScaleDerivative(posd)),
        paramIndex + 6,
        this->parameterTransform_,
        jacobianRow);
  }
}

template <typename T>
void VertexSDFErrorFunctionT<T>::accumulateColliderHierarchyGradient(
    const SkeletonStateT<T>& state,
    size_t colliderParent,
    size_t commonAncestor,
    const Eigen::Vector3<T>& sdfSurfacePoint,
    const Eigen::Vector3<T>& direction,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  size_t jointIndex = colliderParent;
  while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = sdfSurfacePoint - jointState.translation();

    accumulateJointGradient(jointState, paramIndex, posd, direction, gradient);

    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }
}

template <typename T>
void VertexSDFErrorFunctionT<T>::accumulateColliderHierarchyJacobian(
    const SkeletonStateT<T>& state,
    size_t colliderParent,
    size_t commonAncestor,
    const Eigen::Vector3<T>& sdfSurfacePoint,
    const Eigen::Vector3<T>& direction,
    Eigen::Ref<Eigen::MatrixX<T>> jacobianRow) const {
  size_t jointIndex = colliderParent;
  while (jointIndex != kInvalidIndex && jointIndex != commonAncestor) {
    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = sdfSurfacePoint - jointState.translation();

    accumulateJointJacobian(jointState, paramIndex, posd, direction, jacobianRow);

    jointIndex = this->skeleton_.joints[jointIndex].parent;
  }
}

template <typename T>
double VertexSDFErrorFunctionT<T>::getError(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState) {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState.posedMesh_);

  double totalError = 0.0;

  if (constraints_.empty()) {
    return totalError;
  }

  const auto worldToSdfTransform = computeWorldToSDFTransform(state);

  for (const auto& constraint : constraints_) {
    const auto& worldPos = meshState.posedMesh_->vertices[constraint.vertexIndex];
    const T distance = sampleSDFDistanceWithTransform(worldPos, worldToSdfTransform);
    const T distErr = distance - constraint.targetDistance;

    totalError += static_cast<double>(
        constraint.weight * distErr * distErr * kVertexSDFWeight * this->weight_);
  }

  return totalError;
}

template <typename T>
double VertexSDFErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  MT_CHECK_NOTNULL(meshState.posedMesh_);
  MT_CHECK_NOTNULL(meshState.restMesh_);

  double totalError = 0.0;

  if (constraints_.empty()) {
    return totalError;
  }

  const auto worldToSdfTransform = computeWorldToSDFTransform(state);
  const auto sdfToWorldTransform = worldToSdfTransform.inverse();

  for (const auto& constraint : constraints_) {
    const auto& worldPos = meshState.posedMesh_->vertices[constraint.vertexIndex];
    const auto [distance, sdfGradient] = sampleSDFDistanceAndGradientWithTransform(
        worldPos, worldToSdfTransform, sdfToWorldTransform);

    const T distErr = distance - constraint.targetDistance;

    totalError += static_cast<double>(
        constraint.weight * distErr * distErr * kVertexSDFWeight * this->weight_);

    // de/d(distance) = 2 * constraintWeight * kVertexSDFWeight * weight_ * distErr
    // de/d(worldPos) = de/d(distance) * sdfGradient
    const T wgt = constraint.weight * 2.0f * kVertexSDFWeight * this->weight_;
    const Eigen::Vector3<T> diff = distErr * sdfGradient;

    // Compute the closest point on the target isosurface (lever arm for collider side)
    const Eigen::Vector3<T> sdfSurfacePoint = worldPos - distErr * sdfGradient;

    const size_t commonAncestor = findCommonAncestorForVertex(
        *character_.skinWeights, this->skeleton_, constraint.vertexIndex, sdfCollider_.parent);

    // Process mesh vertex (skinning weights)
    SkinningWeightIteratorT<T> skinningIter(
        character_, *meshState.restMesh_, state, constraint.vertexIndex);
    while (!skinningIter.finished()) {
      const auto [jointIndex, boneWeight, transformedVertex] = skinningIter.next();

      if (jointIndex == commonAncestor) {
        break;
      }

      if (std::abs(boneWeight) < std::numeric_limits<T>::epsilon()) {
        continue;
      }

      MT_CHECK(jointIndex < this->skeleton_.joints.size());

      const auto& jointState = state.jointState[jointIndex];
      const size_t paramIndex = jointIndex * kParametersPerJoint;
      const Eigen::Vector3<T> posd = transformedVertex - jointState.translation();

      accumulateJointGradient(jointState, paramIndex, posd, boneWeight * wgt * diff, gradient);
    }

    // Process collider hierarchy (walk up from collider parent)
    if (sdfCollider_.parent != kInvalidIndex) {
      accumulateColliderHierarchyGradient(
          state, sdfCollider_.parent, commonAncestor, sdfSurfacePoint, -wgt * diff, gradient);
    }
  }

  return totalError;
}

template <typename T>
double VertexSDFErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /* params */,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      jacobian.cols() == static_cast<Eigen::Index>(this->parameterTransform_.transform.cols()),
      "Jacobian column count mismatch");

  MT_CHECK_NOTNULL(meshState.posedMesh_);
  MT_CHECK_NOTNULL(meshState.restMesh_);

  double totalError = 0.0;
  int currentRow = 0;

  if (constraints_.empty()) {
    usedRows = 0;
    return totalError;
  }

  const auto worldToSdfTransform = computeWorldToSDFTransform(state);
  const auto sdfToWorldTransform = worldToSdfTransform.inverse();

  for (const auto& constraint : constraints_) {
    const auto& worldPos = meshState.posedMesh_->vertices[constraint.vertexIndex];
    const auto [distance, sdfGradient] = sampleSDFDistanceAndGradientWithTransform(
        worldPos, worldToSdfTransform, sdfToWorldTransform);

    const T distErr = distance - constraint.targetDistance;

    MT_CHECK(
        currentRow < jacobian.rows() && currentRow < residual.rows(),
        "Insufficient Jacobian/residual rows: need {}, have jacobian: {}, residual: {}",
        currentRow + 1,
        jacobian.rows(),
        residual.rows());

    const T wgt = std::sqrt(constraint.weight * kVertexSDFWeight * this->weight_);
    residual[currentRow] = wgt * distErr;

    totalError += static_cast<double>(
        constraint.weight * distErr * distErr * kVertexSDFWeight * this->weight_);

    jacobian.row(currentRow).setZero();

    // dr/d(worldPos) = wgt * sdfGradient
    const Eigen::Vector3<T> worldJacobianContribution = wgt * sdfGradient;

    // Compute the closest point on the target isosurface (lever arm for collider side)
    const Eigen::Vector3<T> sdfSurfacePoint = worldPos - distErr * sdfGradient;

    const size_t commonAncestor = findCommonAncestorForVertex(
        *character_.skinWeights, this->skeleton_, constraint.vertexIndex, sdfCollider_.parent);

    // Process mesh vertex (skinning weights)
    SkinningWeightIteratorT<T> skinningIter(
        character_, *meshState.restMesh_, state, constraint.vertexIndex);
    while (!skinningIter.finished()) {
      const auto [jointIndex, boneWeight, transformedVertex] = skinningIter.next();

      if (jointIndex == commonAncestor) {
        break;
      }

      if (std::abs(boneWeight) < std::numeric_limits<T>::epsilon()) {
        continue;
      }

      MT_CHECK(jointIndex < this->skeleton_.joints.size());

      const auto& jointState = state.jointState[jointIndex];
      const size_t paramIndex = jointIndex * kParametersPerJoint;
      const Eigen::Vector3<T> posd = transformedVertex - jointState.translation();

      accumulateJointJacobian(
          jointState,
          paramIndex,
          posd,
          boneWeight * worldJacobianContribution,
          jacobian.middleRows(currentRow, 1));
    }

    // Process collider hierarchy (walk up from collider parent)
    if (sdfCollider_.parent != kInvalidIndex) {
      accumulateColliderHierarchyJacobian(
          state,
          sdfCollider_.parent,
          commonAncestor,
          sdfSurfacePoint,
          -worldJacobianContribution,
          jacobian.middleRows(currentRow, 1));
    }

    currentRow++;
  }

  usedRows = currentRow;
  return totalError;
}

template <typename T>
size_t VertexSDFErrorFunctionT<T>::getJacobianSize() const {
  return constraints_.size();
}

// Explicit template instantiations
template class VertexSDFErrorFunctionT<float>;
template class VertexSDFErrorFunctionT<double>;

} // namespace momentum
