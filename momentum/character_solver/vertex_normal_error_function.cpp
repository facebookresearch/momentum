/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/vertex_normal_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"

#include "momentum/common/checks.h"
#include "momentum/math/mesh.h"

#include <dispenso/parallel_for.h>

#include <numeric>

namespace momentum {

template <typename T>
VertexNormalErrorFunctionT<T>::VertexNormalErrorFunctionT(
    const Character& character,
    const ParameterTransform& parameterTransform,
    T sourceNormalWeight,
    T targetNormalWeight,
    const T& lossAlpha,
    const T& lossC,
    bool computeAccurateNormalDerivatives)
    : VertexConstraintErrorFunctionT<T, VertexNormalDataT<T>, 1>(
          character,
          parameterTransform,
          lossAlpha,
          lossC),
      sourceNormalWeight_(sourceNormalWeight),
      targetNormalWeight_(targetNormalWeight),
      computeAccurateNormalDerivatives_(computeAccurateNormalDerivatives) {
  this->legacyWeight_ = this->kLegacyWeight;
}

template <typename T>
void VertexNormalErrorFunctionT<T>::evalFunction(
    size_t constrIndex,
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& meshState,
    std::span<const Eigen::Vector3<T>> worldVecs,
    FuncType& f,
    std::span<DfdvType> dfdv) const {
  MT_CHECK(constrIndex < this->constraints_.size(), "Constraint index out of range");
  MT_CHECK(!worldVecs.empty(), "World vectors must not be empty");
  MT_CHECK_NOTNULL(meshState.posedMesh_);

  const auto& constr = this->constraints_[constrIndex];
  const Eigen::Vector3<T>& worldPos = worldVecs[0];

  // Get source normal from posed mesh
  const Eigen::Vector3<T> sourceNormal =
      meshState.posedMesh_->normals[constr.vertexIndex].template cast<T>();
  Eigen::Vector3<T> targetNormal = constr.targetNormal.template cast<T>();

  // Flip target normal if facing away from source
  if (sourceNormal.dot(targetNormal) < T(0)) {
    targetNormal *= T(-1);
  }

  const Eigen::Vector3<T> normal =
      sourceNormalWeight_ * sourceNormal + targetNormalWeight_ * targetNormal;

  const Eigen::Vector3<T> diff = worldPos - constr.targetPosition.template cast<T>();
  f[0] = normal.dot(diff);

  if (!dfdv.empty()) {
    // df/d(worldPos) = normal^T
    // The normal rotation correction is handled by the normal correction methods
    // in SkeletonDerivativeT.
    dfdv[0] = normal.transpose();
  }
}

template <typename T>
void VertexNormalErrorFunctionT<T>::buildVertexToFaceAdjacency(
    const MeshStateT<T>& meshState) const {
  if (vertexToFaceAdjacencyValid_) {
    return;
  }

  MT_CHECK(meshState.posedMesh_ != nullptr);

  const auto& faces = meshState.posedMesh_->faces;
  const size_t numVertices = meshState.posedMesh_->vertices.size();

  vertexToFaceAdjacency_.clear();
  vertexToFaceAdjacency_.resize(numVertices);

  for (size_t fIdx = 0; fIdx < faces.size(); ++fIdx) {
    const auto& face = faces[fIdx];
    vertexToFaceAdjacency_[face[0]].push_back(fIdx);
    vertexToFaceAdjacency_[face[1]].push_back(fIdx);
    vertexToFaceAdjacency_[face[2]].push_back(fIdx);
  }

  vertexToFaceAdjacencyValid_ = true;
}

template <typename T>
double VertexNormalErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  const size_t numConstraints = this->constraints_.size();
  if (numConstraints == 0) {
    return 0.0;
  }

  const SkeletonDerivativeT<T> skeletonDerivative(
      this->skeleton_,
      this->parameterTransform_,
      this->activeJointParams_,
      this->enabledParameters_);

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = static_cast<uint32_t>(this->maxThreads_);

  // L2 fast path: skip loss_.value()/deriv() calls entirely when loss is L2
  const bool isL2 = this->loss_.isL2();
  const T lossInvC2 = this->loss_.invC2();

  // Issue 1 fix: Check blend shape existence OUTSIDE the loop.
  // Without blend shapes, accurate and approximate modes are identical in performance.
  const bool hasBlendShapesForAccurate = (this->character_.blendShape != nullptr) ||
      (this->character_.faceExpressionBlendShape != nullptr);
  const bool doAccurateNormal =
      computeAccurateNormalDerivatives_ && sourceNormalWeight_ != T(0) && hasBlendShapesForAccurate;

  // Issue 2 fix: Pre-build vertex-to-face adjacency ONCE before the loop.
  if (doAccurateNormal) {
    buildVertexToFaceAdjacency(meshState);
  }

  std::vector<std::tuple<double, VectorX<T>>> errorGradThread;
  dispenso::parallel_for(
      errorGradThread,
      [&]() -> std::tuple<double, VectorX<T>> {
        return {0.0, VectorX<T>::Zero(this->parameterTransform_.numAllModelParameters())};
      },
      size_t(0),
      numConstraints,
      [&](std::tuple<double, VectorX<T>>& errorGradLocal, const size_t i) {
        double& errorLocal = std::get<0>(errorGradLocal);
        auto& gradLocal = std::get<1>(errorGradLocal);

        const T constrWeight = static_cast<T>(this->constraints_[i].weight);
        if (constrWeight == T(0)) {
          return;
        }

        const auto& constr = this->constraints_[i];
        const size_t vertexIndex = constr.vertexIndex;
        const Eigen::Vector3<T> worldPos =
            meshState.posedMesh_->vertices[vertexIndex].template cast<T>();

        // INLINE evalFunction — NO virtual call
        const Eigen::Vector3<T> sourceNormal =
            meshState.posedMesh_->normals[vertexIndex].template cast<T>();
        Eigen::Vector3<T> targetNormal = constr.targetNormal.template cast<T>();
        if (sourceNormal.dot(targetNormal) < T(0)) {
          targetNormal *= T(-1);
        }
        const Eigen::Vector3<T> normal =
            sourceNormalWeight_ * sourceNormal + targetNormalWeight_ * targetNormal;
        const Eigen::Vector3<T> diff = worldPos - constr.targetPosition.template cast<T>();
        const T f = normal.dot(diff);

        if (f == T(0)) {
          return;
        }

        const T sqrError = f * f;
        const T w = constrWeight * this->weight_ * this->legacyWeight_;

        T weightedResidualScalar;
        if (isL2) {
          errorLocal += w * sqrError * lossInvC2;
          weightedResidualScalar = T(2) * w * lossInvC2 * f;
        } else {
          errorLocal += w * this->loss_.value(sqrError);
          weightedResidualScalar = T(2) * w * this->loss_.deriv(sqrError) * f;
        }

        // Use accumulateVertexGradientWithNormalCorrectionDirect for scalar operations
        if (sourceNormalWeight_ != T(0)) {
          const T normalScale = weightedResidualScalar * sourceNormalWeight_;
          skeletonDerivative.accumulateVertexGradientWithNormalCorrectionDirect(
              vertexIndex,
              worldPos,
              normal,
              weightedResidualScalar,
              sourceNormal,
              constr.targetPosition,
              normalScale,
              state,
              meshState,
              this->character_,
              gradLocal);
        } else {
          // No normal correction — use standard vertex gradient
          DfdvType dfdv = normal.transpose();
          FuncType wr;
          wr[0] = weightedResidualScalar;
          skeletonDerivative.template accumulateVertexGradient<1>(
              vertexIndex, worldPos, dfdv, wr, state, meshState, this->character_, gradLocal);
        }

        // Accurate mode: d(normal)/d(blendShapeWeight) with cached adjacency
        if (doAccurateNormal) {
          const T normalScale = weightedResidualScalar * sourceNormalWeight_;
          const Eigen::Vector3<T> leverArm = worldPos - constr.targetPosition;
          skeletonDerivative.accumulateVertexNormalBlendShapeGradient(
              vertexIndex,
              leverArm,
              normalScale,
              std::span<const size_t>(vertexToFaceAdjacency_[vertexIndex]),
              state,
              meshState,
              this->character_,
              gradLocal);
        }
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

    gradient += std::get<1>(errorGradThread[0]);
    return std::get<0>(errorGradThread[0]);
  }

  return 0.0;
}

template <typename T>
double VertexNormalErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  const size_t numConstraints = this->constraints_.size();
  usedRows = 0;
  if (numConstraints == 0) {
    return 0.0;
  }

  const SkeletonDerivativeT<T> skeletonDerivative(
      this->skeleton_,
      this->parameterTransform_,
      this->activeJointParams_,
      this->enabledParameters_);

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = static_cast<uint32_t>(this->maxThreads_);

  // L2 fast path: skip loss_.value()/deriv() calls entirely when loss is L2
  const bool isL2Jac = this->loss_.isL2();
  const T lossInvC2Jac = this->loss_.invC2();

  // Issue 1 fix: Check blend shape existence OUTSIDE the loop.
  const bool hasBlendShapesForAccurate = (this->character_.blendShape != nullptr) ||
      (this->character_.faceExpressionBlendShape != nullptr);
  const bool doAccurateNormal =
      computeAccurateNormalDerivatives_ && sourceNormalWeight_ != T(0) && hasBlendShapesForAccurate;

  // Issue 2 fix: Pre-build vertex-to-face adjacency ONCE before the loop.
  if (doAccurateNormal) {
    buildVertexToFaceAdjacency(meshState);
  }

  std::vector<double> errorThread;
  dispenso::parallel_for(
      errorThread,
      [&]() -> double { return 0.0; },
      size_t(0),
      numConstraints,
      [&](double& errorLocal, const size_t i) {
        const T constrWeight = static_cast<T>(this->constraints_[i].weight);
        if (constrWeight == T(0)) {
          return;
        }

        const auto& constr = this->constraints_[i];
        const size_t vertexIndex = constr.vertexIndex;
        const Eigen::Vector3<T> worldPos =
            meshState.posedMesh_->vertices[vertexIndex].template cast<T>();

        // INLINE evalFunction — NO virtual call
        const Eigen::Vector3<T> sourceNormal =
            meshState.posedMesh_->normals[vertexIndex].template cast<T>();
        Eigen::Vector3<T> targetNormal = constr.targetNormal.template cast<T>();
        if (sourceNormal.dot(targetNormal) < T(0)) {
          targetNormal *= T(-1);
        }
        const Eigen::Vector3<T> normal =
            sourceNormalWeight_ * sourceNormal + targetNormalWeight_ * targetNormal;
        const Eigen::Vector3<T> diff = worldPos - constr.targetPosition.template cast<T>();
        const T f = normal.dot(diff);

        const size_t rowIndex = i;
        const T sqrError = f * f;
        const T w = constrWeight * this->weight_ * this->legacyWeight_;

        T deriv;
        if (isL2Jac) {
          errorLocal += w * sqrError * lossInvC2Jac;
          deriv = std::sqrt(w * lossInvC2Jac);
        } else {
          errorLocal += w * this->loss_.value(sqrError);
          deriv = std::sqrt(w * this->loss_.deriv(sqrError));
        }

        residual[rowIndex] = deriv * f;

        if (deriv == T(0)) {
          return;
        }

        // Use scalar-specialized Direct methods
        if (sourceNormalWeight_ != T(0)) {
          const T normalScale = deriv * sourceNormalWeight_;

          skeletonDerivative.accumulateVertexJacobianWithNormalCorrectionDirect(
              vertexIndex,
              worldPos,
              normal,
              deriv,
              sourceNormal,
              constr.targetPosition,
              normalScale,
              state,
              meshState,
              this->character_,
              jacobian,
              rowIndex);

          // Accurate mode: d(normal)/d(blendShapeWeight) with cached adjacency
          if (doAccurateNormal) {
            const Eigen::Vector3<T> leverArm = worldPos - constr.targetPosition;
            skeletonDerivative.accumulateVertexNormalBlendShapeJacobian(
                vertexIndex,
                leverArm,
                normalScale,
                std::span<const size_t>(vertexToFaceAdjacency_[vertexIndex]),
                state,
                meshState,
                this->character_,
                jacobian,
                rowIndex);
          }
        } else {
          DfdvType dfdv = normal.transpose();
          skeletonDerivative.template accumulateVertexJacobian<1>(
              vertexIndex,
              worldPos,
              dfdv,
              deriv,
              state,
              meshState,
              this->character_,
              jacobian,
              rowIndex);
        }
      },
      dispensoOptions);

  usedRows = static_cast<int>(numConstraints);

  if (!errorThread.empty()) {
    return std::accumulate(errorThread.begin() + 1, errorThread.end(), errorThread[0]);
  }

  return 0.0;
}

template class VertexNormalErrorFunctionT<float>;
template class VertexNormalErrorFunctionT<double>;

} // namespace momentum
