/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/vertex_position_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/common/checks.h"

#include <dispenso/parallel_for.h>

#include <numeric>

namespace momentum {

template <typename T>
VertexPositionErrorFunctionT<T>::VertexPositionErrorFunctionT(
    const Character& character,
    const ParameterTransform& parameterTransform,
    const T& lossAlpha,
    const T& lossC)
    : VertexConstraintErrorFunctionT<T, VertexPositionDataT<T>, 3>(
          character,
          parameterTransform,
          lossAlpha,
          lossC) {
  this->legacyWeight_ = this->kLegacyWeight;
}

template <typename T>
void VertexPositionErrorFunctionT<T>::evalFunction(
    size_t constrIndex,
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& /*meshState*/,
    std::span<const Eigen::Vector3<T>> worldVecs,
    FuncType& f,
    std::span<DfdvType> dfdv) const {
  MT_CHECK(constrIndex < this->constraints_.size(), "Constraint index out of range");
  MT_CHECK(worldVecs.size() == 1, "Expected exactly one world vector for vertex position");

  const auto& constraint = this->constraints_[constrIndex];

  // Residual: f = vertexPosition - target
  f = worldVecs[0] - constraint.target;

  // Derivative: df/dv = I (identity matrix)
  if (!dfdv.empty()) {
    MT_CHECK(dfdv.size() == 1, "Expected exactly one dfdv for vertex position");
    dfdv[0] = Eigen::Matrix<T, 3, 3>::Identity();
  }
}

template <typename T>
double VertexPositionErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  const auto& constraints = this->constraints_;
  const size_t numConstraints = constraints.size();
  if (numConstraints == 0) {
    return 0.0;
  }

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = static_cast<uint32_t>(this->maxThreads_);
  const SkeletonDerivativeT<T> skeletonDerivative(
      this->skeleton_,
      this->parameterTransform_,
      this->activeJointParams_,
      this->enabledParameters_);

  const bool isL2 = this->loss_.isL2();
  const T lossInvC2 = this->loss_.invC2();

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

        const T constrWeight = static_cast<T>(constraints[i].weight);
        if (constrWeight == T(0)) {
          return;
        }

        const size_t vertexIndex = constraints[i].vertexIndex;
        const Eigen::Vector3<T> worldPos =
            meshState.posedMesh_->vertices[vertexIndex].template cast<T>();

        // Inline the position residual: f = worldPos - target (no virtual call)
        const Eigen::Vector3<T> f = worldPos - constraints[i].target;

        if (f.isZero()) {
          return;
        }

        const T sqrError = f.squaredNorm();
        const T w = constrWeight * this->weight_ * this->legacyWeight_;

        Eigen::Vector3<T> weightedResidual;
        if (isL2) {
          errorLocal += w * sqrError * lossInvC2;
          weightedResidual = T(2) * w * lossInvC2 * f;
        } else {
          errorLocal += w * this->loss_.value(sqrError);
          weightedResidual = T(2) * w * this->loss_.deriv(sqrError) * f;
        }

        // Use IdentityDfdv specialization — eliminates dfdv*derivative matrix multiply
        skeletonDerivative.accumulateVertexGradientIdentityDfdv(
            vertexIndex, weightedResidual, state, meshState, this->character_, gradLocal);
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
double VertexPositionErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  const auto& constraints = this->constraints_;
  const size_t numConstraints = constraints.size();
  usedRows = 0;
  if (numConstraints == 0) {
    return 0.0;
  }

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = static_cast<uint32_t>(this->maxThreads_);
  const SkeletonDerivativeT<T> skeletonDerivative(
      this->skeleton_,
      this->parameterTransform_,
      this->activeJointParams_,
      this->enabledParameters_);

  constexpr size_t FuncDim = 3;
  const bool isL2 = this->loss_.isL2();
  const T lossInvC2 = this->loss_.invC2();

  std::vector<double> errorThread;
  dispenso::parallel_for(
      errorThread,
      [&]() -> double { return 0.0; },
      size_t(0),
      numConstraints,
      [&](double& errorLocal, const size_t i) {
        const T constrWeight = static_cast<T>(constraints[i].weight);
        if (constrWeight == T(0)) {
          return;
        }

        const size_t vertexIndex = constraints[i].vertexIndex;
        const Eigen::Vector3<T> worldPos =
            meshState.posedMesh_->vertices[vertexIndex].template cast<T>();

        const size_t rowIndex = i * FuncDim;

        // Inline the position residual: f = worldPos - target (no virtual call)
        const Eigen::Vector3<T> f = worldPos - constraints[i].target;

        const T sqrError = f.squaredNorm();
        const T w = constrWeight * this->weight_ * this->legacyWeight_;

        T deriv;
        if (isL2) {
          errorLocal += w * sqrError * lossInvC2;
          deriv = std::sqrt(w * lossInvC2);
        } else {
          errorLocal += w * this->loss_.value(sqrError);
          deriv = std::sqrt(w * this->loss_.deriv(sqrError));
        }

        residual.template segment<FuncDim>(rowIndex).noalias() = deriv * f;

        if (deriv == T(0)) {
          return;
        }

        // Use IdentityDfdv specialization — eliminates dfdv*derivative matrix multiply
        skeletonDerivative.accumulateVertexJacobianIdentityDfdv(
            vertexIndex, deriv, state, meshState, this->character_, jacobian, rowIndex);
      },
      dispensoOptions);

  usedRows = static_cast<int>(numConstraints * FuncDim);

  if (!errorThread.empty()) {
    return std::accumulate(errorThread.begin() + 1, errorThread.end(), errorThread[0]);
  }

  return 0.0;
}

// Explicit template instantiations
template class VertexPositionErrorFunctionT<float>;
template class VertexPositionErrorFunctionT<double>;

} // namespace momentum
