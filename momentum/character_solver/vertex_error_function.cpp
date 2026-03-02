/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/vertex_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton.h"
#include "momentum/common/checks.h"

#include <dispenso/parallel_for.h>

#include <numeric>

namespace momentum {

template <typename T, class Data, size_t FuncDim>
VertexErrorFunctionT<T, Data, FuncDim>::VertexErrorFunctionT(
    const Character& character,
    const ParameterTransform& parameterTransform,
    const T& lossAlpha,
    const T& lossC)
    : SkeletonErrorFunctionT<T>(character.skeleton, parameterTransform),
      character_(character),
      loss_(lossAlpha, lossC) {}

template <typename T, class Data, size_t FuncDim>
const Character* VertexErrorFunctionT<T, Data, FuncDim>::getCharacter() const {
  return &character_;
}

template <typename T, class Data, size_t FuncDim>
bool VertexErrorFunctionT<T, Data, FuncDim>::needsMesh() const {
  return true;
}

template <typename T, class Data, size_t FuncDim>
void VertexErrorFunctionT<T, Data, FuncDim>::addConstraint(const Data& constraint) {
  MT_CHECK(constraint.vertexIndex != kInvalidIndex, "Constraint must have a valid vertex index");
  constraints_.push_back(constraint);
}

template <typename T, class Data, size_t FuncDim>
void VertexErrorFunctionT<T, Data, FuncDim>::setConstraints(std::span<const Data> constraints) {
  constraints_.assign(constraints.begin(), constraints.end());
}

template <typename T, class Data, size_t FuncDim>
void VertexErrorFunctionT<T, Data, FuncDim>::clearConstraints() {
  constraints_.clear();
}

template <typename T, class Data, size_t FuncDim>
const std::vector<Data>& VertexErrorFunctionT<T, Data, FuncDim>::getConstraints() const {
  return constraints_;
}

template <typename T, class Data, size_t FuncDim>
double VertexErrorFunctionT<T, Data, FuncDim>::getError(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState) {
  const size_t numConstraints = constraints_.size();
  if (numConstraints == 0) {
    return 0.0;
  }

  double error = 0.0;
  for (size_t i = 0; i < numConstraints; ++i) {
    const T constrWeight = static_cast<T>(constraints_[i].weight);
    if (constrWeight == T(0)) {
      continue;
    }

    const size_t vertexIndex = constraints_[i].vertexIndex;
    const Eigen::Vector3<T> worldVec =
        meshState.posedMesh_->vertices[vertexIndex].template cast<T>();

    FuncType f;
    std::span<DfdvType> emptyDfdv;
    evalFunction(
        i, state, meshState, std::span<const Eigen::Vector3<T>>(&worldVec, 1), f, emptyDfdv);

    const T sqrError = f.squaredNorm();
    error += constrWeight * loss_.value(sqrError);
  }
  return this->weight_ * legacyWeight_ * error;
}

template <typename T, class Data, size_t FuncDim>
double VertexErrorFunctionT<T, Data, FuncDim>::getGradient(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  const size_t numConstraints = constraints_.size();
  if (numConstraints == 0) {
    return 0.0;
  }

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = static_cast<uint32_t>(maxThreads_);

  const SkeletonDerivativeT<T> skeletonDerivative(
      this->skeleton_,
      this->parameterTransform_,
      this->activeJointParams_,
      this->enabledParameters_);

  const bool isL2 = loss_.isL2();
  const T lossInvC2 = loss_.invC2();

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

        const T constrWeight = static_cast<T>(constraints_[i].weight);
        if (constrWeight == T(0)) {
          return;
        }

        const size_t vertexIndex = constraints_[i].vertexIndex;
        const Eigen::Vector3<T> worldVec =
            meshState.posedMesh_->vertices[vertexIndex].template cast<T>();

        FuncType f;
        DfdvType dfdv;
        dfdv.setZero();

        evalFunction(
            i,
            state,
            meshState,
            std::span<const Eigen::Vector3<T>>(&worldVec, 1),
            f,
            std::span<DfdvType>(&dfdv, 1));

        if (f.isZero()) {
          return;
        }

        const T sqrError = f.squaredNorm();
        const T w = constrWeight * this->weight_ * legacyWeight_;

        FuncType weightedResidual;
        if (isL2) {
          errorLocal += w * sqrError * lossInvC2;
          weightedResidual = T(2) * w * lossInvC2 * f;
        } else {
          errorLocal += w * loss_.value(sqrError);
          weightedResidual = T(2) * w * loss_.deriv(sqrError) * f;
        }

        skeletonDerivative.template accumulateVertexGradient<FuncDim>(
            vertexIndex, worldVec, dfdv, weightedResidual, state, meshState, character_, gradLocal);
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

template <typename T, class Data, size_t FuncDim>
double VertexErrorFunctionT<T, Data, FuncDim>::getJacobian(
    const ModelParametersT<T>& /*params*/,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  const size_t numConstraints = constraints_.size();
  usedRows = 0;
  if (numConstraints == 0) {
    return 0.0;
  }

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = static_cast<uint32_t>(maxThreads_);

  const SkeletonDerivativeT<T> skeletonDerivative(
      this->skeleton_,
      this->parameterTransform_,
      this->activeJointParams_,
      this->enabledParameters_);

  const bool isL2Jac = loss_.isL2();
  const T lossInvC2Jac = loss_.invC2();

  std::vector<double> errorThread;
  dispenso::parallel_for(
      errorThread,
      [&]() -> double { return 0.0; },
      size_t(0),
      numConstraints,
      [&](double& errorLocal, const size_t i) {
        const T constrWeight = static_cast<T>(constraints_[i].weight);
        if (constrWeight == T(0)) {
          return;
        }

        const size_t vertexIndex = constraints_[i].vertexIndex;
        const Eigen::Vector3<T> worldVec =
            meshState.posedMesh_->vertices[vertexIndex].template cast<T>();

        const size_t rowIndex = i * FuncDim;
        FuncType f;
        DfdvType dfdv;
        dfdv.setZero();

        evalFunction(
            i,
            state,
            meshState,
            std::span<const Eigen::Vector3<T>>(&worldVec, 1),
            f,
            std::span<DfdvType>(&dfdv, 1));

        const T sqrError = f.squaredNorm();
        const T w = constrWeight * this->weight_ * legacyWeight_;

        T deriv;
        if (isL2Jac) {
          errorLocal += w * sqrError * lossInvC2Jac;
          deriv = std::sqrt(w * lossInvC2Jac);
        } else {
          errorLocal += w * loss_.value(sqrError);
          deriv = std::sqrt(w * loss_.deriv(sqrError));
        }

        residual.template segment<FuncDim>(rowIndex).noalias() = deriv * f;

        if (deriv == T(0)) {
          return;
        }

        skeletonDerivative.template accumulateVertexJacobian<FuncDim>(
            vertexIndex, worldVec, dfdv, deriv, state, meshState, character_, jacobian, rowIndex);
      },
      dispensoOptions);

  usedRows = static_cast<int>(numConstraints * FuncDim);

  if (!errorThread.empty()) {
    return std::accumulate(errorThread.begin() + 1, errorThread.end(), errorThread[0]);
  }

  return 0.0;
}

// Explicit instantiations for common constraint types
template class VertexErrorFunctionT<float, VertexConstraintData, 1>;
template class VertexErrorFunctionT<float, VertexConstraintData, 2>;
template class VertexErrorFunctionT<float, VertexConstraintData, 3>;
template class VertexErrorFunctionT<double, VertexConstraintData, 1>;
template class VertexErrorFunctionT<double, VertexConstraintData, 2>;
template class VertexErrorFunctionT<double, VertexConstraintData, 3>;

} // namespace momentum

// Include leaf class headers for explicit instantiations
#include "momentum/character_solver/vertex_normal_error_function.h"
#include "momentum/character_solver/vertex_plane_error_function.h"
#include "momentum/character_solver/vertex_position_error_function.h"
#include "momentum/character_solver/vertex_projection_error_function.h"

namespace momentum {

template class VertexErrorFunctionT<float, VertexPositionDataT<float>, 3>;
template class VertexErrorFunctionT<double, VertexPositionDataT<double>, 3>;

template class VertexErrorFunctionT<float, VertexNormalDataT<float>, 1>;
template class VertexErrorFunctionT<double, VertexNormalDataT<double>, 1>;

template class VertexErrorFunctionT<float, VertexPlaneDataT<float>, 1>;
template class VertexErrorFunctionT<double, VertexPlaneDataT<double>, 1>;

template class VertexErrorFunctionT<float, VertexProjectionDataT<float>, 2>;
template class VertexErrorFunctionT<double, VertexProjectionDataT<double>, 2>;

} // namespace momentum
