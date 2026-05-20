/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/camera_vertex_projection_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/mesh_state.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character_solver/skeleton_derivative.h"
#include "momentum/common/checks.h"

#include <dispenso/parallel_for.h>

#include <numeric>
#include <tuple>

namespace momentum {

template <typename T>
CameraVertexProjectionErrorFunctionT<T>::CameraVertexProjectionErrorFunctionT(
    const Character& character,
    const ParameterTransform& pt,
    std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel,
    size_t cameraParent,
    const Eigen::Transform<T, 3, Eigen::Affine>& cameraOffset)
    : SkeletonErrorFunctionT<T>(character.skeleton, pt),
      character_(character),
      intrinsicsModel_(std::move(intrinsicsModel)),
      cameraParent_(cameraParent),
      cameraOffset_(cameraOffset),
      skeletonDerivative_(
          character.skeleton,
          pt,
          this->activeJointParams_,
          this->enabledParameters_) {
  if (!intrinsicsModel_->name().empty()) {
    intrinsicsMapping_.emplace(pt, *intrinsicsModel_);
  }
}

template <typename T>
CameraVertexProjectionErrorFunctionT<T>::CameraVertexProjectionErrorFunctionT(
    const Character& character,
    const ParameterTransform& pt,
    const CameraT<T>& camera,
    size_t cameraParent)
    : SkeletonErrorFunctionT<T>(character.skeleton, pt),
      character_(character),
      intrinsicsModel_(camera.intrinsicsModel()),
      cameraParent_(cameraParent),
      cameraOffset_(camera.eyeFromWorld()),
      skeletonDerivative_(
          character.skeleton,
          pt,
          this->activeJointParams_,
          this->enabledParameters_) {
  if (!intrinsicsModel_->name().empty()) {
    intrinsicsMapping_.emplace(pt, *intrinsicsModel_);
  }
}

template <typename T>
const Character* CameraVertexProjectionErrorFunctionT<T>::getCharacter() const {
  return &character_;
}

template <typename T>
bool CameraVertexProjectionErrorFunctionT<T>::needsMesh() const {
  return true;
}

template <typename T>
void CameraVertexProjectionErrorFunctionT<T>::addConstraint(
    const CameraVertexProjectionDataT<T>& constraint) {
  constraints_.push_back(constraint);
}

template <typename T>
double CameraVertexProjectionErrorFunctionT<T>::getError(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState) {
  MT_CHECK_NOTNULL(meshState.posedMesh_);

  double error = 0;

  const auto& intrinsics = (intrinsicsMapping_ && intrinsicsMapping_->hasActiveParams())
      ? intrinsicsMapping_->updateIntrinsics(params)
      : *intrinsicsModel_;

  const auto& jointState = skeletonState.jointState;

  Eigen::Transform<T, 3, Eigen::Affine> eyeFromWorld;
  if (cameraParent_ == kInvalidIndex) {
    eyeFromWorld = cameraOffset_;
  } else {
    const auto worldFromEye = jointState[cameraParent_].transform * cameraOffset_;
    eyeFromWorld = worldFromEye.inverse();
  }

  for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
    const auto& cons = constraints_[iCons];
    if (cons.weight == T(0)) {
      continue;
    }

    const Eigen::Vector3<T> p_world =
        meshState.posedMesh_->vertices[cons.vertexIndex].template cast<T>();
    const Eigen::Vector3<T> p_eye = eyeFromWorld * p_world;

    const auto [projected, valid] = intrinsics.project(p_eye);
    if (!valid) {
      continue;
    }

    const Eigen::Vector2<T> residual = projected.template head<2>() - cons.target;
    error += cons.weight * this->weight_ * residual.squaredNorm();
  }

  return error;
}

template <typename T>
double CameraVertexProjectionErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  const size_t numConstraints = constraints_.size();
  if (numConstraints == 0) {
    return 0.0;
  }
  MT_CHECK_NOTNULL(meshState.posedMesh_);

  const bool hasIntrinsics = intrinsicsMapping_ && intrinsicsMapping_->hasActiveParams();
  const auto& intrinsics =
      hasIntrinsics ? intrinsicsMapping_->updateIntrinsics(params) : *intrinsicsModel_;

  const auto& jointState = skeletonState.jointState;

  Eigen::Transform<T, 3, Eigen::Affine> eyeFromWorld;
  if (cameraParent_ == kInvalidIndex) {
    eyeFromWorld = cameraOffset_;
  } else {
    const auto worldFromEye = jointState[cameraParent_].transform * cameraOffset_;
    eyeFromWorld = worldFromEye.inverse();
  }
  const Eigen::Matrix<T, 3, 3> R_eyeFromWorld = eyeFromWorld.linear();

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = static_cast<uint32_t>(maxThreads_);

  std::vector<std::tuple<double, Eigen::VectorX<T>>> errorGradThread;
  dispenso::parallel_for(
      errorGradThread,
      [&]() -> std::tuple<double, Eigen::VectorX<T>> {
        return {0.0, Eigen::VectorX<T>::Zero(this->parameterTransform_.numAllModelParameters())};
      },
      size_t(0),
      numConstraints,
      [&](std::tuple<double, Eigen::VectorX<T>>& errorGradLocal, const size_t iCons) {
        double& errorLocal = std::get<0>(errorGradLocal);
        auto& gradLocal = std::get<1>(errorGradLocal);

        const auto& cons = constraints_[iCons];
        if (cons.weight == T(0)) {
          return;
        }

        const Eigen::Vector3<T> p_world =
            meshState.posedMesh_->vertices[cons.vertexIndex].template cast<T>();
        const Eigen::Vector3<T> p_eye = eyeFromWorld * p_world;

        const auto [projected, J_eye_3x3, valid] = intrinsics.projectJacobian(p_eye);
        if (!valid) {
          return;
        }

        const Eigen::Vector2<T> residual = projected.template head<2>() - cons.target;
        errorLocal += cons.weight * this->weight_ * residual.squaredNorm();

        const T wgt = T(2) * cons.weight * this->weight_;
        const Eigen::Matrix<T, 2, 3> J_pixel_world =
            J_eye_3x3.template topRows<2>() * R_eyeFromWorld;

        // Walk 1: vertex skinning chain (positive sign)
        {
          const Eigen::Vector2<T> weightedResidual = wgt * residual;
          skeletonDerivative_.template accumulateVertexGradient<2>(
              cons.vertexIndex,
              p_world,
              J_pixel_world,
              weightedResidual,
              skeletonState,
              meshState,
              character_,
              gradLocal);
        }

        // Walk 2: camera bone chain (negative sign)
        if (cameraParent_ != kInvalidIndex) {
          const Eigen::Vector2<T> weightedResidual = -wgt * residual;
          std::array<Eigen::Vector3<T>, 1> worldVecs = {p_world};
          std::array<Eigen::Matrix<T, 2, 3>, 1> dfdvArr = {J_pixel_world};
          std::array<uint8_t, 1> isPoints = {1};

          skeletonDerivative_.template accumulateJointGradient<2>(
              cameraParent_,
              std::span<const Eigen::Vector3<T>>(worldVecs.data(), 1),
              std::span<const Eigen::Matrix<T, 2, 3>>(dfdvArr.data(), 1),
              std::span<const uint8_t>(isPoints.data(), 1),
              weightedResidual,
              jointState,
              gradLocal);
        }

        // Walk 3: intrinsics gradient
        if (hasIntrinsics) {
          const auto [proj_intr, J_intr, valid_intr] = intrinsics.projectIntrinsicsJacobian(p_eye);
          if (valid_intr) {
            intrinsicsMapping_->addGradient(J_intr, residual, wgt, gradLocal);
          }
        }
      },
      dispensoOptions);

  if (!errorGradThread.empty()) {
    errorGradThread[0] = std::accumulate(
        errorGradThread.begin() + 1,
        errorGradThread.end(),
        errorGradThread[0],
        [](const auto& a, const auto& b) -> std::tuple<double, Eigen::VectorX<T>> {
          return {std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b)};
        });

    gradient += std::get<1>(errorGradThread[0]);
    return std::get<0>(errorGradThread[0]);
  }

  return 0.0;
}

template <typename T>
double CameraVertexProjectionErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& skeletonState,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  const size_t numConstraints = constraints_.size();
  usedRows = 0;
  if (numConstraints == 0) {
    return 0.0;
  }
  MT_CHECK_NOTNULL(meshState.posedMesh_);

  const bool hasIntrinsics = intrinsicsMapping_ && intrinsicsMapping_->hasActiveParams();
  const auto& intrinsics =
      hasIntrinsics ? intrinsicsMapping_->updateIntrinsics(params) : *intrinsicsModel_;

  const auto& jointState = skeletonState.jointState;

  Eigen::Transform<T, 3, Eigen::Affine> eyeFromWorld;
  if (cameraParent_ == kInvalidIndex) {
    eyeFromWorld = cameraOffset_;
  } else {
    const auto worldFromEye = jointState[cameraParent_].transform * cameraOffset_;
    eyeFromWorld = worldFromEye.inverse();
  }
  const Eigen::Matrix<T, 3, 3> R_eyeFromWorld = eyeFromWorld.linear();

  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = static_cast<uint32_t>(maxThreads_);

  std::vector<double> errorThread;
  dispenso::parallel_for(
      errorThread,
      [&]() -> double { return 0.0; },
      size_t(0),
      numConstraints,
      [&](double& errorLocal, const size_t iCons) {
        const auto& cons = constraints_[iCons];
        if (cons.weight == T(0)) {
          return;
        }

        const Eigen::Vector3<T> p_world =
            meshState.posedMesh_->vertices[cons.vertexIndex].template cast<T>();
        const Eigen::Vector3<T> p_eye = eyeFromWorld * p_world;

        const auto [projected, J_eye_3x3, valid] = intrinsics.projectJacobian(p_eye);
        if (!valid) {
          return;
        }

        const Eigen::Vector2<T> res = projected.template head<2>() - cons.target;
        errorLocal += cons.weight * this->weight_ * res.squaredNorm();

        const T wgt = std::sqrt(cons.weight * this->weight_);
        const size_t rowIndex = 2 * iCons;
        residual.template segment<2>(rowIndex) = wgt * res;

        if (wgt == T(0)) {
          return;
        }

        const Eigen::Matrix<T, 2, 3> J_pixel_world =
            J_eye_3x3.template topRows<2>() * R_eyeFromWorld;

        // Walk 1: vertex skinning Jacobian (positive sign)
        skeletonDerivative_.template accumulateVertexJacobian<2>(
            cons.vertexIndex,
            p_world,
            J_pixel_world,
            wgt,
            skeletonState,
            meshState,
            character_,
            jacobian,
            rowIndex);

        // Walk 2: camera bone Jacobian (negative sign)
        if (cameraParent_ != kInvalidIndex) {
          std::array<Eigen::Vector3<T>, 1> worldVecs = {p_world};
          std::array<Eigen::Matrix<T, 2, 3>, 1> dfdvArr = {J_pixel_world};
          std::array<uint8_t, 1> isPoints = {1};

          skeletonDerivative_.template accumulateJointJacobian<2>(
              cameraParent_,
              std::span<const Eigen::Vector3<T>>(worldVecs.data(), 1),
              std::span<const Eigen::Matrix<T, 2, 3>>(dfdvArr.data(), 1),
              std::span<const uint8_t>(isPoints.data(), 1),
              -wgt,
              jointState,
              jacobian,
              rowIndex);
        }

        // Walk 3: intrinsics Jacobian
        if (hasIntrinsics) {
          const auto [proj_intr, J_intr, valid_intr] = intrinsics.projectIntrinsicsJacobian(p_eye);
          if (valid_intr) {
            intrinsicsMapping_->addJacobian(J_intr, wgt, rowIndex, jacobian);
          }
        }
      },
      dispensoOptions);

  usedRows = static_cast<int>(numConstraints * 2);

  if (!errorThread.empty()) {
    return std::accumulate(errorThread.begin() + 1, errorThread.end(), errorThread[0]);
  }

  return 0.0;
}

template <typename T>
size_t CameraVertexProjectionErrorFunctionT<T>::getJacobianSize() const {
  return 2 * constraints_.size();
}

template class CameraVertexProjectionErrorFunctionT<float>;
template class CameraVertexProjectionErrorFunctionT<double>;

} // namespace momentum
