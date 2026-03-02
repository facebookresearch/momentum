/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/camera_projection_constraint_error_function.h"

#include "momentum/character/skeleton_state.h"
#include "momentum/common/checks.h"

#include <array>

namespace momentum {

// ---------------------------------------------------------------------------
// Bone-parented constructor
// ---------------------------------------------------------------------------
template <typename T>
CameraProjectionConstraintErrorFunctionT<T>::CameraProjectionConstraintErrorFunctionT(
    const Character& character,
    std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel,
    size_t cameraParent,
    const Eigen::Transform<T, 3, Eigen::Affine>& cameraOffset,
    const T& lossAlpha,
    const T& lossC)
    : Base(character.skeleton, character.parameterTransform, lossAlpha, lossC),
      intrinsicsModel_(std::move(intrinsicsModel)),
      isBoneParented_(true),
      cameraParent_(cameraParent),
      cameraOffset_(cameraOffset) {}

// ---------------------------------------------------------------------------
// Static camera constructor
// ---------------------------------------------------------------------------
template <typename T>
CameraProjectionConstraintErrorFunctionT<T>::CameraProjectionConstraintErrorFunctionT(
    const Character& character,
    std::shared_ptr<const IntrinsicsModelT<T>> intrinsicsModel,
    const Eigen::Transform<T, 3, Eigen::Affine>& eyeFromWorld,
    const T& lossAlpha,
    const T& lossC)
    : Base(character.skeleton, character.parameterTransform, lossAlpha, lossC),
      intrinsicsModel_(std::move(intrinsicsModel)),
      isBoneParented_(false),
      cameraParent_(kInvalidIndex),
      cameraOffset_(eyeFromWorld) {}

// ---------------------------------------------------------------------------
// ensureContributionsCache
// ---------------------------------------------------------------------------
template <typename T>
void CameraProjectionConstraintErrorFunctionT<T>::ensureContributionsCache() const {
  const size_t n = this->constraints_.size();
  if (contributionsCacheSize_ == n) {
    return;
  }

  perConstraintContributions_.resize(n);
  for (size_t i = 0; i < n; ++i) {
    auto& contribs = perConstraintContributions_[i];
    contribs.clear();

    const auto& constr = this->constraints_[i];

    // Contribution 0: the 3D point attached to constraint's parent joint
    contribs.push_back(SourceT<T>::jointPoint(constr.parent, constr.offset));

    if (isBoneParented_) {
      // Contribution 1: camera position (JointPoint on camera parent)
      contribs.push_back(SourceT<T>::jointPoint(cameraParent_, cameraOffset_.translation()));

      // Contributions 2-4: camera X/Y/Z axes (JointDirection on camera parent)
      contribs.push_back(SourceT<T>::jointDirection(cameraParent_, cameraOffset_.linear().col(0)));
      contribs.push_back(SourceT<T>::jointDirection(cameraParent_, cameraOffset_.linear().col(1)));
      contribs.push_back(SourceT<T>::jointDirection(cameraParent_, cameraOffset_.linear().col(2)));
    }
  }
  contributionsCacheSize_ = n;
}

// ---------------------------------------------------------------------------
// getContributions
// ---------------------------------------------------------------------------
template <typename T>
std::span<const SourceT<T>> CameraProjectionConstraintErrorFunctionT<T>::getContributions(
    size_t constrIndex) const {
  MT_CHECK(constrIndex < this->constraints_.size());
  ensureContributionsCache();
  return std::span<const SourceT<T>>(perConstraintContributions_[constrIndex]);
}

// ---------------------------------------------------------------------------
// evalFunction
// ---------------------------------------------------------------------------
template <typename T>
void CameraProjectionConstraintErrorFunctionT<T>::evalFunction(
    size_t constrIndex,
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& /*meshState*/,
    std::span<const Eigen::Vector3<T>> worldVecs,
    FuncType& f,
    std::span<DfdvType> dfdv) const {
  const auto& constr = this->constraints_[constrIndex];

  if (isBoneParented_) {
    // worldVecs: [0]=p_world, [1]=t_camera, [2]=cam_x, [3]=cam_y, [4]=cam_z
    const Eigen::Vector3<T>& p_world = worldVecs[0];
    const Eigen::Vector3<T>& t_camera = worldVecs[1];

    // Reconstruct camera rotation matrix from world-space axes
    Eigen::Matrix3<T> R_cam;
    R_cam.col(0) = worldVecs[2];
    R_cam.col(1) = worldVecs[3];
    R_cam.col(2) = worldVecs[4];

    const Eigen::Vector3<T> diff = p_world - t_camera;
    const Eigen::Vector3<T> p_eye = R_cam.transpose() * diff;

    // Project through intrinsics
    const auto [projected, J_eye_3x3, valid] = intrinsicsModel_->projectJacobian(p_eye);
    if (!valid) {
      f.setZero();
      for (auto& d : dfdv) {
        d.setZero();
      }
      return;
    }

    f = projected.template head<2>() - constr.target;

    if (!dfdv.empty()) {
      const Eigen::Matrix<T, 2, 3> J_proj = J_eye_3x3.template topRows<2>();

      // R_cam^T is the same for all constraints (camera axes from same parent joint).
      // Materialize it once to avoid repeated transpose evaluation in the product.
      const Eigen::Matrix3<T> R_camT = R_cam.transpose();
      const Eigen::Matrix<T, 2, 3> J_pixel_world = J_proj * R_camT;

      // dfdv[0]: df/d(p_world) = +J_pixel_world
      dfdv[0] = J_pixel_world;

      // dfdv[1]: df/d(t_camera) = -J_pixel_world
      dfdv[1] = -J_pixel_world;

      // dfdv[2-4]: dp_eye/d(cam_k) has rank-1 structure: only row k is diff^T.
      // So J_proj * dpEye_dCamK = J_proj.col(k) * diff^T (outer product).
      const Eigen::Matrix<T, 2, 1> Jcol0 = J_proj.col(0);
      const Eigen::Matrix<T, 2, 1> Jcol1 = J_proj.col(1);
      const Eigen::Matrix<T, 2, 1> Jcol2 = J_proj.col(2);
      const Eigen::Matrix<T, 1, 3> diffT = diff.transpose();
      dfdv[2].noalias() = Jcol0 * diffT;
      dfdv[3].noalias() = Jcol1 * diffT;
      dfdv[4].noalias() = Jcol2 * diffT;
    }
  } else {
    // Static camera: worldVecs[0] = p_world
    const Eigen::Vector3<T> p_eye = cameraOffset_ * worldVecs[0];

    const auto [projected, J_eye_3x3, valid] = intrinsicsModel_->projectJacobian(p_eye);
    if (!valid) {
      f.setZero();
      for (auto& d : dfdv) {
        d.setZero();
      }
      return;
    }

    f = projected.template head<2>() - constr.target;

    if (!dfdv.empty()) {
      const Eigen::Matrix<T, 2, 3> J_proj = J_eye_3x3.template topRows<2>();
      // df/d(p_world) = J_proj * R_eyeFromWorld
      dfdv[0] = J_proj * cameraOffset_.linear();
    }
  }
}

// ---------------------------------------------------------------------------
// getGradient — stable override for bone-parented cameras
// ---------------------------------------------------------------------------
template <typename T>
double CameraProjectionConstraintErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  // Static cameras use the standard multi-contribution path (1 contribution, no cancellation)
  if (!isBoneParented_) {
    return Base::getGradient(params, state, meshState, gradient);
  }

  // Bone-parented: use a stable 2-contribution formulation to avoid numerical
  // cancellation from the 5-contribution decomposition (1 JointPoint + 3 JointDirection).
  //
  // The correct total derivative for the camera joint's rotation DOF is:
  //   df/dθ = J_pixel_world * (-(a × (p_world - joint_origin)))
  //
  // The 5-contribution scheme splits this into a point term and 3 direction terms
  // that have large intermediate values cancelling to the correct result. On platforms
  // with FMA (ARM), rounding differences make this cancellation imprecise.
  //
  // Fix: use 2 contributions — the constraint point (on its joint) and a virtual
  // point at p_world (on the camera joint). For the camera joint, using
  // worldVec = p_world with dfdv = -J_pixel_world gives the exact combined effect:
  //   rotation: (-J_pixel_world) * (a × (p_world - joint_origin)) — correct
  //   translation: (-J_pixel_world) * translation_axis — correct

  const size_t numConstraints = getNumConstraints();
  if (numConstraints == 0) {
    return 0.0;
  }

  double error = 0.0;

  for (size_t i = 0; i < numConstraints; ++i) {
    const T constrWeight = getConstraintWeight(i);
    if (constrWeight == T(0)) {
      continue;
    }

    const auto& constr = constraints_[i];

    // Compute world position of the 3D point
    const Eigen::Vector3<T> p_world = state.jointState[constr.parent].transform * constr.offset;

    // Compute camera transform from skeleton state
    const auto& camJointState = state.jointState[cameraParent_];
    const Eigen::Matrix3<T> R_cam =
        camJointState.rotation() * cameraOffset_.linear().template cast<T>();
    const Eigen::Vector3<T> t_camera = camJointState.transform * cameraOffset_.translation();

    // Compute projection
    const Eigen::Vector3<T> diff = p_world - t_camera;
    const Eigen::Vector3<T> p_eye = R_cam.transpose() * diff;

    const auto [projected, J_eye_3x3, valid] = intrinsicsModel_->projectJacobian(p_eye);
    if (!valid) {
      continue;
    }

    FuncType f = projected.template head<2>() - constr.target;
    if (f.isZero()) {
      continue;
    }

    const T sqrError = f.squaredNorm();
    const T w = constrWeight * this->weight_;
    error += w * this->loss_.value(sqrError);

    const FuncType weightedResidual = T(2) * w * this->loss_.deriv(sqrError) * f;

    const Eigen::Matrix<T, 2, 3> J_proj = J_eye_3x3.template topRows<2>();
    const DfdvType J_pixel_world = J_proj * R_cam.transpose();

    // Contribution 0: constraint point (JointPoint on constr.parent)
    {
      const std::array<Eigen::Vector3<T>, 1> worldVecs = {p_world};
      const std::array<DfdvType, 1> dfdvs = {J_pixel_world};
      const std::array<uint8_t, 1> isPoints = {1};
      this->skeletonDerivative_.template accumulateJointGradient<2>(
          constr.parent,
          std::span<const Eigen::Vector3<T>>(worldVecs),
          std::span<const DfdvType>(dfdvs),
          std::span<const uint8_t>(isPoints),
          weightedResidual,
          state.jointState,
          gradient);
    }

    // Contribution 1: camera joint — JointPoint at t_camera + JointDirection of diff
    // Using t_camera as the JointPoint gives correct translation and scale derivatives.
    // Using diff as a JointDirection gives the rotation contribution (scale is skipped
    // for directions). Combined rotation: (-J_pixel_world) * axis x (t_camera - t_j + diff)
    //   = (-J_pixel_world) * axis x (p_world - t_j), which is correct.
    {
      const std::array<Eigen::Vector3<T>, 2> cameraWorldVecs = {t_camera, diff};
      const std::array<DfdvType, 2> cameraDfdvs = {-J_pixel_world, -J_pixel_world};
      const std::array<uint8_t, 2> cameraIsPoints = {1, 0};
      this->skeletonDerivative_.template accumulateJointGradient<2>(
          cameraParent_,
          std::span<const Eigen::Vector3<T>>(cameraWorldVecs),
          std::span<const DfdvType>(cameraDfdvs),
          std::span<const uint8_t>(cameraIsPoints),
          weightedResidual,
          state.jointState,
          gradient);
    }
  }

  return error;
}

// ---------------------------------------------------------------------------
// getJacobian — stable override for bone-parented cameras
// ---------------------------------------------------------------------------
template <typename T>
double CameraProjectionConstraintErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& state,
    const MeshStateT<T>& meshState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  if (!isBoneParented_) {
    return Base::getJacobian(params, state, meshState, jacobian, residual, usedRows);
  }

  const size_t numConstraints = getNumConstraints();
  usedRows = 0;
  if (numConstraints == 0) {
    return 0.0;
  }

  double error = 0.0;
  int row = 0;

  for (size_t i = 0; i < numConstraints; ++i) {
    const T constrWeight = getConstraintWeight(i);
    if (constrWeight == T(0)) {
      continue;
    }

    const auto& constr = constraints_[i];

    const Eigen::Vector3<T> p_world = state.jointState[constr.parent].transform * constr.offset;

    const auto& camJointState = state.jointState[cameraParent_];
    const Eigen::Matrix3<T> R_cam =
        camJointState.rotation() * cameraOffset_.linear().template cast<T>();
    const Eigen::Vector3<T> t_camera = camJointState.transform * cameraOffset_.translation();

    const Eigen::Vector3<T> diff = p_world - t_camera;
    const Eigen::Vector3<T> p_eye = R_cam.transpose() * diff;

    const auto [projected, J_eye_3x3, valid] = intrinsicsModel_->projectJacobian(p_eye);
    if (!valid) {
      jacobian.row(row).setZero();
      residual(row) = T(0);
      jacobian.row(row + 1).setZero();
      residual(row + 1) = T(0);
      row += 2;
      continue;
    }

    FuncType f = projected.template head<2>() - constr.target;

    const T sqrError = f.squaredNorm();
    const T w = constrWeight * this->weight_;
    error += w * this->loss_.value(sqrError);

    const T scale = std::sqrt(w * this->loss_.deriv(sqrError));

    const Eigen::Matrix<T, 2, 3> J_proj = J_eye_3x3.template topRows<2>();
    const DfdvType J_pixel_world = J_proj * R_cam.transpose();

    // Constraint point
    {
      const std::array<Eigen::Vector3<T>, 1> worldVecs = {p_world};
      const std::array<DfdvType, 1> dfdvs = {J_pixel_world};
      const std::array<uint8_t, 1> isPoints = {1};
      this->skeletonDerivative_.template accumulateJointJacobian<2>(
          constr.parent,
          std::span<const Eigen::Vector3<T>>(worldVecs),
          std::span<const DfdvType>(dfdvs),
          std::span<const uint8_t>(isPoints),
          scale,
          state.jointState,
          jacobian,
          row);
    }

    // Camera joint — JointPoint at t_camera + JointDirection of diff
    {
      const std::array<Eigen::Vector3<T>, 2> cameraWorldVecs = {t_camera, diff};
      const std::array<DfdvType, 2> cameraDfdvs = {-J_pixel_world, -J_pixel_world};
      const std::array<uint8_t, 2> cameraIsPoints = {1, 0};
      this->skeletonDerivative_.template accumulateJointJacobian<2>(
          cameraParent_,
          std::span<const Eigen::Vector3<T>>(cameraWorldVecs),
          std::span<const DfdvType>(cameraDfdvs),
          std::span<const uint8_t>(cameraIsPoints),
          scale,
          state.jointState,
          jacobian,
          row);
    }

    residual.segment(row, 2) = scale * f;
    row += 2;
  }

  usedRows = row;
  return error;
}

template class CameraProjectionConstraintErrorFunctionT<float>;
template class CameraProjectionConstraintErrorFunctionT<double>;

} // namespace momentum
