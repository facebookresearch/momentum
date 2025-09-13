/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/distance_error_function.h"

#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"

namespace momentum {

template <typename T>
DistanceErrorFunctionT<T>::DistanceErrorFunctionT(
    const momentum::Skeleton& skel,
    const momentum::ParameterTransform& pt)
    : SkeletonErrorFunctionT<T>(skel, pt) {}

template <typename T>
double DistanceErrorFunctionT<T>::getError(
    const momentum::ModelParametersT<T>& /*params*/,
    const momentum::SkeletonStateT<T>& skeletonState) {
  double error = 0;

  const auto& jointState = skeletonState.jointState;

  for (const auto& cons : constraints_) {
    const auto& js = jointState[cons.parent];
    const Eigen::Vector3<T> p_world_cm = js.transformation * cons.offset;
    const Eigen::Vector3<T> diff_vec = p_world_cm - cons.origin;

    const T distance = diff_vec.norm();
    error += cons.weight * kDistanceWeight * this->weight_ * sqr(distance - cons.target);
  }

  return error;
}

template <typename T>
double DistanceErrorFunctionT<T>::getGradient(
    const momentum::ModelParametersT<T>& /*params*/,
    const momentum::SkeletonStateT<T>& skeletonState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  double error = 0;

  for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
    const auto& cons = constraints_[iCons];

    const auto& jsCons = skeletonState.jointState[cons.parent];
    const Eigen::Vector3<T> p_world_cm = jsCons.transformation * cons.offset;
    const Eigen::Vector3<T> diff_vec = p_world_cm - cons.origin;

    const T distance = diff_vec.norm();
    if (distance == 0) {
      // Gradients are undefined in this case, skip to avoid NaNs in gradient.
      continue;
    }

    const Eigen::Vector3<T> diff_distance_diff_p_world = diff_vec / distance;

    const T diff = distance - cons.target;
    error += cons.weight * kDistanceWeight * this->weight_ * sqr(diff);
    const T wgt = T(2) * cons.weight * kDistanceWeight * this->weight_;

    auto addGradient = [&](const int jFullBodyDOF, const Eigen::Vector3<T>& diff_p_world_cm) {
      // d(distance)/dTheta = d(distance)/d(p_world_cm) * d(p_world_cm)/dTheta
      // distance is defined as ||p_world_cm - cons.origin||, so
      // d(distance)/d(p_world_cm) =
      //   (p_world_cm - cons.origin) / ||p_world_cm - cons.origin|| . p_world_cm
      const T diff_p_distance = diff_distance_diff_p_world.dot(diff_p_world_cm);
      const auto gradFull = diff_p_distance * diff;

      // multiply by the parameter transform:
      for (momentum::SparseRowMatrixf::InnerIterator it(
               this->parameterTransform_.transform, jFullBodyDOF);
           it;
           ++it) {
        assert(it.row() == (Eigen::Index)jFullBodyDOF);
        const auto jReducedDOF = it.col();
        gradient(jReducedDOF) += it.value() * wgt * gradFull;
      }
    };

    // loop over all joints the constraint is attached to and calculate gradient
    size_t jointIndex = cons.parent;
    while (jointIndex != kInvalidIndex) {
      // check for valid index
      assert(jointIndex < this->skeleton_.joints.size());

      const auto& jointState = skeletonState.jointState[jointIndex];
      const size_t paramIndex = jointIndex * 7;
      const Eigen::Vector3<T> posd_cm = p_world_cm - jointState.translation();

      // calculate derivatives based on active joints
      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          addGradient(paramIndex + d, jointState.getTranslationDerivative(d));
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          addGradient(paramIndex + 3 + d, jointState.getRotationDerivative(d, posd_cm));
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        addGradient(paramIndex + 6, jointState.getScaleDerivative(posd_cm));
      }

      // go to the next joint
      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }
  }

  return error;
}

template <typename T>
double DistanceErrorFunctionT<T>::getJacobian(
    const momentum::ModelParametersT<T>& /*params*/,
    const momentum::SkeletonStateT<T>& skeletonState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  double error = 0;

  for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
    const auto& cons = constraints_[iCons];

    const auto& jsCons = skeletonState.jointState[cons.parent];
    const Eigen::Vector3<T> p_world_cm = jsCons.transformation * cons.offset;
    const Eigen::Vector3<T> diff_vec = p_world_cm - cons.origin;

    const T distance = diff_vec.norm();
    if (distance == 0) {
      // Gradients are undefined in this case, skip to avoid NaNs in gradient.
      continue;
    }

    const Eigen::Vector3<T> diff_vec_normalized = diff_vec / distance;

    const T diff = distance - cons.target;
    error += cons.weight * kDistanceWeight * this->weight_ * sqr(diff);
    const T wgt = std::sqrt(cons.weight * kDistanceWeight * this->weight_);
    residual(iCons) = wgt * diff;

    auto addJacobian = [&](const size_t jFullBodyDOF, const Eigen::Vector3<T>& diff_p_world_cm) {
      const T diff_p_distance = diff_vec_normalized.dot(diff_p_world_cm);

      // multiply by the parameter transform:
      for (momentum::SparseRowMatrixf::InnerIterator it(
               this->parameterTransform_.transform, (int)jFullBodyDOF);
           it;
           ++it) {
        assert(it.row() == (Eigen::Index)jFullBodyDOF);
        const auto jReducedDOF = it.col();
        jacobian(iCons, jReducedDOF) += it.value() * wgt * diff_p_distance;
      }
    };

    // loop over all joints the constraint is attached to and calculate gradient
    size_t jointIndex = cons.parent;
    while (jointIndex != kInvalidIndex) {
      // check for valid index
      assert(jointIndex < this->skeleton_.joints.size());

      const auto& jointState = skeletonState.jointState[jointIndex];
      const size_t paramIndex = jointIndex * 7;
      const Eigen::Vector3<T> posd_cm = p_world_cm - jointState.translation();

      // calculate derivatives based on active joints
      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d]) {
          addJacobian(paramIndex + d, jointState.getTranslationDerivative(d));
        }
        if (this->activeJointParams_[paramIndex + 3 + d]) {
          addJacobian(paramIndex + 3 + d, jointState.getRotationDerivative(d, posd_cm));
        }
      }
      if (this->activeJointParams_[paramIndex + 6]) {
        addJacobian(paramIndex + 6, jointState.getScaleDerivative(posd_cm));
      }

      // go to the next joint
      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }
  }

  usedRows = static_cast<int>(jacobian.rows());

  return error;
}

template <typename T>
size_t DistanceErrorFunctionT<T>::getJacobianSize() const {
  return constraints_.size();
}

template <typename T>
DistanceConstraintDataT<T> DistanceConstraintDataT<T>::createFromLocator(
    const momentum::Locator& locator) {
  DistanceConstraintDataT<T> result;
  result.parent = locator.parent;
  result.offset = locator.offset.template cast<T>();
  result.weight = 1;
  result.target = 0;
  return result;
}

template class DistanceErrorFunctionT<float>;
template class DistanceErrorFunctionT<double>;

template struct DistanceConstraintDataT<float>;
template struct DistanceConstraintDataT<double>;

} // namespace momentum
