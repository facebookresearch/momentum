/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/joint_state.h"

#include "momentum/common/checks.h"
#include "momentum/math/constants.h"

namespace momentum {

template <typename T>
const std::array<Vector3<T>, 3> RotationAxis = {
    Vector3<T>::UnitX(),
    Vector3<T>::UnitY(),
    Vector3<T>::UnitZ()};

template <typename T>
void JointStateT<T>::set(
    const JointT<T>& joint,
    const JointVectorT<T>& parameters,
    const JointStateT<T>* parentState,
    bool computeDeriv) noexcept {
  derivDirty = !computeDeriv;

  TransformT<T> parent;
  if (parentState != nullptr) {
    parent = parentState->transform;
  }

  // Translation acts in the parent's frame, so the per-axis translation derivative is
  // the parent's rotated unit axis. Identity when there is no parent.
  if (computeDeriv) {
    if (parentState != nullptr) {
      translationAxis = parentState->transform.toLinear();
    } else {
      translationAxis.setIdentity();
    }
  }

  localTransform.translation.noalias() = joint.translationOffset + parameters.template head<3>();

  localTransform.rotation = joint.preRotation;

  // Apply rotations in reverse order (Z, Y, X) so the resulting product is Rpre * Rx * Ry * Rz.
  // The rotation derivative axis for index i is the parent rotation composed with the
  // partially-accumulated local rotation up to (but not including) axis i.
  for (int index = 2; index >= 0; --index) {
    if (computeDeriv) {
      rotationAxis.col(index).noalias() =
          (parent.rotation * localTransform.rotation) * RotationAxis<T>[index];
    }
    localTransform.rotation *=
        Quaternion<T>(AngleAxis<T>((T)parameters[3 + index], RotationAxis<T>[index]));
  }

  // Scale parameter is stored as a log2 value: scale = exp2(parameters[6]).
  // Keeps scale strictly positive and makes the parameter space uniform in log-ratio.
  localTransform.scale = std::exp2((T)parameters[6]);

  transform = parent * localTransform;
}

template <typename T>
Vector3<T> JointStateT<T>::getRotationDerivative(const size_t index, const Vector3<T>& ref) const {
  MT_CHECK(!derivDirty, "Derivatives haven't been computed yet.");
  return rotationAxis.col(index).cross(ref);
}

template <typename T>
Vector3<T> JointStateT<T>::getTranslationDerivative(const size_t index) const {
  MT_CHECK(!derivDirty, "Derivatives haven't been computed yet.");
  return translationAxis.col(index);
}

template <typename T>
Vector3<T> JointStateT<T>::getScaleDerivative(const Vector3<T>& ref) const noexcept {
  return ref * ln2<T>();
}

template <typename T>
template <typename T2>
void JointStateT<T>::set(const JointStateT<T2>& rhs) {
  localTransform = rhs.localTransform.template cast<T>();
  transform = rhs.transform.template cast<T>();

  translationAxis = rhs.translationAxis.template cast<T>();
  rotationAxis = rhs.rotationAxis.template cast<T>();

  derivDirty = rhs.derivDirty;
}

template struct JointStateT<float>;
template struct JointStateT<double>;

template void JointStateT<float>::set(const JointStateT<float>& rhs);
template void JointStateT<float>::set(const JointStateT<double>& rhs);
template void JointStateT<double>::set(const JointStateT<float>& rhs);
template void JointStateT<double>::set(const JointStateT<double>& rhs);

} // namespace momentum
