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

// Joint transform is : WorldTransform = ParentWorldTransform * T * Rpre * R * S,
// with R depending on rotation order.
// Each joint thus has 7 parameters, 3 translation, 3 rotation, 1 scale.

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

  // calculate state based on parameters and parent transform
  if (computeDeriv) {
    if (parentState != nullptr) {
      translationAxis = parentState->transformation.linear();
    } else {
      translationAxis.setIdentity();
    }
  }

  // do the translations
  localTransform.translation.noalias() = joint.translationOffset + parameters.template head<3>();

  // apply pre-rotation
  localTransform.rotation = joint.preRotation;

  // do the rotations
  for (int index = 2; index >= 0; --index) {
    if (computeDeriv) {
      rotationAxis.col(index).noalias() =
          (parent.rotation * localTransform.rotation) * RotationAxis<T>[index];
    }
    localTransform.rotation *=
        Quaternion<T>(AngleAxis<T>((T)parameters[3 + index], RotationAxis<T>[index]));
  }

  // perform scale if necessary
  localTransform.scale = std::exp2((T)parameters[6]);

  // set global transformation
  transform = parent * localTransform;

  // TODO: Remove
  transformation = transform.toAffine3();
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
  // TODO: Remove
  transformation = rhs.transformation.template cast<T>();

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
