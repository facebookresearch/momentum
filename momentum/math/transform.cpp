/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/transform.h"
#include "momentum/common/checks.h"
#include "momentum/math/constants.h"
#include "momentum/math/random.h"

namespace momentum {

template <typename T>
TransformT<T> TransformT<T>::makeRotation(const Quaternion<T>& rotation_in) {
  return TransformT<T>(Vector3<T>::Zero(), rotation_in, T(1));
}

template <typename T>
TransformT<T> TransformT<T>::makeTranslation(const Vector3<T>& translation_in) {
  return TransformT<T>(translation_in, Quaternion<T>::Identity(), T(1));
}

template <typename T>
TransformT<T> TransformT<T>::makeScale(const T& scale_in) {
  return TransformT<T>(Vector3<T>::Zero(), Quaternion<T>::Identity(), scale_in);
}

template <typename T>
TransformT<T> TransformT<T>::makeRandom(bool translation, bool rotation, bool scale) {
  TransformT<T> result;

  if (translation) {
    result.translation.setRandom();
  }

  if (rotation) {
    result.rotation = Quaternion<T>::UnitRandom();
  }

  if (scale) {
    result.scale = uniform<T>(0.1, 2);
  }

  return result;
}

template <typename T>
TransformT<T> TransformT<T>::fromAffine3(const Affine3<T>& other) {
  return fromMatrix(other.matrix());
}

template <typename T>
TransformT<T> TransformT<T>::fromMatrix(const Matrix4<T>& other) {
  TransformT<T> result;
  result.translation = other.template topRightCorner<3, 1>();
  // Calculate the scale by taking the norm of the first column, assuming uniform scaling
  const auto& scaledR = other.template topLeftCorner<3, 3>();
  result.scale = scaledR.col(0).norm();
  MT_CHECK(result.scale >= Eps<T>(), "Scale is too small: {}", result.scale);
  MT_CHECK(result.scale < T(1) / Eps<T>(), "Inverse scale is too small: {}", result.scale);
  result.rotation = scaledR / result.scale;
  return result;
}

template <typename T>
Affine3<T> TransformT<T>::toAffine3() const {
  Affine3<T> xf;
  xf.makeAffine();
  xf.linear().noalias() = rotation.toRotationMatrix() * scale;
  xf.translation() = translation;
  return xf;
}

template <typename T>
Vector3<T> TransformT<T>::transformPoint(const Vector3<T>& pt) const {
  return translation + rotation * (scale * pt).eval();
}

template <typename T>
Vector3<T> TransformT<T>::rotate(const Vector3<T>& vec) const {
  return rotation * vec;
}

template <typename T>
TransformT<T> TransformT<T>::inverse() const {
  // (translate(t) * rotation(R) * scale(s)).inv() =
  //     scale(s).inv() * rotation(R).inv() * translate(t).inv()
  //   = scale(1/s) * rotation(invR) * translate(-t)
  //   = translate(-invR*s*t) * rotation(invR) * scale(1/s)
  const Quaternion<T> invRot = rotation.inverse();
  const double invScale = T(1) / scale;
  return TransformT<T>(-invScale * (invRot * translation), invRot, invScale);
}

template struct TransformT<float>;
template struct TransformT<double>;

} // namespace momentum
