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

#include <Eigen/Eigenvalues>

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

template <typename T>
TransformT<T> blendTransforms(
    gsl::span<const TransformT<T>> transforms,
    gsl::span<const T> weights) {
  assert(transforms.size() == weights.size());
  if (transforms.empty()) {
    return TransformT<T>();
  }

  // Find average rotation by means described in
  // https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
  // http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf
  //
  // i.e. Stack quaternion coeffs in Q, compute M = Q^T x Q, and yield the eigenvector corresponding
  // to the largest eigenvalue as the average rotation
  Matrix4<T> QtQ = Matrix4<T>::Zero();

  const auto n = transforms.size();
  for (size_t i = 0; i < n; ++i) {
    const auto& q = transforms[i].rotation;
    QtQ += weights[i] * (q.coeffs() * q.coeffs().transpose());
  }

  const Eigen::SelfAdjointEigenSolver<Matrix4<T>> eigDecomp(QtQ);
  assert(
      eigDecomp.eigenvalues()[2] <=
      eigDecomp.eigenvalues()[3]); // Eigen is supposed to sort smallest to largest.
  Vector4<T> avgcoeffs = eigDecomp.eigenvectors().col(3);
  assert(std::abs(avgcoeffs.norm() - 1) < 1e-4); // should be guaranteed by the eigensolver.
  Quaternion<T> rot(avgcoeffs);

  Vector3<T> pos = Vector3<T>::Zero();
  T weightSum = 0;
  T scale = 0;
  for (size_t i = 0; i < n; ++i) {
    pos += weights[i] * transforms[i].translation;
    scale += weights[i] * transforms[i].scale;
    weightSum += weights[i];
  }

  if (weightSum != 0) {
    pos /= weightSum;
    scale /= weightSum;
  }

  return TransformT<T>{pos, rot, scale};
}

template <typename T>
TransformT<T> slerp(const TransformT<T>& t1, const TransformT<T>& t2, T weight) {
  return TransformT<T>(
      t1.translation + weight * (t2.translation - t1.translation),
      t1.rotation.slerp(weight, t2.rotation),
      t1.scale + weight * (t2.scale - t1.scale));
}

template TransformT<float> blendTransforms(
    gsl::span<const TransformT<float>> transforms,
    gsl::span<const float> weights);
template TransformT<double> blendTransforms(
    gsl::span<const TransformT<double>> transforms,
    gsl::span<const double> weights);

template TransformT<float> slerp(const TransformT<float>&, const TransformT<float>&, float);
template TransformT<double> slerp(const TransformT<double>&, const TransformT<double>&, double);

template struct TransformT<float>;
template struct TransformT<double>;

} // namespace momentum
