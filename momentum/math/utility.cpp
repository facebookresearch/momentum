/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/utility.h"

#include "momentum/common/checks.h"
#include "momentum/math/constants.h"

#include <Eigen/Eigenvalues>

#include <type_traits>

namespace momentum {

namespace {

template <typename T>
constexpr T eulerTol() {
  // Tolerance used to determine when sin(tol) can be considered approximately zero for the given
  // floating-point type (float or double). These values have been chosen based on a balance between
  // the precision of the floating-point type and numerical stability.
  return Eps<T>(1e-6f, 1e-12);
}

} // namespace

template <typename T>
Vector3<T> quaternionToRotVec(const Quaternion<T>& q) {
  T angle = T(2) * std::acos(q.w());
  // Note: This branch is theoretically unreachable since std::acos() always returns
  // a value in the range [0, pi], making angle always non-negative.
  // It's kept as defensive programming.
  if (angle < -pi<T>())
    angle = T(2) * pi<T>() - angle;
  else if (angle > pi<T>())
    angle = -(2) * pi<T>() + angle;
  const T ww = q.w() * q.w();
  if (ww > 1.0f - 1e-7f)
    return Vector3<T>::Zero();
  else {
    const T mul = T(1) / std::sqrt(T(1) - ww);
    return Vector3<T>(q.x(), q.y(), q.z()) * mul * angle;
  }
}

template <typename T>
Quaternion<T> rotVecToQuaternion(const Vector3<T>& v) {
  const T angle = v.norm();
  if (angle < 1e-5f)
    return Quaternion<T>::Identity();
  const Vector3<T> axis = v.normalized();
  return Quaternion<T>(Eigen::AngleAxis<T>(angle, axis));
}

template <typename T>
Vector3<T> rotationMatrixToEuler(
    const Matrix3<T>& m,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention) {
  if (convention == EulerConvention::Extrinsic) {
    return rotationMatrixToEuler(m, axis2, axis1, axis0, EulerConvention::Intrinsic).reverse();
  }

  return m.eulerAngles(axis0, axis1, axis2);
}

template <typename T>
Vector3<T> rotationMatrixToEulerXYZ(const Matrix3<T>& m, EulerConvention convention) {
  // If the convention is extrinsic, convert it to intrinsic and reverse the order
  if (convention == EulerConvention::Extrinsic) {
    return rotationMatrixToEulerZYX(m, EulerConvention::Intrinsic).reverse();
  }

  // Reference: https://en.wikipedia.org/wiki/Euler_angles
  // Rotation matrix representation:
  // | r00 r01 r02 |   |  cy*cz             -cy*sz              sy    |
  // | r10 r11 r12 | = |  cx*sz + sx*sy*cz   cx*cz - s1*s2*s3  -sx*cy |
  // | r20 r21 r22 |   |  ...                ...                cx*cy |

  // Computes the rotation matrix from Euler angles similarly in the following way but with a
  // different Euler angle order: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
  Vector3<T> res;
  // Check if the matrix element m(0, 2) == sin(y) is not close to 1 or -1
  if (m(0, 2) < T(1) - eulerTol<T>()) {
    if (m(0, 2) > T(-1) + eulerTol<T>()) {
      res.x() = std::atan2(-m(1, 2), m(2, 2));
      res.y() = std::asin(m(0, 2));
      res.z() = std::atan2(-m(0, 1), m(0, 0));
    } else {
      // Case sin(y) is close to -1:
      // So cos(y) == 0 which leads to m(0, 0), m(0, 1), m(0, 1), and m(0, 2) becoming
      // zero. So we use other non-zero elements in the rotation matrix
      res.x() = 0; // any angle can be OK, but we choose zero
      res.y() = -pi<T>() * 0.5; // choose in [-pi, pi]
      res.z() = std::atan2(m(1, 0), m(1, 1)); // -res.x() - atan2(...) but we use res.x() == 0
    }
  } else {
    // Case sin(y) is close to 1:
    // So cos(y) == 0 which leads to m(0, 0), m(0, 1), m(0, 1), and m(0, 2) becoming
    // zero. So we use other non-zero elements in the rotation matrix
    res.x() = 0; // any angle can be OK, but we choose zero
    res.y() = pi<T>() * 0.5; // choose in [-pi, pi]
    res.z() = std::atan2(m(1, 0), m(1, 1)); // res.x() - atan2(...) but we use res.x() == 0
  }
  return res;
}

template <typename T>
Vector3<T> rotationMatrixToEulerZYX(const Matrix3<T>& m, EulerConvention convention) {
  // If the convention is extrinsic, convert it to intrinsic and reverse the order
  if (convention == EulerConvention::Extrinsic) {
    return rotationMatrixToEulerXYZ(m, EulerConvention::Intrinsic).reverse();
  }

  // Reference: https://en.wikipedia.org/wiki/Euler_angles
  // Rotation matrix representation:
  // | r00 r01 r02 |   |  cx*cy   cx*sy*sz - sx*cz   sx*sz + cx*sy*cz |
  // | r10 r11 r12 | = |  sx*cy   ...                ...              |
  // | r20 r21 r22 |   | -sy      cy*sz              cy*cz            |

  // Computes the rotation matrix from Euler angles following:
  // http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
  Vector3<T> res;
  // Check if the matrix element m(2, 0) == -sin(y) is not close to 1 or -1
  if (m(2, 0) < T(1) - eulerTol<T>()) {
    if (m(2, 0) > T(-1) + eulerTol<T>()) {
      res.x() = std::atan2(m(1, 0), m(0, 0));
      res.y() = std::asin(-m(2, 0));
      res.z() = std::atan2(m(2, 1), m(2, 2));
    } else {
      // Case sin(y) is close to 1:
      // So cos(y) == 0 which leads to m(0, 0), m(1, 0), m(2, 1), and m(2, 2) becoming
      // zero. So we use other non-zero elements in the rotation matrix
      res.x() = 0; // any angle can be OK, but we choose zero
      res.y() = pi<T>() * 0.5; // choose in [-pi, pi]
      res.z() = std::atan2(m(0, 1), m(0, 2)); // res.x() - atan2(...) but we use res.x() == 0
    }
  } else {
    // Case sin(y) is close to -1:
    // So cos(y) == 0 which leads to m(0, 0), m(1, 0), m(2, 1), and m(2, 2) becoming
    // zero. So we use other non-zero elements in the rotation matrix
    res.x() = 0; // any angle can be OK, but we choose zero
    res.y() = -pi<T>() * 0.5; // choose in [-pi, pi]
    res.z() = std::atan2(-m(0, 1), -m(0, 2)); // -res.x() - atan2(...) but we use res.x() == 0
    // Note that this is not identical to std::atan2(m(0, 1), m(0, 2))
  }
  return res;
}

template <typename T>
Quaternion<T> eulerToQuaternion(
    const Vector3<T>& angles,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention) {
  if (convention == EulerConvention::Extrinsic) {
    return eulerToQuaternion(
        angles.reverse().eval(), axis2, axis1, axis0, EulerConvention::Intrinsic);
  }

  return Quaternion<T>(
      AngleAxis<T>(angles[0], Vector3<T>::Unit(axis0)) *
      AngleAxis<T>(angles[1], Vector3<T>::Unit(axis1)) *
      AngleAxis<T>(angles[2], Vector3<T>::Unit(axis2)));
}

template <typename T>
Matrix3<T> eulerToRotationMatrix(
    const Vector3<T>& angles,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention) {
  return eulerToQuaternion<T>(angles, axis0, axis1, axis2, convention).toRotationMatrix();
}

template <typename T>
Matrix3<T> eulerXYZToRotationMatrix(const Vector3<T>& angles, EulerConvention convention) {
  if (convention == EulerConvention::Extrinsic) {
    return eulerZYXToRotationMatrix(angles.reverse().eval(), EulerConvention::Intrinsic);
  }

  // | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
  // | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
  // | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |

  Matrix3<T> res;

  const T cx = std::cos(angles[0]);
  const T sx = std::sin(angles[0]);
  const T cy = std::cos(angles[1]);
  const T sy = std::sin(angles[1]);
  const T cz = std::cos(angles[2]);
  const T sz = std::sin(angles[2]);

  res(0, 0) = cy * cz;
  res(1, 0) = cx * sz + cz * sx * sy;
  res(2, 0) = sx * sz - cx * cz * sy;

  res(0, 1) = -cy * sz;
  res(1, 1) = cx * cz - sx * sy * sz;
  res(2, 1) = cz * sx + cx * sy * sz;

  res(0, 2) = sy;
  res(1, 2) = -cy * sx;
  res(2, 2) = cx * cy;

  return res;
}

template <typename T>
Matrix3<T> eulerZYXToRotationMatrix(const Vector3<T>& angles, EulerConvention convention) {
  if (convention == EulerConvention::Extrinsic) {
    return eulerXYZToRotationMatrix(angles.reverse().eval(), EulerConvention::Intrinsic);
  }

  // | r00 r01 r02 |   |  cy*cz  cz*sx*sy-cx*sz  cx*cz*sy+sx*sz |
  // | r10 r11 r12 | = |  cy*sz  cx*cz+sx*sy*sz -cz*sx+cx*sy*sz |
  // | r20 r21 r22 |   | -sy     cy*sx           cx*cy          |

  Matrix3<T> res;

  const T cz = std::cos(angles[0]);
  const T sz = std::sin(angles[0]);
  const T cy = std::cos(angles[1]);
  const T sy = std::sin(angles[1]);
  const T cx = std::cos(angles[2]);
  const T sx = std::sin(angles[2]);

  res(0, 0) = cz * cy;
  res(1, 0) = sz * cy;
  res(2, 0) = -sy;

  res(0, 1) = cz * sy * sx - sz * cx;
  res(1, 1) = sz * sy * sx + cz * cx;
  res(2, 1) = cy * sx;

  res(0, 2) = cz * sy * cx + sz * sx;
  res(1, 2) = sz * sy * cx - cz * sx;
  res(2, 2) = cy * cx;

  return res;
}

template <typename T>
Vector3<T> quaternionToEuler(const Quaternion<T>& q) {
  Vector3<T> res;
  res.x() =
      std::atan2(T(2) * (q.w() * q.x() + q.y() * q.z()), T(1) - T(2) * (sqr(q.x()) + sqr(q.y())));
  res.y() = std::asin(T(2) * (q.w() * q.y() - q.z() * q.x()));
  res.z() =
      std::atan2(T(2) * (q.w() * q.z() + q.x() * q.y()), T(1) - T(2) * (sqr(q.y()) + sqr(q.z())));
  return res;
}

Quaternionf quaternionAverage(gsl::span<const Quaternionf> q, gsl::span<const float> w) {
  Matrix4f Q = Matrix4f::Zero();

  // calculate the matrix
  for (size_t i = 0; i < q.size(); i++) {
    if (i < w.size())
      Q += (q[i].coeffs() * w[i]) * (q[i].coeffs() * w[i]).transpose();
    else
      Q += q[i].coeffs() * q[i].coeffs().transpose();
  }

  // get the largest eigenvector of the matrix
  return Quaternionf(Eigen::SelfAdjointEigenSolver<Matrix4f>(Q).eigenvectors().col(3));
}

template <typename T>
MatrixX<T> pseudoInverse(const MatrixX<T>& mat) {
  constexpr T pinvtoler = Eps<T>(1e-6f, 1e-60); // choose your tolerance wisely!
  const auto svd = mat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  VectorX<T> singularValues_inv = svd.singularValues();
  for (int j = 0; j < singularValues_inv.size(); ++j) {
    if (singularValues_inv(j) > pinvtoler) {
      singularValues_inv(j) = T(1.0) / singularValues_inv(j);
    } else {
      singularValues_inv(j) = T(0);
    }
  }
  return (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().transpose());
}

template <typename T>
MatrixX<T> pseudoInverse(const SparseMatrix<T>& mat) {
  return pseudoInverse(mat.toDense());
}

template <typename T>
std::tuple<bool, T, Eigen::Vector2<T>> closestPointsOnSegments(
    const Eigen::Vector3<T>& o1,
    const Eigen::Vector3<T>& d1,
    const Eigen::Vector3<T>& o2,
    const Eigen::Vector3<T>& d2,
    const T maxDist) {
  Eigen::Vector2<T> res;

  const T maxSquareDist = maxDist * maxDist;

  // first calculate closest point on the lines
  const auto w = o1 - o2;
  const auto a = d1.squaredNorm();
  const auto b = d1.dot(d2);
  const auto c = d2.squaredNorm();
  const auto d = d1.dot(w);
  const auto e = d2.dot(w);
  const auto D = (a * c - b * b);
  T sN;
  T sD = D;
  T tN;
  T tD = D;

  // check if lines are nearly parallel
  if (D < 1e-7f) {
    sN = 0.0f;
    sD = 1.0f;
    tN = e;
    tD = c;

    // early check if we are too far
    if (w.squaredNorm() > maxSquareDist)
      return std::make_tuple(false, std::numeric_limits<T>::max(), Eigen::Vector2<T>::Zero());
  } else {
    sN = (b * e - c * d);
    tN = (a * e - b * d);

    // early check if the infinite line is too far
    if ((w + (d1 * sN / D) - (d2 * tN / D)).squaredNorm() > maxSquareDist)
      return std::make_tuple(false, std::numeric_limits<T>::max(), Eigen::Vector2<T>::Zero());

    if (sN < 0.0) {
      // sc < 0 => the s=0 edge is visible
      sN = 0.0;
      tN = e;
      tD = c;
    } else if (sN > sD) {
      // sc > 1  => the s=1 edge is visible
      sN = sD;
      tN = e + b;
      tD = c;
    }
  }

  // tc < 0 => the t=0 edge is visible
  if (tN < 0.0) {
    tN = 0.0;
    // recompute sc for this edge
    if (-d < 0.0)
      sN = 0.0;
    else if (-d > a)
      sN = sD;
    else {
      sN = -d;
      sD = a;
    }
  } else if (tN > tD) {
    // tc > 1  => the t=1 edge is visible
    tN = tD;
    // recompute sc for this edge
    if ((-d + b) < 0.0)
      sN = 0;
    else if ((-d + b) > a)
      sN = sD;
    else {
      sN = (-d + b);
      sD = a;
    }
  }

  // finally do the division to get sc and tc
  res[0] = (std::abs(sN) < 1e-7f ? 0.0f : sN / sD);
  res[1] = (std::abs(tN) < 1e-7f ? 0.0f : tN / tD);

  // get the difference of the two closest points
  const auto dP = w + (d1 * res[0]) - (d2 * res[1]);

  // check if this is acceptable
  const T distance = dP.squaredNorm();
  if (distance > maxSquareDist)
    return std::make_tuple(false, std::numeric_limits<T>::max(), Eigen::Vector2<T>::Zero());

  return std::make_tuple(true, std::sqrt(distance), res);
}

template MatrixX<float> pseudoInverse(const MatrixX<float>& mat);
template MatrixX<double> pseudoInverse(const MatrixX<double>& mat);

template MatrixX<float> pseudoInverse(const SparseMatrix<float>& mat);
template MatrixX<double> pseudoInverse(const SparseMatrix<double>& mat);

template Vector3<float> quaternionToRotVec(const Quaternion<float>& q);
template Vector3<double> quaternionToRotVec(const Quaternion<double>& q);
template Quaternion<float> rotVecToQuaternion(const Vector3<float>& v);
template Quaternion<double> rotVecToQuaternion(const Vector3<double>& v);

template Vector3<float> rotationMatrixToEuler(
    const Matrix3<float>& m,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention);

template Vector3<double> rotationMatrixToEuler(
    const Matrix3<double>& m,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention);

template Vector3<float> rotationMatrixToEulerXYZ(
    const Matrix3<float>& m,
    EulerConvention convention);

template Vector3<double> rotationMatrixToEulerXYZ(
    const Matrix3<double>& m,
    EulerConvention convention);

template Vector3<float> rotationMatrixToEulerZYX(
    const Matrix3<float>& m,
    EulerConvention convention);

template Vector3<double> rotationMatrixToEulerZYX(
    const Matrix3<double>& m,
    EulerConvention convention);

template Quaternion<float> eulerToQuaternion(
    const Vector3<float>& angles,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention);

template Quaternion<double> eulerToQuaternion(
    const Vector3<double>& angles,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention);

template Matrix3<float> eulerToRotationMatrix(
    const Vector3<float>& angles,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention);

template Matrix3<double> eulerToRotationMatrix(
    const Vector3<double>& angles,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention);

template Matrix3<float> eulerXYZToRotationMatrix(
    const Vector3<float>& angles,
    EulerConvention convention);

template Matrix3<double> eulerXYZToRotationMatrix(
    const Vector3<double>& angles,
    EulerConvention convention);

template Matrix3<float> eulerZYXToRotationMatrix(
    const Vector3<float>& angles,
    EulerConvention convention);

template Matrix3<double> eulerZYXToRotationMatrix(
    const Vector3<double>& angles,
    EulerConvention convention);

template Vector3f quaternionToEuler(const Quaternionf& q);
template Vector3d quaternionToEuler(const Quaterniond& q);

template std::tuple<bool, float, Eigen::Vector2<float>> closestPointsOnSegments<float>(
    const Eigen::Vector3<float>& o1,
    const Eigen::Vector3<float>& d1,
    const Eigen::Vector3<float>& o2,
    const Eigen::Vector3<float>& d2,
    const float maxDist);

template std::tuple<bool, double, Eigen::Vector2<double>> closestPointsOnSegments<double>(
    const Eigen::Vector3<double>& o1,
    const Eigen::Vector3<double>& d1,
    const Eigen::Vector3<double>& o2,
    const Eigen::Vector3<double>& d2,
    const double maxDist);

} // namespace momentum
