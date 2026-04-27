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
  return Eps<T>(T(1e-6), T(1e-12));
}

} // namespace

template <typename T>
Quaternion<T> quaternionExpMap(const Vector3<T>& v) {
  // Reference: "Practical Parameterization of Rotations Using the Exponential Map"
  // by F. Sebastian Grassia, Journal of Graphics Tools, 1998.
  // https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf

  // Compute the rotation angle (norm of the rotation vector)
  const T theta = v.norm();

  // For numerical stability, use Taylor series for small angles
  // Threshold is sqrt(epsilon): ~3.5e-4 for float, ~1.5e-8 for double
  constexpr T kSmallAngleThreshold = Eps<T>(T(3.5e-4), T(1.5e-8));

  if (theta < kSmallAngleThreshold) {
    // Taylor expansion of exp(v) = [cos(theta/2), sin(theta/2)/theta * v] about
    // theta = 0, truncated at O(theta^2). The expansion avoids the cancellation that
    // would occur in 1 - cos(theta/2) and the 0/0 division in sin(theta/2)/theta for
    // tiny theta. The truncation error is ~O(theta^4), which at the threshold
    // (sqrt(eps)) is on the order of eps^2 — well below floating-point precision.
    // The explicit normalize() compensates for the residual ~theta^4 deviation from
    // unit norm; the standard branch below produces a unit quaternion analytically
    // and so does not need it.
    const T theta2 = theta * theta;
    const T w = T(1) - theta2 / T(8);
    const T scale = T(0.5) * (T(1) - theta2 / T(24));

    Quaternion<T> q(w, scale * v.x(), scale * v.y(), scale * v.z());
    q.normalize();
    return q;
  } else {
    // Standard computation: exp([0, v]) = [cos(theta), sin(theta)/theta * v]
    const T halfTheta = theta / T(2);
    const T sinHalfTheta = std::sin(halfTheta);
    const T cosHalfTheta = std::cos(halfTheta);
    const T scale = sinHalfTheta / theta;

    return Quaternion<T>(cosHalfTheta, scale * v.x(), scale * v.y(), scale * v.z());
  }
}

template <typename T>
Vector3<T> quaternionLogMap(const Quaternion<T>& q) {
  // Reference: "Practical Parameterization of Rotations Using the Exponential Map"
  // by F. Sebastian Grassia, Journal of Graphics Tools, 1998.
  // https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf

  // Ensure the quaternion is normalized
  Quaternion<T> qNorm = q.normalized();

  // Extract components
  const T w = qNorm.w();
  const Vector3<T> vec(qNorm.x(), qNorm.y(), qNorm.z());
  const T vecNorm = vec.norm();

  // For numerical stability, use Taylor series for small angles
  // Threshold is sqrt(epsilon): ~3.5e-4 for float, ~1.5e-8 for double
  constexpr T kSmallAngleThreshold = Eps<T>(T(3.5e-4), T(1.5e-8));

  if (vecNorm < kSmallAngleThreshold) {
    if (w > T(0)) {
      // Close to identity: Taylor-expand log(q) = 2*atan2(||v||,w)/||v|| * v about
      // ||v||=0 keeping w = sqrt(1-||v||^2). To O(||v||^2): log(q) ≈ 2*v*(1 + ||v||^2/6).
      // This avoids the 0/0 in atan2/||v||.
      const T vecNorm2 = vec.squaredNorm();
      const T scale = T(2) * (T(1) + vecNorm2 / T(6));
      return scale * vec;
    } else {
      // Close to -identity (a pi rotation): the log is multi-valued — any axis n
      // gives the same -identity quaternion via exp(pi*n). We pick the x-axis
      // arbitrarily for determinism. Callers that need a continuous log near pi
      // should track the rotation axis externally.
      return Vector3<T>(pi<T>(), T(0), T(0));
    }
  }

  // Standard computation: log([w, v]) = 2 * atan2(||v||, w) / ||v|| * v.
  // atan2 (rather than acos(w)) is used for accuracy near both 0 and pi.
  const T theta = T(2) * std::atan2(vecNorm, w);
  const T scale = theta / vecNorm;

  return scale * vec;
}

// Reference: "Practical Parameterization of Rotations Using the Exponential Map"
// by F. Sebastian Grassia, Journal of Graphics Tools, 1998.
// https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf
template <typename T>
Eigen::Matrix<T, 3, 4> quaternionLogMapDerivative(const Quaternion<T>& q) {
  // Ensure the quaternion is normalized
  const Quaternion<T> qNorm = q.normalized();

  // Extract components: q.coeffs() = [x, y, z, w]
  const T w = qNorm.w();
  const Vector3<T> vec(qNorm.x(), qNorm.y(), qNorm.z());
  const T vecNorm = vec.norm();

  // For numerical stability, use Taylor series for small angles
  constexpr T kSmallAngleThreshold = Eps<T>(T(3.5e-4), T(1.5e-8));

  Eigen::Matrix<T, 3, 4> jacobian;

  if (vecNorm < kSmallAngleThreshold) {
    // Near identity: log(q) ≈ 2 * vec * (1 + vec.squaredNorm()/6)
    // For small ||vec||, we can compute the derivatives analytically
    const T vecNorm2 = vec.squaredNorm();
    const T scale = T(2) * (T(1) + vecNorm2 / T(6));

    // d(log)/d(vec) ≈ 2 * I * (1 + vecNorm2/6) + 2 * vec * vec^T / 3
    // Columns 0, 1, 2 are for x, y, z
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        jacobian(i, j) = (i == j ? scale : T(0)) + T(2) * vec(i) * vec(j) / T(3);
      }
    }

    // d(log)/dw ≈ 0 for small angles (higher order terms)
    // Column 3 is for w
    jacobian.col(3).setZero();
  } else {
    // Standard computation: log([w, v]) = theta / ||v|| * v
    // where theta = 2 * atan2(||v||, w)
    const T theta = T(2) * std::atan2(vecNorm, w);
    const T scale = theta / vecNorm;

    // Compute d(theta)/dw and d(theta)/d(vec)
    // theta = 2 * atan2(||v||, w)
    // d(theta)/dw = -2 * ||v|| / (w^2 + ||v||^2)
    // d(theta)/d(vec_i) = 2 * w * vec_i / (||v|| * (w^2 + ||v||^2))
    const T denom = w * w + vecNorm * vecNorm;
    const T dthetaDw = -T(2) * vecNorm / denom;
    const Vector3<T> dthetaDvec = T(2) * w * vec / (vecNorm * denom);

    // d(log)/d(vec_j) = d(theta/||v||)/d(vec_j) * v + (theta/||v||) * d(v)/d(vec_j)
    // where d(theta/||v||)/d(vec_j) = d(theta)/d(vec_j) / ||v|| - theta * vec_j / ||v||^3
    // Columns 0, 1, 2 are for x, y, z
    for (int j = 0; j < 3; ++j) {
      const T dScaleDvecj =
          dthetaDvec(j) / vecNorm - theta * vec(j) / (vecNorm * vecNorm * vecNorm);
      for (int i = 0; i < 3; ++i) {
        jacobian(i, j) = dScaleDvecj * vec(i) + (i == j ? scale : T(0));
      }
    }

    // Now compute d(log)/dw using product rule:
    // log = (theta / ||v||) * v
    // d(log)/dw = d(theta/||v||)/dw * v = (d(theta)/dw) / ||v|| * v
    // Column 3 is for w
    jacobian.col(3) = dthetaDw / vecNorm * vec;
  }

  return jacobian;
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
  // | r10 r11 r12 | = |  cx*sz + sx*sy*cz   cx*cz - sx*sy*sz  -sx*cy |
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
      // Gimbal lock at sin(y) == -1: cos(y) == 0, so all entries with cy as a factor
      // (m(0,0), m(0,1), m(1,2), m(2,2)) become zero. Use the remaining non-zero
      // elements (m(1,0), m(1,1)) and choose res.x() = 0 to resolve the family of
      // solutions; analytically the result has -res.x() - atan2(...) freedom.
      res.x() = 0; // any angle can be OK, but we choose zero
      res.y() = -pi<T>() * T(0.5); // choose in [-pi, pi]
      res.z() = std::atan2(m(1, 0), m(1, 1)); // -res.x() - atan2(...) but we use res.x() == 0
    }
  } else {
    // Gimbal lock at sin(y) == 1: same zero pattern as the sin(y) == -1 branch above;
    // here the analytical freedom is res.x() - atan2(...).
    res.x() = 0; // any angle can be OK, but we choose zero
    res.y() = pi<T>() * T(0.5); // choose in [-pi, pi]
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
  // Rotation matrix representation. NOTE: in the matrix below, (cx,sx) denotes cos/sin
  // of the FIRST returned angle (the Z rotation in ZYX), and (cz,sz) denotes the LAST
  // (the X rotation). This labels-by-output-position convention is the opposite of the
  // labels-by-rotation-axis convention used by eulerZYXToRotationMatrix; the indices
  // and signs below are consistent under this aliasing.
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
      // Gimbal lock at sin(y) close to 1 (since m(2,0) == -sin(y) is close to -1):
      // cos(y) == 0 zeros m(0,0), m(1,0), m(2,1), m(2,2). Recover res.z from the
      // remaining non-zero entries m(0,1), m(0,2); res.x has analytic freedom and
      // we pin it to 0.
      res.x() = 0; // any angle can be OK, but we choose zero
      res.y() = pi<T>() * T(0.5); // choose in [-pi, pi]
      res.z() = std::atan2(m(0, 1), m(0, 2)); // res.x() - atan2(...) but we use res.x() == 0
    }
  } else {
    // Gimbal lock at sin(y) close to -1 (since m(2,0) == -sin(y) is close to 1):
    // same zero pattern as above. The negated arguments to atan2 below are required
    // because the analytic family is -res.x() - atan2(...), which simplifies to
    // atan2(-m(0,1), -m(0,2)) at res.x()==0 — NOT the same as atan2(m(0,1),m(0,2)).
    res.x() = 0; // any angle can be OK, but we choose zero
    res.y() = -pi<T>() * T(0.5); // choose in [-pi, pi]
    res.z() = std::atan2(-m(0, 1), -m(0, 2)); // -res.x() - atan2(...) but we use res.x() == 0
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
  // Standard closed-form XYZ Tait-Bryan extraction. Caller must pass a normalized
  // quaternion; if not, the asin argument can exceed [-1,1] and produce NaN. There
  // is no gimbal-lock guard here because callers (skeleton_state) work in regimes
  // where |2(wy - zx)| < 1.
  // TODO: Clamp the asin argument to [-1, 1] for robustness if non-unit inputs are possible.
  Vector3<T> res;
  res.x() =
      std::atan2(T(2) * (q.w() * q.x() + q.y() * q.z()), T(1) - T(2) * (sqr(q.x()) + sqr(q.y())));
  res.y() = std::asin(T(2) * (q.w() * q.y() - q.z() * q.x()));
  res.z() =
      std::atan2(T(2) * (q.w() * q.z() + q.x() * q.y()), T(1) - T(2) * (sqr(q.y()) + sqr(q.z())));
  return res;
}

Quaternionf quaternionAverage(momentum::span<const Quaternionf> q, momentum::span<const float> w) {
  // Markley et al. "Averaging Quaternions": the average is the unit eigenvector of
  // M = sum_i w_i^2 * q_i * q_i^T corresponding to the largest eigenvalue. This is
  // sign-invariant (q and -q produce the same outer product), so it sidesteps the
  // double-cover ambiguity that breaks naive coefficient averaging.
  // NOTE: When the i-th weight is missing, this falls back to weight 1, NOT to
  // (sum of provided weights)/N — partially-specified weight vectors will yield
  // an unevenly weighted average. Pass a complete `w` or none at all.
  // NOTE: Weights enter quadratically because each q_i contributes (w_i*q_i)(w_i*q_i)^T.
  Matrix4f Q = Matrix4f::Zero();

  for (size_t i = 0; i < q.size(); i++) {
    if (i < w.size()) {
      Q += (q[i].coeffs() * w[i]) * (q[i].coeffs() * w[i]).transpose();
    } else {
      Q += q[i].coeffs() * q[i].coeffs().transpose();
    }
  }

  // SelfAdjointEigenSolver sorts eigenvalues ascending, so col(3) is the eigenvector
  // for the largest eigenvalue — i.e. the average quaternion (up to sign).
  return Quaternionf(Eigen::SelfAdjointEigenSolver<Matrix4f>(Q).eigenvectors().col(3));
}

template <typename T>
MatrixX<T> pseudoInverse(const MatrixX<T>& mat) {
  constexpr T pinvtoler = Eps<T>(T(1e-6), T(1e-60)); // choose your tolerance wisely!
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
  // Standard "closest points on two segments" algorithm: parametrize the segments as
  // o_i + t * d_i for t in [0,1] and minimize the squared distance subject to that
  // box constraint. The variables a,b,c,d,e and D = ac-b^2 are the standard names from
  // the Geometric Tools formulation; sN/sD and tN/tD are the numerator/denominator of
  // the s and t parameters carried as fractions to defer division until after clamping.
  // d1 and d2 are NOT required to be unit vectors; their magnitudes set the segment
  // length, and the returned res values are the parametric distances along d1/d2.
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

  // D is the determinant of the 2x2 normal-equations matrix; when small, the segments
  // are nearly parallel (or one direction has near-zero length) and the linear system
  // is ill-conditioned, so we pin s=0 and solve for t along d2 only.
  // TODO: T(1e-7) is a hard-coded tolerance independent of T's precision. For double
  // this is conservative but for float it is roughly 1 ulp at unit scale; consider
  // using Eps<T>(...) like the Euler routines do.
  if (D < T(1e-7)) {
    sN = T(0);
    sD = T(1);
    tN = e;
    tD = c;

    // early check if we are too far
    if (w.squaredNorm() > maxSquareDist) {
      return std::make_tuple(false, std::numeric_limits<T>::max(), Eigen::Vector2<T>::Zero());
    }
  } else {
    sN = (b * e - c * d);
    tN = (a * e - b * d);

    // early check if the infinite line is too far
    if ((w + (d1 * sN / D) - (d2 * tN / D)).squaredNorm() > maxSquareDist) {
      return std::make_tuple(false, std::numeric_limits<T>::max(), Eigen::Vector2<T>::Zero());
    }

    if (sN < T(0)) {
      // sc < 0 => the s=0 edge is visible
      sN = T(0);
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
  if (tN < T(0)) {
    tN = T(0);
    // recompute sc for this edge
    if (-d < T(0)) {
      sN = T(0);
    } else if (-d > a) {
      sN = sD;
    } else {
      sN = -d;
      sD = a;
    }
  } else if (tN > tD) {
    // tc > 1  => the t=1 edge is visible
    tN = tD;
    // recompute sc for this edge
    if ((-d + b) < T(0)) {
      sN = T(0);
    } else if ((-d + b) > a) {
      sN = sD;
    } else {
      sN = (-d + b);
      sD = a;
    }
  }

  // finally do the division to get sc and tc
  // Check for small denominators to avoid division by zero or very small numbers
  res[0] = (std::abs(sN) < T(1e-7) || std::abs(sD) < T(1e-7)) ? T(0) : sN / sD;
  res[1] = (std::abs(tN) < T(1e-7) || std::abs(tD) < T(1e-7)) ? T(0) : tN / tD;

  // get the difference of the two closest points
  const auto dP = w + (d1 * res[0]) - (d2 * res[1]);

  // check if this is acceptable
  const T distance = dP.squaredNorm();
  if (distance > maxSquareDist) {
    return std::make_tuple(false, std::numeric_limits<T>::max(), Eigen::Vector2<T>::Zero());
  }

  return std::make_tuple(true, std::sqrt(distance), res);
}

template MatrixX<float> pseudoInverse(const MatrixX<float>& mat);
template MatrixX<double> pseudoInverse(const MatrixX<double>& mat);

template MatrixX<float> pseudoInverse(const SparseMatrix<float>& mat);
template MatrixX<double> pseudoInverse(const SparseMatrix<double>& mat);

template Quaternion<float> quaternionExpMap(const Vector3<float>& v);
template Quaternion<double> quaternionExpMap(const Vector3<double>& v);
template Vector3<float> quaternionLogMap(const Quaternion<float>& q);
template Vector3<double> quaternionLogMap(const Quaternion<double>& q);

template Eigen::Matrix<float, 3, 4> quaternionLogMapDerivative(const Quaternion<float>& q);
template Eigen::Matrix<double, 3, 4> quaternionLogMapDerivative(const Quaternion<double>& q);

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

namespace {

// Rotation matrix for a single axis rotation.
template <typename T>
Matrix3<T> singleAxisRotationMatrix(int axis, T angle) {
  return AngleAxis<T>(angle, Vector3<T>::Unit(axis)).toRotationMatrix();
}

// Derivative of a single axis rotation matrix with respect to the angle.
template <typename T>
Matrix3<T> singleAxisRotationMatrixDerivative(int axis, T angle) {
  Matrix3<T> dR = Matrix3<T>::Zero();
  const T c = std::cos(angle);
  const T s = std::sin(angle);

  if (axis == 0) { // X-axis
    dR(1, 1) = -s;
    dR(1, 2) = -c;
    dR(2, 1) = c;
    dR(2, 2) = -s;
  } else if (axis == 1) { // Y-axis
    dR(0, 0) = -s;
    dR(0, 2) = c;
    dR(2, 0) = -c;
    dR(2, 2) = -s;
  } else { // Z-axis
    dR(0, 0) = -s;
    dR(0, 1) = -c;
    dR(1, 0) = c;
    dR(1, 1) = -s;
  }

  return dR;
}

// Rotation matrix for a composition R(axis1) * R(axis0). Note the order: axis0 is
// applied first (innermost), then axis1 — this matches the intrinsic angle ordering
// used by the Levenberg-Marquardt callers below.
template <typename T>
Matrix3<T> twoAxisRotationMatrix(int axis0, int axis1, const Vector2<T>& angles) {
  return singleAxisRotationMatrix(axis1, angles[1]) * singleAxisRotationMatrix(axis0, angles[0]);
}

// Returns (dR/dangle0, dR/dangle1) for R = R(axis1) * R(axis0), via the product rule.
template <typename T>
std::pair<Matrix3<T>, Matrix3<T>>
twoAxisRotationMatrixDerivatives(int axis0, int axis1, const Vector2<T>& angles) {
  const Matrix3<T> R0 = singleAxisRotationMatrix(axis0, angles[0]);
  const Matrix3<T> R1 = singleAxisRotationMatrix(axis1, angles[1]);
  const Matrix3<T> dR0 = singleAxisRotationMatrixDerivative(axis0, angles[0]);
  const Matrix3<T> dR1 = singleAxisRotationMatrixDerivative(axis1, angles[1]);

  return {R1 * dR0, dR1 * R0};
}

// Levenberg-Marquardt solver minimizing ||R(axis, angle) - target||_F^2 in the angle.
// Despite the name, this is a plain Gauss-Newton step (no LM damping term is added).
// TODO: Either add real LM damping or rename to gaussNewtonSolveScalar to match behavior.
template <typename T>
T levenbergMarquardtSolveScalar(
    const Matrix3<T>& target,
    int axis,
    T initialGuess = T(0),
    int maxIterations = 20,
    T tolerance = eulerTol<T>()) {
  T angle = initialGuess;

  T previousResidual = std::numeric_limits<T>::max();
  for (int iter = 0; iter < maxIterations; ++iter) {
    // Compute current rotation matrix and derivative
    const Matrix3<T> R = singleAxisRotationMatrix(axis, angle);

    // Compute residual (vectorized matrix difference)
    const Matrix3<T> residualMatrix = R - target;
    const T residual = residualMatrix.squaredNorm();

    // Convergence: residual already small, OR residual decreased by less than tolerance
    // since the last iteration. NOTE: the second clause also fires when the residual
    // *increased* (previousResidual - residual is negative), so this also acts as a
    // divergence guard rather than a strict monotone-progress check. It does NOT roll
    // back the diverging step — the last `angle` is whatever the last GN iteration left.
    if (residual < tolerance || (previousResidual - residual) < tolerance) {
      break;
    }
    previousResidual = residual;

    // Jacobian of the FLATTENED residual r(a) = vec(R(a) - target) w.r.t. the angle a
    // (a 9x1 column). This is the residual-vector Jacobian, NOT the gradient of the
    // squared norm — the householderQr().solve below performs a least-squares step
    // J * delta ≈ -r, which is the Gauss-Newton update.
    const Matrix3<T> dR = singleAxisRotationMatrixDerivative(axis, angle);
    const Eigen::Matrix<T, 9, 1> jacobian = dR.reshaped(9, 1);

    const Eigen::Vector<T, 9> residualVector = residualMatrix.reshaped(9, 1);

    // Gauss-Newton step (no LM damping is added — see TODO above).
    const Eigen::Vector<T, 1> delta = jacobian.householderQr().solve(-residualVector);

    // No line search or trust-region — we trust the Gauss-Newton direction directly.
    angle += delta(0);
  }

  return angle;
}

// Levenberg-Marquardt solver minimizing ||R(axis0, axis1, angles) - target||_F^2 in the
// two-vector of angles. As with the scalar version, this is currently a Gauss-Newton step
// (the comment about (J^T J + lambda I) below describes the intended LM update, not what
// is actually solved by the householder QR call).
// TODO: Add proper LM damping (lambda) or rename this to gaussNewtonSolveTwoAxis.
template <typename T>
Vector2<T> levenbergMarquardtSolveTwoAxis(
    const Matrix3<T>& target,
    int axis0,
    int axis1,
    const Vector2<T>& initialGuess = Vector2<T>::Zero(),
    int maxIterations = 20,
    T tolerance = eulerTol<T>()) {
  Vector2<T> angles = initialGuess;

  T previousResidualNorm = std::numeric_limits<T>::max();
  for (int iter = 0; iter < maxIterations; ++iter) {
    // Compute current rotation matrix and derivatives
    const Matrix3<T> R = twoAxisRotationMatrix(axis0, axis1, angles);
    const auto [dR0, dR1] = twoAxisRotationMatrixDerivatives(axis0, axis1, angles);

    // Compute residual matrix and flatten to vector (9x1)
    const Matrix3<T> residualMatrix = R - target;
    const T residualNorm = residualMatrix.squaredNorm();

    // Same convergence/divergence behavior as the scalar version above: the second
    // clause exits both on slow progress and on residual increase, without rolling
    // back the diverging step.
    if (residualNorm < tolerance || (previousResidualNorm - residualNorm) < tolerance) {
      break;
    }
    previousResidualNorm = residualNorm;

    // Flatten residual matrix and derivative matrices using utility functions
    const Eigen::Matrix<T, 9, 1> residualVector = residualMatrix.reshaped(9, 1);

    Eigen::Matrix<T, 9, 2> jacobian;
    jacobian.template block<9, 1>(0, 0) = dR0.reshaped(9, 1);
    jacobian.template block<9, 1>(0, 1) = dR1.reshaped(9, 1);

    // Gauss-Newton step solved as a 9x2 linear least-squares: minimize ||J*delta + r||.
    // No LM damping (lambda) is applied; see TODO above the function.
    const Vector2<T> delta = jacobian.householderQr().solve(-residualVector);

    // Check if solution is valid
    if (!delta.allFinite()) {
      break;
    }

    angles = angles + delta;
  }

  return angles;
}

} // anonymous namespace

template <typename T>
T rotationMatrixToOneAxisEuler(const Matrix3<T>& m, int axis0) {
  MT_CHECK(axis0 >= 0 && axis0 <= 2, "Invalid axis0");

  // Always extract the best single-axis approximation using atan2 as initial guess
  T initialGuess = T(0);
  if (axis0 == 0) { // X-axis rotation
    // For X-axis rotation: R_x(θ) has cos(θ) at (1,1) and (2,2), sin(θ) at (2,1), -sin(θ) at (1,2)
    initialGuess = std::atan2(m(2, 1), m(1, 1));
  } else if (axis0 == 1) { // Y-axis rotation
    // For Y-axis rotation: R_y(θ) has cos(θ) at (0,0) and (2,2), sin(θ) at (0,2), -sin(θ) at (2,0)
    initialGuess = std::atan2(m(0, 2), m(0, 0));
  } else { // Z-axis rotation
    // For Z-axis rotation: R_z(θ) has cos(θ) at (0,0) and (1,1), sin(θ) at (1,0), -sin(θ) at (0,1)
    initialGuess = std::atan2(m(1, 0), m(0, 0));
  }

  // Use Levenberg-Marquardt starting from the atan2 initial guess
  // For exact single-axis rotations, this will converge immediately
  // For non-exact cases, this provides a much better starting point
  return levenbergMarquardtSolveScalar(m, axis0, initialGuess);
}

template <typename T>
Vector2<T> rotationMatrixToTwoAxisEuler(const Matrix3<T>& m, int axis0, int axis1) {
  MT_CHECK(axis0 != axis1, "Can't call two-axis rotation with the same axis");
  MT_CHECK(axis0 >= 0 && axis0 <= 2, "Invalid axis0");
  MT_CHECK(axis1 >= 0 && axis1 <= 2, "Invalid axis1");

  // Use Levenberg-Marquardt to find optimal two-axis rotation
  return levenbergMarquardtSolveTwoAxis(
      m,
      axis0,
      axis1,
      Eigen::Vector2<T>{
          rotationMatrixToOneAxisEuler(m, axis0), rotationMatrixToOneAxisEuler(m, axis1)});
}

template float rotationMatrixToOneAxisEuler(const Matrix3<float>& m, int axis0);
template double rotationMatrixToOneAxisEuler(const Matrix3<double>& m, int axis0);

template Vector2<float> rotationMatrixToTwoAxisEuler(const Matrix3<float>& m, int axis0, int axis1);
template Vector2<double>
rotationMatrixToTwoAxisEuler(const Matrix3<double>& m, int axis0, int axis1);

} // namespace momentum
