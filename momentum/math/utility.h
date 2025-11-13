/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/constants.h>
#include <momentum/math/types.h>

#include <gsl/gsl>

#ifdef _WIN32
#include <malloc.h>
#endif // _WIN32

namespace momentum {

/// NaN check function that isn't optimized away by fastmath
///
/// Unlike std::isnan(), this implementation is not affected by compiler optimizations
/// when using fast math options
[[nodiscard]] inline bool IsNanNoOpt(float f) {
  union {
    float f;
    std::uint32_t x;
  } u = {f};
  return (u.x << 1) > 0xff000000u;
}

/// @name Math Utilities
/// @{

/// Creates a ParameterSet with all parameters set to active
inline ParameterSet allParams() {
  ParameterSet params;
  params.set();
  return params;
}

/// Computes the square of a value
///
/// Convenience function for x * x
template <class T>
inline T sqr(const T& x) {
  return x * x;
};

/// Determines if two values are approximately equal within a given tolerance.
template <typename T = float>
[[nodiscard]] bool isApprox(T l, T r, T eps = Eps<T>(1e-4, 1e-6)) noexcept {
  return (std::abs(l - r) < eps);
};

/// @}

/// @name Matrix Utilities
/// @{

/// Calculates the pseudo-inverse of a matrix
///
/// Uses singular value decomposition (SVD) to compute the Moore-Penrose pseudo-inverse
/// @tparam T The scalar type
/// @param mat Input matrix
/// @return The pseudo-inverse of the input matrix
template <typename T>
MatrixX<T> pseudoInverse(const MatrixX<T>& mat);

/// Calculates the pseudo-inverse of a sparse matrix
///
/// Converts the sparse matrix to a dense matrix before computing the pseudo-inverse
/// @tparam T The scalar type
/// @param mat Input sparse matrix
/// @return The pseudo-inverse as a dense matrix
template <typename T>
MatrixX<T> pseudoInverse(const SparseMatrix<T>& mat);

/// @}

/// @name Geometry Utilities
/// @{

/// Computes the exponential map from a rotation vector (axis-angle in R^3) to a quaternion.
///
/// The exponential map takes a 3D rotation vector (where the direction is the rotation axis
/// and the magnitude is the rotation angle in radians) and returns the corresponding unit
/// quaternion. This is numerically robust for small angles using Taylor series expansion.
///
/// Reference: "Practical Parameterization of Rotations Using the Exponential Map"
/// by F. Sebastian Grassia, Journal of Graphics Tools, 1998.
///
/// @tparam T The scalar type
/// @param v The rotation vector (axis-angle representation)
/// @return The corresponding unit quaternion
template <typename T>
Quaternion<T> quaternionExpMap(const Vector3<T>& v);

/// Computes the logarithmic map from a quaternion to a rotation vector (axis-angle in R^3).
///
/// The logarithmic map takes a unit quaternion and returns the corresponding 3D rotation
/// vector (where the direction is the rotation axis and the magnitude is the rotation angle
/// in radians). This is numerically robust for small angles using Taylor series expansion.
/// For quaternions close to -I (180 degree rotation), a consistent branch is chosen.
///
/// Reference: "Practical Parameterization of Rotations Using the Exponential Map"
/// by F. Sebastian Grassia, Journal of Graphics Tools, 1998.
///
/// @tparam T The scalar type
/// @param q The input quaternion (should be normalized)
/// @return The corresponding rotation vector (axis-angle representation)
template <typename T>
Vector3<T> quaternionLogMap(const Quaternion<T>& q);

/// Computes the derivative of the logarithmic map with respect to quaternion components.
///
/// Given a quaternion q with coeffs [x, y, z, w], computes the Jacobian matrix J where:
/// J(i,j) = d(log(q)[i]) / dq.coeffs()[j]
/// where log(q) is the 3D rotation vector and q.coeffs() = [x, y, z, w] (Eigen's convention).
///
/// This is the fundamental building block for computing derivatives of functions
/// involving logmap through the chain rule.
///
/// Reference: "Practical Parameterization of Rotations Using the Exponential Map"
/// by F. Sebastian Grassia, Journal of Graphics Tools, 1998.
///
/// @tparam T The scalar type
/// @param q The input quaternion (should be normalized)
/// @return 3x4 Jacobian matrix where column j is the gradient w.r.t. q.coeffs()[j],
///         i.e., columns are ordered as [d/dx, d/dy, d/dz, d/dw]
template <typename T>
Eigen::Matrix<T, 3, 4> quaternionLogMapDerivative(const Quaternion<T>& q);

/// The Euler angles convention.
enum class EulerConvention {
  /// The intrinsic convention. Intrinsic rotations (also called local, relative, or body-fixed
  /// rotations) are performed around the axes of the coordinate system that is attached to the
  /// rotating object.
  ///
  /// Intrinsic rotations are usually represented by a sequence of three letters, like XYZ, which
  /// means rotation around the X-axis, followed by rotation around the Y-axis, and finally rotation
  /// around the Z-axis. For example, intrinsic XYZ is equivalent to RotX(a) * RotY(b) * RotZ(c).
  Intrinsic,

  /// The extrinsic convention. Extrinsic rotations (also called global, absolute, or space-fixed
  /// rotations) are performed around the axes of a fixed coordinate system.
  ///
  /// Extrinsic rotations are often represented by uppercase letters, like XYZ, which means rotation
  /// around the fixed X-axis, followed by rotation around the fixed Y-axis, and finally rotation
  /// around the fixed Z-axis. For example, extrinsic XYZ is equivalent to RotZ(c) * RotY(b) *
  /// RotX(a).
  Extrinsic,
};

/// Converts rotation matrix to Euler angles
///
/// @tparam T The scalar type.
/// @param[in] m The rotation matrix to convert.
/// @param[in] axis0 The index of the first rotation's axis, one of {0, 1, 2}.
/// @param[in] axis1 The index of the second rotation's axis, one of {0, 1, 2}.
/// @param[in] axis2 The index of the third rotation's axis, one of {0, 1, 2}.
/// @param[in] convention The Euler angles convention.
template <typename T>
[[nodiscard]] Vector3<T> rotationMatrixToEuler(
    const Matrix3<T>& m,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention = EulerConvention::Intrinsic);

/// An optimized version of rotationMatrixToEuler(m, 0, 1, 2, convention) or
/// rotationMatrixToEuler(m, 2, 1, 0, convention).reverse().
template <typename T>
[[nodiscard]] Vector3<T> rotationMatrixToEulerXYZ(
    const Matrix3<T>& m,
    EulerConvention convention = EulerConvention::Intrinsic);

/// An optimized version of rotationMatrixToEuler(m, 2, 1, 0, convention) or
/// rotationMatrixToEuler(m, 0, 1, 2, convention).reverse().
template <typename T>
[[nodiscard]] Vector3<T> rotationMatrixToEulerZYX(
    const Matrix3<T>& m,
    EulerConvention convention = EulerConvention::Intrinsic);

/// Fits a two-axis Euler rotation to a rotation matrix
///
/// Finds the best-fit two-axis Euler angles (axis0 then axis1) that minimize the
/// L2 distance to the target rotation matrix using Levenberg-Marquardt optimization.
///
/// @tparam T The scalar type.
/// @param[in] m The target rotation matrix.
/// @param[in] axis0 The index of the first rotation axis, one of {0, 1, 2}.
/// @param[in] axis1 The index of the second rotation axis, one of {0, 1, 2}.
/// @return A 2D vector containing the fitted Euler angles [angle0, angle1].
template <typename T>
[[nodiscard]] Vector2<T> rotationMatrixToTwoAxisEuler(const Matrix3<T>& m, int axis0, int axis1);

/// Fits a single-axis Euler rotation to a rotation matrix
///
/// Finds the best-fit single-axis rotation angle that minimizes the L2 distance
/// to the target rotation matrix using Levenberg-Marquardt optimization. This is
/// NOT a direct conversion - if the input matrix cannot be exactly represented as
/// a single-axis rotation, this function returns the closest approximation.
///
/// For exact single-axis rotations, the result will be precise. For general rotation
/// matrices (e.g., those involving rotations around multiple axes), the result is the
/// best single-axis approximation in the least-squares sense.
///
/// @tparam T The scalar type.
/// @param[in] m The target rotation matrix.
/// @param[in] axis0 The rotation axis index, one of {0, 1, 2} (X, Y, Z).
/// @return The fitted rotation angle around the specified axis.
template <typename T>
[[nodiscard]] T rotationMatrixToOneAxisEuler(const Matrix3<T>& m, int axis0);

/// Converts Euler angles to quaternion
///
/// @tparam T The scalar type.
/// @param[in] angles The 3-dimensional vector of Euler angles in order.
/// @param[in] axis0 The index of the first rotation's axis, one of {0, 1, 2}.
/// @param[in] axis1 The index of the second rotation's axis, one of {0, 1, 2}.
/// @param[in] axis2 The index of the third rotation's axis, one of {0, 1, 2}.
/// @param[in] convention The Euler angles convention.
template <typename T>
[[nodiscard]] Quaternion<T> eulerToQuaternion(
    const Vector3<T>& angles,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention = EulerConvention::Intrinsic);

/// Converts Euler angles to rotation matrix
///
/// @tparam T The scalar type.
/// @param[in] angles The 3-dimensional vector of Euler angles in order.
/// @param[in] axis0 The index of the first rotation's axis, one of {0, 1, 2}.
/// @param[in] axis1 The index of the second rotation's axis, one of {0, 1, 2}.
/// @param[in] axis2 The index of the third rotation's axis, one of {0, 1, 2}.
/// @param[in] convention The Euler angles convention.
template <typename T>
[[nodiscard]] Matrix3<T> eulerToRotationMatrix(
    const Vector3<T>& angles,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention = EulerConvention::Intrinsic);

/// An optimized version of eulerToRotationMatrix(angles, 0, 1, 2, convention) or
/// eulerToRotationMatrix(angles.reverse(), 2, 1, 0, convention).
template <typename T>
[[nodiscard]] Matrix3<T> eulerXYZToRotationMatrix(
    const Vector3<T>& angles,
    EulerConvention convention = EulerConvention::Intrinsic);

/// An optimized version of eulerToRotationMatrix(angles, 2, 1, 0, convention) or
/// eulerToRotationMatrix(angles.reverse(), 0, 1, 2, convention).
template <typename T>
[[nodiscard]] Matrix3<T> eulerZYXToRotationMatrix(
    const Vector3<T>& angles,
    EulerConvention convention = EulerConvention::Intrinsic);

/// Converts a quaternion to Euler angles using the Extrinsic XYZ convention
///
/// The returned angles [x, y, z] are such that applying Extrinsic rotations
/// RotX(x) then RotY(y) then RotZ(z) (i.e., about fixed world axes) reproduces
/// the input quaternion's rotation. Empirically verified by tests.
///
/// Note: This matches the convention used in skeleton_state (skel_state): Extrinsic XYZ.
///
/// @tparam T The scalar type
/// @param q Input quaternion
/// @return Vector of Euler angles [x, y, z]
template <typename T>
Vector3<T> quaternionToEuler(const Quaternion<T>& q);

/// Computes the weighted average of multiple quaternions
///
/// Uses the method described in Markley et al. "Averaging Quaternions"
/// @param q Array of quaternions to average
/// @param w Optional weights for each quaternion (defaults to equal weights)
/// @return The average quaternion
Quaternionf quaternionAverage(
    std::span<const Quaternionf> q,
    std::span<const float> w = std::vector<float>());

/// Returns the closest points on two line segments where the line segments are represented in
/// origin and direction
///
/// @tparam T Scalar type
/// @param o1 Origin of the first line segment
/// @param d1 Direction of the first line segment
/// @param o2 Origin of the second line segment
/// @param d2 Direction of the second line segment
/// @param maxDist Maximum distance to check
/// @return [success, distance, [dist0, dist1]]:
/// - success: Whether the closest points were successfully found. On false, the rest return values
/// are invalid.
/// - distance: Distance between the closest points
/// - [dist0, dist1]: Distances of the closest points from their origins to the directions
template <typename T>
[[nodiscard]] std::tuple<bool, T, Eigen::Vector2<T>> closestPointsOnSegments(
    const Eigen::Vector3<T>& o1,
    const Eigen::Vector3<T>& d1,
    const Eigen::Vector3<T>& o2,
    const Eigen::Vector3<T>& d2,
    T maxDist = std::numeric_limits<T>::max());

/// Creates a skew-symmetric matrix for computing cross products
///
/// Returns a matrix M such that M*u = v×u for any vector u
/// @tparam T The scalar type
/// @param v Input vector
/// @return 3×3 skew-symmetric matrix
template <typename T>
Eigen::Matrix<T, 3, 3> crossProductMatrix(const Eigen::Vector3<T>& v) {
  Eigen::Matrix<T, 3, 3> result;
  result << T(0), -v.z(), v.y(), v.z(), T(0), -v.x(), -v.y(), v.x(), T(0);
  return result;
}

/// @}

} // namespace momentum
