/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Geometry>

/// Custom type caster for Eigen::Quaternion that leverages pybind11's existing
/// Eigen::Vector4 conversion machinery. This enables automatic conversion between
/// Eigen::Quaternion and numpy arrays without manual .coeffs() calls.
///
/// Usage: Include this header after <pybind11/eigen.h> in your binding files.
///
/// Example:
///   #include "pymomentum/python_utility/eigen_quaternion.h"
///
///   Eigen::Quaternionf getRotation() { return Eigen::Quaternionf::Identity(); }
///   void setRotation(const Eigen::Quaternionf& q) { /* use q */ }
///
/// Python usage:
///   q = module.get_rotation()  # Returns np.array([x, y, z, w])
///   module.set_rotation(np.array([0, 0, 0, 1], dtype=np.float32))

namespace pybind11::detail {

/// Type caster for Eigen::Quaternion that uses the existing Vector4 conversion
template <typename Scalar>
struct type_caster<Eigen::Quaternion<Scalar>> {
 public:
  using Type = Eigen::Quaternion<Scalar>;
  using Vector4 = Eigen::Vector<Scalar, 4>;

  /// Standard pybind11 type caster interface
  PYBIND11_TYPE_CASTER(Type, const_name("numpy.ndarray[4]"));

  /// Python -> C++ conversion: delegate to Vector4 caster
  bool load(handle src, bool convert) {
    // Use the existing pybind11 Vector4 type caster
    type_caster<Vector4> vec_caster;
    if (!vec_caster.load(src, convert)) {
      return false;
    }

    // Extract the Vector4 and construct quaternion
    // Vector4 is [x, y, z, w], Quaternion constructor is (w, x, y, z)
    const Vector4& coeffs = vec_caster;
    value = Type(coeffs(3), coeffs(0), coeffs(1), coeffs(2));

    return true;
  }

  /// C++ -> Python conversion: delegate to Vector4 caster
  static handle cast(const Type& src, return_value_policy policy, handle parent) {
    // Get coefficients as Vector4 [x, y, z, w]
    Vector4 coeffs = src.coeffs();

    // Use the existing pybind11 Vector4 type caster
    return type_caster<Vector4>::cast(coeffs, policy, parent);
  }

  /// Support for const pointer return
  static handle cast(const Type* src, return_value_policy policy, handle parent) {
    return cast(*src, policy, parent);
  }
};

} // namespace pybind11::detail
