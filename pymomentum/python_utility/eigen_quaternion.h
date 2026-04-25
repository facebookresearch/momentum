/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
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
  using Vector4 = Eigen::Vector4<Scalar>;

  /// Standard pybind11 type caster interface
  PYBIND11_TYPE_CASTER(Type, const_name("numpy.ndarray[4]"));

  /// Python -> C++ conversion: read [x, y, z, w] from numpy array
  bool load(handle src, bool /*convert*/) {
    auto array =
        pybind11::array_t<Scalar, pybind11::array::c_style | pybind11::array::forcecast>::ensure(
            src);
    if (!array || array.size() != 4) {
      return false;
    }
    const Scalar* data = array.data();
    // Array is [x, y, z, w], Quaternion ctor is (w, x, y, z)
    value = Type(data[3], data[0], data[1], data[2]);
    return true;
  }

  /// C++ -> Python conversion: return [x, y, z, w] numpy array
  static handle cast(const Type& src, return_value_policy /*policy*/, handle /*parent*/) {
    auto result = pybind11::array_t<Scalar>(4);
    auto* buf = result.mutable_data();
    buf[0] = src.x();
    buf[1] = src.y();
    buf[2] = src.z();
    buf[3] = src.w();
    return result.release();
  }

  /// Support for const pointer return
  static handle cast(const Type* src, return_value_policy policy, handle parent) {
    return cast(*src, policy, parent);
  }
};

} // namespace pybind11::detail
