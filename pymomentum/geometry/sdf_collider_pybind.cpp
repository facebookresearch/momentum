/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/sdf_collider_pybind.h"

#include <axel/SignedDistanceField.h>
#include <momentum/character/sdf_collision_geometry.h>
#include <momentum/math/constants.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <memory>
#include <optional>

#include <fmt/format.h>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

namespace {

// Convert Python parent index (int, -1 = world-space) to C++ (size_t, kInvalidIndex = world-space)
size_t parentFromPython(int parent) {
  if (parent < 0) {
    return mm::kInvalidIndex;
  }
  return static_cast<size_t>(parent);
}

// Convert C++ parent index (size_t, kInvalidIndex = world-space) to Python (int, -1 = world-space)
int parentToPython(size_t parent) {
  if (parent == mm::kInvalidIndex) {
    return -1;
  }
  return static_cast<int>(parent);
}

} // namespace

void registerSDFColliderBindings(py::class_<mm::SDFColliderT<float>>& sdfColliderClass) {
  // =====================================================
  // momentum::SDFColliderT<float>
  // - transformation (exposed as translation + rotation)
  // - parent
  // - sdf
  // =====================================================
  sdfColliderClass
      // Default constructor
      .def(py::init<>(), "Create an empty SDF collider (world-space, no SDF).")
      // Parameterized constructor
      .def(
          py::init([](const std::optional<Eigen::Vector3f>& translation,
                      const std::optional<Eigen::Quaternionf>& rotation,
                      int parent,
                      std::optional<std::shared_ptr<axel::SignedDistanceField<float>>> sdf) {
            mm::TransformT<float> transform;
            if (translation.has_value()) {
              transform.translation = translation.value();
            }
            if (rotation.has_value()) {
              transform.rotation = rotation.value();
            }
            // Convert optional<shared_ptr<T>> to shared_ptr<const T>
            std::shared_ptr<const axel::SignedDistanceField<float>> sdfPtr;
            if (sdf.has_value()) {
              sdfPtr = sdf.value();
            }
            return mm::SDFColliderT<float>(transform, parentFromPython(parent), std::move(sdfPtr));
          }),
          R"(Create an SDF collider attached to a skeleton joint.

An SDF collider represents a signed distance field volume attached to a skeleton joint
for collision detection. The SDF data is shared via shared_ptr to avoid expensive copies
of potentially large SDF volumes.

:param translation: Translation offset from parent joint (default: [0, 0, 0]).
:param rotation: Rotation quaternion [x, y, z, w] relative to parent (default: identity).
:param parent: Parent joint index. Use -1 for world-space SDFs (default: -1).
:param sdf: The signed distance field data (:class:`pymomentum.axel.SignedDistanceField`).
)",
          py::arg("translation") = std::nullopt,
          py::arg("rotation") = std::nullopt,
          py::arg("parent") = -1,
          py::arg("sdf") = std::nullopt)
      .def_property_readonly(
          "translation",
          [](const mm::SDFColliderT<float>& collider) {
            return collider.transformation.translation;
          },
          "Translation offset from parent joint.")
      .def_property_readonly(
          "rotation",
          [](const mm::SDFColliderT<float>& collider) { return collider.transformation.rotation; },
          "Rotation quaternion [x, y, z, w] relative to parent joint.")
      .def_property_readonly(
          "parent",
          [](const mm::SDFColliderT<float>& collider) { return parentToPython(collider.parent); },
          "Parent joint index. -1 indicates world-space.")
      .def_readonly(
          "sdf",
          &mm::SDFColliderT<float>::sdf,
          "The signed distance field data (:class:`pymomentum.axel.SignedDistanceField`).")
      .def(
          "is_valid",
          &mm::SDFColliderT<float>::isValid,
          "Check if the collider has a valid SDF pointer.")
      .def(
          "is_approx",
          &mm::SDFColliderT<float>::isApprox,
          R"(Check if this collider is approximately equal to another.

:param other: The other collider to compare with.
:param tol: Tolerance for floating point comparison (default: 1e-4).
:return: True if colliders are approximately equal.
)",
          py::arg("other"),
          py::arg("tol") = 1e-4f)
      .def("__repr__", [](const mm::SDFColliderT<float>& collider) {
        const Eigen::Vector3f& t = collider.transformation.translation;
        const int pyParent = parentToPython(collider.parent);
        return fmt::format(
            "SDFCollider(translation=[{:.3f}, {:.3f}, {:.3f}], parent={}, valid={})",
            t.x(),
            t.y(),
            t.z(),
            pyParent,
            collider.isValid() ? "True" : "False");
      });
}

} // namespace pymomentum
