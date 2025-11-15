/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/locators_pybind.h"

#include <momentum/character/character.h>
#include <momentum/character/locator.h>
#include <momentum/character/skinned_locator.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <optional>
#include <stdexcept>
#include <string>

#include <fmt/format.h>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

void registerLocatorBindings(
    py::class_<mm::Locator>& locatorClass,
    py::class_<mm::SkinnedLocator>& skinnedLocatorClass) {
  // =====================================================
  // momentum::Locator
  // - name
  // - parent
  // - offset
  // =====================================================
  locatorClass
      .def(
          py::init<
              const std::string&,
              const size_t,
              const Eigen::Vector3f&,
              const Eigen::Vector3i&,
              float,
              const Eigen::Vector3f&,
              const Eigen::Vector3f&,
              bool,
              float>(),
          py::arg("name") = "uninitialized",
          py::arg("parent") = mm::kInvalidIndex,
          py::arg("offset") = Eigen::Vector3f::Zero(),
          py::arg("locked") = Eigen::Vector3i::Zero(),
          py::arg("weight") = 1.0f,
          py::arg("limit_origin") = Eigen::Vector3f::Zero(),
          py::arg("limit_weight") = Eigen::Vector3f::Zero(),
          py::arg("attached_to_skin") = false,
          py::arg("skin_offset") = 0.0f)
      .def_readonly("name", &mm::Locator::name, "The locator's name.")
      .def_readonly("parent", &mm::Locator::parent, "The locator's parent joint index.")
      .def_readonly(
          "offset", &mm::Locator::offset, "The locator's offset to parent joint location.")
      .def_readonly(
          "locked",
          &mm::Locator::locked,
          "Flag per axes to indicate whether that axis can be moved during optimization or not.")
      .def_readonly(
          "weight", &mm::Locator::weight, "Weight for this locator during IK optimization.")
      .def_readonly(
          "limit_origin",
          &mm::Locator::limitOrigin,
          "Defines the limit reference position. equal to offset on loading.")
      .def_readonly(
          "limit_weight",
          &mm::Locator::limitWeight,
          "Controls how strongly the locator should maintain its original position.  "
          "Higher values create stronger constraints, zero means completely free.")
      .def_readonly(
          "attached_to_skin",
          &mm::Locator::attachedToSkin,
          "Indicates whether the locator is attached to the skin of a person (e.g. as in mocap tracking), "
          "used to determine whether the locator can safely be converted to a skinned locator.")
      .def_readonly(
          "skin_offset",
          &mm::Locator::skinOffset,
          "Offset from the skin surface, used when trying to solve for body shape using locators.")
      .def("__repr__", [](const mm::Locator& l) {
        return fmt::format(
            "Locator(name={}, parent={}, offset=[{}, {}, {}])",
            l.name,
            l.parent,
            l.offset.x(),
            l.offset.y(),
            l.offset.z());
      });

  // =====================================================
  // momentum::SkinnedLocator
  // - name
  // - parents
  // - skinWeights
  // - position
  // - weight
  // =====================================================
  skinnedLocatorClass
      .def(
          py::init([](const std::string& name,
                      const Eigen::VectorXi& parents,
                      const Eigen::VectorXf& skinWeights,
                      const std::optional<Eigen::Vector3f>& position,
                      float weight) {
            if (parents.size() != skinWeights.size()) {
              throw std::runtime_error("parents and skin_weights must have the same size");
            }

            if (parents.size() > mm::kMaxSkinJoints) {
              throw std::runtime_error(
                  fmt::format(
                      "parents and skin_weights must have at most {} elements",
                      mm::kMaxSkinJoints));
            }

            Eigen::Matrix<uint32_t, mm::kMaxSkinJoints, 1> parentsTmp =
                Eigen::Matrix<uint32_t, mm::kMaxSkinJoints, 1>::Zero();
            Eigen::Matrix<float, mm::kMaxSkinJoints, 1> skinWeightsTmp =
                Eigen::Matrix<float, mm::kMaxSkinJoints, 1>::Zero();

            for (size_t i = 0; i < parents.size(); ++i) {
              if (parents(i) < 0) {
                throw std::runtime_error(
                    "parents must be non-negative, but got " + std::to_string(parents(i)));
              }

              if (skinWeights(i) < 0) {
                throw std::runtime_error(
                    "skin_weights must be non-negative, but got " + std::to_string(skinWeights(i)));
              }
              parentsTmp(i) = parents(i);
              skinWeightsTmp(i) = skinWeights(i);
            }

            return mm::SkinnedLocator(
                name,
                parentsTmp,
                skinWeightsTmp,
                position.value_or(Eigen::Vector3f::Zero()),
                weight);
          }),
          py::arg("name"),
          py::arg("parents"),
          py::arg("skin_weights"),
          py::arg("position") = std::nullopt,
          py::arg("weight") = 1.0f)
      .def_readonly("name", &mm::SkinnedLocator::name, "The skinned locator's name.")
      .def_property_readonly(
          "parents",
          [](const mm::SkinnedLocator& locator) { return locator.parents; },
          "Indices of the parent joints in the skeleton.")
      .def_property_readonly(
          "skin_weights",
          [](const mm::SkinnedLocator& locator) { return locator.skinWeights; },
          "Skinning weights for the parent joints.")
      .def_readonly(
          "position",
          &mm::SkinnedLocator::position,
          "Position relative to rest pose of the character.")
      .def_readonly(
          "weight",
          &mm::SkinnedLocator::weight,
          "Influence weight of this locator when used in constraints.")
      .def("__repr__", [](const mm::SkinnedLocator& l) {
        return fmt::format(
            "SkinnedLocator(name={}, position=[{}, {}, {}], weight={})",
            l.name,
            l.position.x(),
            l.position.y(),
            l.position.z(),
            l.weight);
      });
}

} // namespace pymomentum
