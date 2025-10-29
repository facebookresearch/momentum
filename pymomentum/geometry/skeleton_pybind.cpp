/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/skeleton_pybind.h"

#include "pymomentum/geometry/momentum_geometry.h"

#include <momentum/character/joint.h>
#include <momentum/character/skeleton.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

#include <algorithm>

#include <fmt/format.h>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

void registerJointBindings(py::class_<mm::Joint>& jointClass) {
  // =====================================================
  // momentum::Joint
  // - name
  // - parent
  // - preRotation ((x, y, z), w)
  // - translationOffset
  // =====================================================
  jointClass
      .def(
          py::init([](const std::string& name,
                      const int parent,
                      const Eigen::Vector4f& preRotation,
                      const Eigen::Vector3f& translationOffset) {
            return momentum::Joint{
                name,
                parent == -1 ? mm::kInvalidIndex : parent,
                {preRotation[3], preRotation[0], preRotation[1], preRotation[2]},
                translationOffset};
          }),
          py::arg("name"),
          py::arg("parent"),
          py::arg("pre_rotation"),
          py::arg("translation_offset"))
      .def_property_readonly(
          "name",
          [](const mm::Joint& joint) { return joint.name; },
          "Returns the name of the joint.")
      .def_property_readonly(
          "parent",
          [](const mm::Joint& joint) -> int {
            if (joint.parent == mm::kInvalidIndex) {
              return -1;
            } else {
              return static_cast<int>(joint.parent);
            }
          },
          "Returns the index of the parent joint (-1 if it has no parent)")
      .def_property_readonly(
          "pre_rotation",
          [](const mm::Joint& joint) {
            return Eigen::Vector4f(
                joint.preRotation.x(),
                joint.preRotation.y(),
                joint.preRotation.z(),
                joint.preRotation.w());
          },
          "Returns the pre-rotation for this joint in default pose of the character. Quaternion format: (x, y, z, w)")
      .def_property_readonly(
          "pre_rotation_matrix",
          [](const mm::Joint& joint) { return joint.preRotation.toRotationMatrix(); })
      .def_property_readonly(
          "translation_offset",
          [](const mm::Joint& joint) { return joint.translationOffset; },
          "Returns the translation offset for this joint in default pose of the character.")
      .def("__repr__", [](const mm::Joint& j) {
        return fmt::format(
            "Joint(name='{}', parent={}, offset=[{} {} {}], pre_rotation=[{} {} {} {}])",
            j.name,
            j.parent == mm::kInvalidIndex ? -1 : static_cast<int>(j.parent),
            j.translationOffset.x(),
            j.translationOffset.y(),
            j.translationOffset.z(),
            j.preRotation.x(),
            j.preRotation.y(),
            j.preRotation.z(),
            j.preRotation.w());
      });
}

void registerSkeletonBindings(py::class_<momentum::Skeleton>& skeletonClass) {
  // =====================================================
  // momentum::Skeleton
  // - size
  // - joint_names
  // - joint_parents
  // - get_parent(joint_index)
  // - get_child_joints(rootJointIndex, recursive)
  // - upper_body_joints
  // =====================================================
  skeletonClass
      .def(
          py::init([](const std::vector<mm::Joint>& jointList) { return mm::Skeleton(jointList); }),
          py::arg("joint_list"))
      .def_property_readonly(
          "size",
          [](const mm::Skeleton& skel) { return skel.joints.size(); },
          "Returns the number of joints in the skeleton.")
      .def(
          "__len__",
          [](const mm::Skeleton& skel) { return skel.joints.size(); },
          "Returns the number of joints in the skeleton.")
      .def_property_readonly(
          "joint_names",
          [](const mm::Skeleton& skel) { return skel.getJointNames(); },
          "Returns a list of joint names in the skeleton.")
      .def_property_readonly(
          "joint_parents",
          [](const mm::Skeleton& skel) -> std::vector<int64_t> {
            // For the root joint, we'll use -1 as the reported parent; this
            // just makes a lot more sense in a Python context where it would
            // be hard to compare against SIZE_MAX (and you're relying on the
            // typesystem to keep it as a uint64_t instead of an int64_t which
            // seems unreliable).
            std::vector<int64_t> result(skel.joints.size(), -1);
            for (size_t i = 0; i < skel.joints.size(); ++i) {
              const auto parent = skel.joints[i].parent;
              if (parent != momentum::kInvalidIndex) {
                result[i] = parent;
              }
            }
            return result;
          },
          ":return: the parent of each joint in the skeleton.  The root joint has parent -1.")
      .def(
          "joint_index",
          [](const mm::Skeleton& skel, const std::string& name, bool allow_missing = false) -> int {
            auto result = skel.getJointIdByName(name);
            if (result == momentum::kInvalidIndex) {
              if (allow_missing) {
                return -1;
              } else {
                MT_THROW("Joint '{}' not found in skeleton.", name);
              }
            } else {
              return result;
            }
          },
          "Get the joint index for a given joint name.  Returns -1 if joint is not found and allow_missing is True.",
          py::arg("name"),
          py::arg("allow_missing") = false)
      .def(
          "get_parent",
          [](const mm::Skeleton& skel, int jointIndex) -> int64_t {
            MT_THROW_IF(
                jointIndex < 0 || jointIndex >= skel.joints.size(),
                "get_parent() called with invalid joint index {}",
                jointIndex);
            const auto parent = skel.joints[jointIndex].parent;
            if (parent == momentum::kInvalidIndex) {
              return -1;
            } else {
              return static_cast<int64_t>(parent);
            }
          },
          R"(Get the parent joint index of the given joint. Return -1 for root.

:param joint_index: the index of a skeleton joint.
:return: The index of the parent joint, or -1 if it is the root of the skeleton. )",
          py::arg("joint_index"))
      .def(
          "get_child_joints",
          &mm::Skeleton::getChildrenJoints,
          R"(Find all joints parented under the given joint.

:return: A list of integers, one per joint. )",
          py::arg("root_joint_index"),
          py::arg("recursive"))
      .def(
          "is_ancestor",
          &mm::Skeleton::isAncestor,
          R"(Checks if one joint is an ancestor of another, inclusive.

:param joint_index: The index of a skeleton joint.
:param ancestor_joint_index: The index of a possible ancestor joint.

:return: true if ancestorJointId is an ancestor of jointId; that is,
    if jointId is in the tree rooted at ancestorJointId.
    Note that a joint is considered to be its own ancestor; that is,
    isAncestor(id, id) returns true. )",
          py::arg("joint_index"),
          py::arg("ancestor_joint_index"))
      .def_property_readonly(
          "upper_body_joints",
          &getUpperBodyJoints,
          R"(Convenience function to get all upper-body joints (defined as those parented under 'b_spine0').

:return: A list of integers, one per joint.)")
      .def_property_readonly(
          "offsets",
          [](const mm::Skeleton& skeleton) {
            std::vector<Eigen::Vector3f> translationOffsets;
            std::transform(
                skeleton.joints.cbegin(),
                skeleton.joints.cend(),
                std::back_inserter(translationOffsets),
                [](const mm::Joint& joint) { return joint.translationOffset; });
            return pymomentum::asArray(translationOffsets);
          },
          "Returns skeleton joint offsets tensor for all joints (num_joints, 3")
      .def_property_readonly(
          "pre_rotations",
          [](const mm::Skeleton& skeleton) {
            std::vector<Eigen::Vector4f> preRotations;
            std::transform(
                skeleton.joints.cbegin(),
                skeleton.joints.cend(),
                std::back_inserter(preRotations),
                [](const mm::Joint& joint) { return joint.preRotation.coeffs(); });
            return pymomentum::asArray(preRotations);
          },
          "Returns skeleton joint offsets tensor for all joints shape: (num_joints, 4)")
      .def_readonly("joints", &mm::Skeleton::joints)
      .def("__repr__", [](const mm::Skeleton& s) {
        return fmt::format("Skeleton(joints={})", s.joints.size());
      });
}

} // namespace pymomentum
