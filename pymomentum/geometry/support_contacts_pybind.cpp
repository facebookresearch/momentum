/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/support_contacts_pybind.h"

#include <momentum/character/character.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/plane_collision_query.h>
#include <momentum/character_solver/support_contacts.h>
#include <momentum/common/exception.h>
#include <momentum/math/support_polygon.h>
#include <momentum/math/transform.h>
#include <momentum/math/utility.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {
namespace {

using SkeletonStateArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

mm::JointStateList jointStatesFromSkeletonStateArray(
    const mm::Character& character,
    const SkeletonStateArray& skeletonState) {
  MT_THROW_IF(
      skeletonState.ndim() != 2,
      "Expected skeleton_state to have shape (n_joints, 8), got {} dimensions",
      skeletonState.ndim());
  MT_THROW_IF(
      skeletonState.shape(1) != 8,
      "Expected skeleton_state to have shape (n_joints, 8), got second dimension {}",
      skeletonState.shape(1));
  MT_THROW_IF(
      skeletonState.shape(0) != static_cast<py::ssize_t>(character.skeleton.joints.size()),
      "Expected {} joints in skeleton_state, got {}",
      character.skeleton.joints.size(),
      skeletonState.shape(0));

  const auto state = skeletonState.unchecked<2>();
  mm::JointStateList jointStates;
  jointStates.reserve(static_cast<size_t>(skeletonState.shape(0)));
  for (py::ssize_t iJoint = 0; iJoint < skeletonState.shape(0); ++iJoint) {
    const Eigen::Vector3f translation{
        state(iJoint, 0),
        state(iJoint, 1),
        state(iJoint, 2),
    };
    const Eigen::Quaternionf rotation{
        state(iJoint, 6),
        state(iJoint, 3),
        state(iJoint, 4),
        state(iJoint, 5),
    };
    const float scale = state(iJoint, 7);
    MT_THROW_IF(
        !translation.allFinite() || !rotation.coeffs().allFinite() || !std::isfinite(scale),
        "Skeleton state joint {} must contain only finite values",
        iJoint);
    MT_THROW_IF(
        rotation.norm() <= mm::Eps<float>(1e-8f),
        "Skeleton state joint {} rotation norm must be greater than 1e-8",
        iJoint);
    MT_THROW_IF(
        scale <= 0.0f, "Skeleton state joint {} scale must be positive, got {}", iJoint, scale);

    mm::JointState jointState;
    // Support-contact geometry only reads world transforms from these states.
    jointState.transform = mm::Transform(translation, rotation.normalized(), scale);
    jointStates.push_back(jointState);
  }
  return jointStates;
}

std::vector<mm::Vector3f> supportPointsFromPython(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& supportPoints) {
  MT_THROW_IF(
      supportPoints.ndim() != 2,
      "Expected support_points to have shape (n_points, 3), got {} dimensions",
      supportPoints.ndim());
  MT_THROW_IF(
      supportPoints.shape(1) != 3,
      "Expected support_points to have shape (n_points, 3), got second dimension {}",
      supportPoints.shape(1));

  const auto points = supportPoints.unchecked<2>();
  std::vector<mm::Vector3f> result;
  result.reserve(static_cast<size_t>(supportPoints.shape(0)));
  for (py::ssize_t iPoint = 0; iPoint < supportPoints.shape(0); ++iPoint) {
    mm::Vector3f point{points(iPoint, 0), points(iPoint, 1), points(iPoint, 2)};
    MT_THROW_IF(
        !point.allFinite(), "support_points row {} must contain only finite values", iPoint);
    result.push_back(point);
  }
  return result;
}

py::array_t<float> vector3Array(const std::vector<mm::Vector3f>& values) {
  py::array_t<float> result({static_cast<py::ssize_t>(values.size()), static_cast<py::ssize_t>(3)});
  auto resultAccessor = result.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
    for (py::ssize_t j = 0; j < 3; ++j) {
      resultAccessor(i, j) = values[static_cast<size_t>(i)](static_cast<int>(j));
    }
  }
  return result;
}

int64_t parentIndexToPython(const size_t parentJoint) {
  MT_THROW_IF(
      parentJoint > static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "Parent joint index {} cannot be represented as a Python int64",
      parentJoint);
  return static_cast<int64_t>(parentJoint);
}

py::array_t<int64_t> int64Array(const std::vector<int64_t>& values) {
  py::array_t<int64_t> result(static_cast<py::ssize_t>(values.size()));
  auto resultAccessor = result.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
    resultAccessor(i) = values[static_cast<size_t>(i)];
  }
  return result;
}

py::array_t<float> supportPolygonArrayFromHull(
    const std::vector<mm::Vector2f>& hull,
    const mm::SupportPlane& supportPlane) {
  py::array_t<float> result({static_cast<py::ssize_t>(hull.size()), static_cast<py::ssize_t>(3)});
  auto resultAccessor = result.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(hull.size()); ++i) {
    const mm::Vector3f point = supportPlane.pointFromCoordinates(hull[static_cast<size_t>(i)]);
    for (py::ssize_t j = 0; j < 3; ++j) {
      resultAccessor(i, j) = point(static_cast<int>(j));
    }
  }
  return result;
}

mm::SupportPlane supportPlaneFromPython(
    const std::optional<mm::Vector3f>& planeNormal,
    const float planeOffset,
    const std::optional<mm::Vector3f>& planeUAxis = std::nullopt) {
  const mm::Vector3f normal = planeNormal.value_or(mm::Vector3f::UnitY());
  MT_THROW_IF(!normal.allFinite(), "plane_normal must contain only finite values");
  MT_THROW_IF(!std::isfinite(planeOffset), "plane_offset must be finite");
  const float normalNorm = normal.norm();
  MT_THROW_IF(
      normalNorm <= mm::Eps<float>(1e-8f), "plane_normal magnitude must be greater than 1e-8");
  const mm::Vector3f normalizedNormal = normal / normalNorm;
  const float normalizedOffset = planeOffset / normalNorm;

  if (!planeUAxis.has_value()) {
    return mm::SupportPlane(normalizedNormal, normalizedOffset);
  }

  const mm::Vector3f& uAxis = *planeUAxis;
  MT_THROW_IF(!uAxis.allFinite(), "plane_u_axis must contain only finite values");
  const float uAxisNorm = uAxis.norm();
  MT_THROW_IF(
      uAxisNorm <= mm::Eps<float>(1e-8f), "plane_u_axis magnitude must be greater than 1e-8");
  const mm::Vector3f tangent = uAxis - normalizedNormal * uAxis.dot(normalizedNormal);
  const float tangentNorm = tangent.norm();
  MT_THROW_IF(
      tangentNorm <= mm::Eps<float>(1e-8f), "plane_u_axis must not be parallel to plane_normal");

  return mm::SupportPlane(normalizedNormal, normalizedOffset, tangent / tangentNorm);
}

struct ContactArrays {
  std::vector<mm::Vector3f> positions;
  std::vector<int64_t> parents;
  std::vector<mm::Vector3f> parentOffsets;
  std::vector<mm::Vector3f> floorPositions;
};

ContactArrays contactArraysFromSupportContacts(
    const mm::SupportContactList& supportContacts,
    const bool includeFloorPositions) {
  ContactArrays result;
  result.positions.reserve(supportContacts.contacts.size());
  result.parents.reserve(supportContacts.contacts.size());
  result.parentOffsets.reserve(supportContacts.contacts.size());
  if (includeFloorPositions) {
    result.floorPositions = supportContacts.floorLocatorPositions;
  }
  for (const mm::SupportContact& contact : supportContacts.contacts) {
    result.positions.push_back(contact.position);
    result.parents.push_back(parentIndexToPython(contact.parentJoint));
    result.parentOffsets.push_back(contact.parentOffset);
  }
  return result;
}

py::tuple contactArraysToPython(const ContactArrays& contacts, bool includeFloorPositions) {
  py::tuple result(includeFloorPositions ? 4 : 3);
  result[0] = vector3Array(contacts.positions);
  result[1] = int64Array(contacts.parents);
  result[2] = vector3Array(contacts.parentOffsets);
  if (includeFloorPositions) {
    result[3] = vector3Array(contacts.floorPositions);
  }
  return result;
}

py::tuple planeCollisionContactsByParent(
    const mm::Character& character,
    const SkeletonStateArray& skeletonState,
    const float contactMargin,
    const std::optional<mm::Vector3f>& planeNormal,
    const float planeOffset) {
  MT_THROW_IF(
      !std::isfinite(contactMargin) || contactMargin < 0.0f,
      "contact_margin must be finite and non-negative, got {}",
      contactMargin);

  ContactArrays result;
  const mm::JointStateList jointStates =
      jointStatesFromSkeletonStateArray(character, skeletonState);
  const mm::SupportPlane supportPlane = supportPlaneFromPython(planeNormal, planeOffset);
  if (character.collision && !character.collision->empty()) {
    const std::span<const mm::JointState> jointStateSpan{jointStates.data(), jointStates.size()};
    mm::PlaneCollisionQuery collisionQuery(character, supportPlane.normal, supportPlane.offset);
    mm::SupportContactList supportContacts;
    supportContacts.contacts =
        mm::computePlaneCollisionSupportContacts(collisionQuery, jointStateSpan, contactMargin);
    result = contactArraysFromSupportContacts(supportContacts, /*includeFloorPositions=*/false);
  }
  return contactArraysToPython(result, /*includeFloorPositions=*/false);
}

void validateContactHeight(const float contactHeight) {
  MT_THROW_IF(
      !std::isfinite(contactHeight) || contactHeight < 0.0f,
      "contact_height must be finite and non-negative, got {}",
      contactHeight);
}

ContactArrays supportContactArrays(
    const mm::Character& character,
    const SkeletonStateArray& skeletonState,
    const float contactHeight,
    const bool includeFloorPositions,
    const mm::SupportPlane& supportPlane) {
  validateContactHeight(contactHeight);
  const mm::JointStateList jointStates =
      jointStatesFromSkeletonStateArray(character, skeletonState);
  const std::span<const mm::JointState> jointStateSpan{jointStates.data(), jointStates.size()};
  const mm::SupportContactList supportContacts = mm::computeSupportContacts(
      character,
      jointStateSpan,
      contactHeight,
      static_cast<mm::PlaneCollisionQuery*>(nullptr),
      supportPlane);
  return contactArraysFromSupportContacts(supportContacts, includeFloorPositions);
}

std::vector<mm::Vector3f> supportContactPositions(
    const mm::Character& character,
    const SkeletonStateArray& skeletonState,
    const float contactHeight,
    const mm::SupportPlane& supportPlane) {
  validateContactHeight(contactHeight);
  const mm::JointStateList jointStates =
      jointStatesFromSkeletonStateArray(character, skeletonState);
  const std::span<const mm::JointState> jointStateSpan{jointStates.data(), jointStates.size()};
  return mm::computeSupportContactPositions(
      character,
      jointStateSpan,
      contactHeight,
      static_cast<mm::PlaneCollisionQuery*>(nullptr),
      supportPlane);
}

py::tuple supportContacts(
    const mm::Character& character,
    const SkeletonStateArray& skeletonState,
    const float contactHeight,
    const std::optional<mm::Vector3f>& planeNormal,
    const float planeOffset) {
  const mm::SupportPlane supportPlane = supportPlaneFromPython(planeNormal, planeOffset);
  return contactArraysToPython(
      supportContactArrays(
          character,
          skeletonState,
          contactHeight,
          /*includeFloorPositions=*/true,
          supportPlane),
      /*includeFloorPositions=*/true);
}

py::array_t<float> supportPolygonFromWorldPoints(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& supportPoints,
    const std::optional<mm::Vector3f>& planeNormal,
    const float planeOffset,
    const std::optional<mm::Vector3f>& planeUAxis) {
  const std::vector<mm::Vector3f> points = supportPointsFromPython(supportPoints);
  const mm::SupportPlane supportPlane =
      supportPlaneFromPython(planeNormal, planeOffset, planeUAxis);
  return supportPolygonArrayFromHull(
      mm::computeSupportPolygonFromWorldPoints(points, supportPlane), supportPlane);
}

py::array_t<float> supportPolygon(
    const mm::Character& character,
    const SkeletonStateArray& skeletonState,
    const float contactHeight,
    const std::optional<mm::Vector3f>& planeNormal,
    const float planeOffset,
    const std::optional<mm::Vector3f>& planeUAxis) {
  const mm::SupportPlane supportPlane =
      supportPlaneFromPython(planeNormal, planeOffset, planeUAxis);
  return supportPolygonArrayFromHull(
      mm::computeSupportPolygonFromWorldPoints(
          supportContactPositions(character, skeletonState, contactHeight, supportPlane),
          supportPlane),
      supportPlane);
}

} // namespace

void addSupportContactBindings(py::module_& m) {
  m.def(
      "plane_collision_contacts_by_parent",
      &planeCollisionContactsByParent,
      R"(Return deepest plane-collision contacts, grouped by parent joint.

The query uses Momentum's plane-collision contact convention: each collision
primitive contributes when its signed surface distance to the support plane
is less than or equal to ``contact_margin``. If multiple primitives under the
same parent joint contribute, only the deepest contact point for that parent is
returned.

:param character: Character whose collision geometry is queried.
:param skeleton_state: Global skeleton state array with shape ``(n_joints, 8)``.
  Columns 0-2 are translation, 3-5 are quaternion xyz, 6 is quaternion w, and
  7 is uniform scale.
:param contact_margin: Maximum signed surface distance from the support plane,
  in character units, for a primitive to count as contact.
:param plane_normal: Optional support-plane normal, normalized internally.
  Defaults to the Y-up ground plane normal ``[0, 1, 0]``.
:param plane_offset: Support-plane offset ``d`` for points satisfying
  ``dot(plane_normal, point) = d``.
:return: ``(positions, parent_indices, parent_offsets)``. ``positions`` has shape
  ``(n_contacts, 3)`` in world space. ``parent_indices`` has shape
  ``(n_contacts,)``. ``parent_offsets`` has shape ``(n_contacts, 3)`` and stores
  the rotation-local, scale-applied parent offset for each contact point.)",
      py::arg("character"),
      py::arg("skeleton_state"),
      py::arg("contact_margin") = 0.0f,
      py::arg("plane_normal") = std::nullopt,
      py::arg("plane_offset") = 0.0f);

  m.def(
      "support_contacts",
      &supportContacts,
      R"(Return floor-locator and collision contacts used for support polygons.

Floor locators whose names start with ``"Floor"`` contribute when their signed
distance to the support plane is less than or equal to
``contact_height``; this includes below-plane locators and locators up to
``contact_height`` above the plane. Collision primitives use the same signed
surface-distance contact convention as
:func:`pymomentum.geometry.plane_collision_contacts_by_parent`, with
``contact_height`` applied as the collision contact margin. This mirrors
Momentum's retargeter support-polygon contact source.

When collision geometry is present, this Python binding builds a temporary
collision query on each call. Account for that overhead when calling it in
per-frame loops.

:param character: Character whose floor locators and collision geometry are queried.
:param skeleton_state: Global skeleton state array with shape ``(n_joints, 8)``.
  Columns 0-2 are translation, 3-5 are quaternion xyz, 6 is quaternion w, and
  7 is uniform scale.
:param contact_height: Maximum floor-locator signed distance and maximum
  collision surface distance from the support plane, in character units.
:param plane_normal: Optional support-plane normal, normalized internally.
  Defaults to the Y-up ground plane normal ``[0, 1, 0]``.
:param plane_offset: Support-plane offset ``d`` for points satisfying
  ``dot(plane_normal, point) = d``.
:return: ``(positions, parent_indices, parent_offsets, floor_positions)``.
  ``positions`` contains active floor-locator and collision contacts.
  ``parent_indices`` and ``parent_offsets`` describe each active contact's parent
  joint and rotation-local, scale-applied offset. ``floor_positions`` contains
  all valid ``Floor*`` locator world positions, including non-contacting ones.)",
      py::arg("character"),
      py::arg("skeleton_state"),
      py::arg("contact_height") = 2.0f,
      py::arg("plane_normal") = std::nullopt,
      py::arg("plane_offset") = 0.0f);

  m.def(
      "support_polygon_from_world_points",
      &supportPolygonFromWorldPoints,
      R"(Project world-space support points onto a support plane and return their convex hull.

The input points are projected to ``plane_normal``/``plane_offset``, duplicate
projected points are removed, and the returned vertices form the 3D support
polygon in counter-clockwise order in the support-plane basis. Degenerate inputs
return zero, one, or two vertices.

:param support_points: World-space support/contact points with shape
  ``(n_points, 3)``.
:param plane_normal: Optional support-plane normal, normalized internally.
  Defaults to the Y-up ground plane normal ``[0, 1, 0]``.
:param plane_offset: Support-plane offset ``d`` for points satisfying
  ``dot(plane_normal, point) = d``.
:param plane_u_axis: Optional axis hint for the first support-plane coordinate.
  Defaults to ``[1, 0, 0]`` when that axis is not parallel to the plane normal;
  otherwise a perpendicular fallback axis is used.
:return: Support polygon vertices with shape ``(n_vertices, 3)``.)",
      py::arg("support_points"),
      py::arg("plane_normal") = std::nullopt,
      py::arg("plane_offset") = 0.0f,
      py::arg("plane_u_axis") = std::nullopt);

  m.def(
      "support_polygon",
      &supportPolygon,
      R"(Return the support polygon for a character skeleton state.

This is equivalent to calling :func:`pymomentum.geometry.support_contacts` and
then :func:`pymomentum.geometry.support_polygon_from_world_points` on the active
contact positions with the same support plane.

When collision geometry is present, this Python binding builds a temporary
collision query on each call through the support-contact computation. Account
for that overhead when calling it in per-frame loops.

:param character: Character whose floor locators and collision geometry are queried.
:param skeleton_state: Global skeleton state array with shape ``(n_joints, 8)``.
  Columns 0-2 are translation, 3-5 are quaternion xyz, 6 is quaternion w, and
  7 is uniform scale.
:param contact_height: Contact threshold in character units.
:param plane_normal: Optional support-plane normal, normalized internally.
  Defaults to the Y-up ground plane normal ``[0, 1, 0]``.
:param plane_offset: Support-plane offset ``d`` for points satisfying
  ``dot(plane_normal, point) = d``.
:param plane_u_axis: Optional axis hint for the first support-plane coordinate.
  Defaults to ``[1, 0, 0]`` when that axis is not parallel to the plane normal;
  otherwise a perpendicular fallback axis is used.
:return: Support polygon vertices with shape ``(n_vertices, 3)``.)",
      py::arg("character"),
      py::arg("skeleton_state"),
      py::arg("contact_height") = 2.0f,
      py::arg("plane_normal") = std::nullopt,
      py::arg("plane_offset") = 0.0f,
      py::arg("plane_u_axis") = std::nullopt);
}

} // namespace pymomentum
