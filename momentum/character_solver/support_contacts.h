/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/types.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/math/support_polygon.h>
#include <momentum/math/types.h>

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace momentum {

/// Support/contact point selected for floor-locator and collision support queries.
template <typename T>
struct SupportContactT {
  size_t parentJoint = kInvalidIndex;
  Vector3<T> position = Vector3<T>::Zero();
  /// Parent-local offset in character units; applying the parent transform reconstructs position.
  Vector3<T> parentOffset = Vector3<T>::Zero();
};

using SupportContact = SupportContactT<float>;
using SupportContactd = SupportContactT<double>;

/// Floor-locator and collision support-contact query result for a skeleton state.
template <typename T>
struct SupportContactListT {
  std::vector<SupportContactT<T>> contacts;
  std::vector<Vector3<T>> floorLocatorPositions;
};

using SupportContactList = SupportContactListT<float>;
using SupportContactListd = SupportContactListT<double>;

/// Return true when @p name uses Momentum's floor-locator prefix.
[[nodiscard]] bool isFloorLocatorName(std::string_view name);

/// Return support contacts for an explicit plane-collision query object.
///
/// The query uses @p collisionQuery's configured plane. Use computeSupportContacts when
/// floor-locator contacts and collision contacts should be evaluated against one shared support
/// plane.
template <typename T>
[[nodiscard]] std::vector<SupportContactT<T>> computePlaneCollisionSupportContacts(
    PlaneCollisionQueryT<T>& collisionQuery,
    std::span<const JointStateT<T>> states,
    T contactMargin);

/// Return floor-locator support contacts and all valid floor-locator positions.
///
/// A floor locator is considered an active support contact when its signed distance to
/// @p supportPlane is <= @p contactHeight. Points below the plane are treated as contacts.
template <typename T>
[[nodiscard]] SupportContactListT<T> computeFloorLocatorSupportContacts(
    std::span<const JointStateT<T>> states,
    std::span<const Locator> locators,
    T contactHeight,
    const SupportPlaneT<T>& supportPlane);

/// Return only the world-space positions from floor locators and optional plane-collision contacts.
///
/// Use this when callers need merged support positions but do not need parent indices, parent-local
/// offsets, or the full set of floor-locator positions.
///
/// When @p collisionQuery is null and @p character owns collision geometry, a temporary
/// collision query over @p supportPlane is used for the query. For repeated per-frame queries
/// with collision geometry, pass a persistent collision query to avoid rebuilding its candidate
/// state.
template <typename T>
[[nodiscard]] std::vector<Vector3<T>> computeSupportContactPositions(
    const Character& character,
    std::span<const JointStateT<T>> states,
    T contactHeight,
    PlaneCollisionQueryT<T>* collisionQuery,
    const SupportPlaneT<T>& supportPlane);

/// Return support contacts from floor locators and optional plane-collision contacts.
///
/// The same threshold is used as the signed plane-distance limit for floor locators and as the
/// signed surface-distance margin for plane-collision contacts. Call the floor-locator or
/// plane-collision helpers directly when those tolerances need to differ.
///
/// When @p collisionQuery is non-null, it must use the same oriented plane as @p supportPlane
/// so collision contacts are evaluated against the same plane side as floor locators.
///
/// When @p collisionQuery is null and @p character owns collision geometry, a temporary
/// collision query over @p supportPlane is used for the query. For repeated per-frame queries
/// with collision geometry, pass a persistent collision query to avoid rebuilding its candidate
/// state.
template <typename T>
[[nodiscard]] SupportContactListT<T> computeSupportContacts(
    const Character& character,
    std::span<const JointStateT<T>> states,
    T contactHeight,
    PlaneCollisionQueryT<T>* collisionQuery,
    const SupportPlaneT<T>& supportPlane);

} // namespace momentum
