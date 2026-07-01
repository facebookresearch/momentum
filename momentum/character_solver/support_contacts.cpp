/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/support_contacts.h"

#include "momentum/character/character.h"
#include "momentum/character/joint_state.h"
#include "momentum/character/locator.h"
#include "momentum/character_solver/plane_collision_query.h"
#include "momentum/common/exception.h"
#include "momentum/math/utility.h"

#include <algorithm>
#include <cmath>

namespace momentum {

bool isFloorLocatorName(const std::string_view name) {
  return name.starts_with("Floor");
}

namespace {

template <typename T>
Vector3<T> parentOffsetFromWorldPoint(
    const JointStateT<T>& parentState,
    const Vector3<T>& worldPoint) {
  const T parentScale = parentState.transform.scale;
  const T minParentScale = Eps<T>(1e-8f, 1e-17);
  MT_THROW_IF(
      !std::isfinite(parentScale) || std::abs(parentScale) <= minParentScale,
      "Support contact parent scale magnitude must be finite and greater than {}, got {}",
      minParentScale,
      parentScale);
  return (parentState.transform.rotation.conjugate() *
          (worldPoint - parentState.transform.translation)) /
      parentScale;
}

template <typename T>
void validateMatchingCollisionPlane(
    const PlaneCollisionQueryT<T>& collisionQuery,
    const SupportPlaneT<T>& supportPlane) {
  const Vector3<T>& planeNormal = collisionQuery.getPlaneNormal();
  const T planeOffset = collisionQuery.getPlaneOffset();
  const T normalTolerance = Eps<T>(1e-5f, 1e-12);
  const T offsetTolerance = normalTolerance *
      std::max(T(1), std::max(std::abs(planeOffset), std::abs(supportPlane.offset)));
  const T normalError = (planeNormal - supportPlane.normal).cwiseAbs().maxCoeff();
  const T offsetError = std::abs(planeOffset - supportPlane.offset);
  MT_THROW_IF(
      !planeNormal.allFinite() || !std::isfinite(planeOffset) || normalError > normalTolerance ||
          offsetError > offsetTolerance,
      "Support contact collision function must use the same oriented support plane: "
      "collision normal ({}, {}, {}), offset {}; support normal ({}, {}, {}), offset {}",
      planeNormal.x(),
      planeNormal.y(),
      planeNormal.z(),
      planeOffset,
      supportPlane.normal.x(),
      supportPlane.normal.y(),
      supportPlane.normal.z(),
      supportPlane.offset);
}

template <typename T, typename HandleFloorLocator>
void visitFloorLocatorPositions(
    const std::span<const JointStateT<T>> states,
    const std::span<const Locator> locators,
    const SupportPlaneT<T>& supportPlane,
    HandleFloorLocator handleFloorLocator) {
  for (const Locator& locator : locators) {
    if (!isFloorLocatorName(locator.name)) {
      continue;
    }
    MT_THROW_IF(
        locator.parent >= states.size(),
        "Floor locator '{}' parent {} is outside skeleton state with {} joints",
        locator.name,
        locator.parent,
        states.size());

    const JointStateT<T>& parentState = states[locator.parent];
    const Vector3<T> position = parentState.transform * locator.offset.template cast<T>();
    MT_THROW_IF(
        !position.allFinite(),
        "Floor locator '{}' position must contain only finite values",
        locator.name);
    const T signedDistance = supportPlane.signedDistance(position);
    MT_THROW_IF(
        !std::isfinite(signedDistance),
        "Floor locator '{}' support-plane distance must be finite",
        locator.name);
    handleFloorLocator(locator, parentState, position, signedDistance);
  }
}

template <typename T, typename HandleCollisionQuery>
void withSupportCollisionQuery(
    const Character& character,
    PlaneCollisionQueryT<T>* collisionQuery,
    const SupportPlaneT<T>& supportPlane,
    HandleCollisionQuery handleCollisionQuery) {
  if (collisionQuery != nullptr) {
    validateMatchingCollisionPlane(*collisionQuery, supportPlane);
    handleCollisionQuery(*collisionQuery);
    return;
  }

  if (!character.collision || character.collision->empty()) {
    return;
  }

  PlaneCollisionQueryT<T> defaultCollisionQuery(
      *character.collision, supportPlane.normal, supportPlane.offset);
  handleCollisionQuery(defaultCollisionQuery);
}

} // namespace

template <typename T>
std::vector<SupportContactT<T>> computePlaneCollisionSupportContacts(
    PlaneCollisionQueryT<T>& collisionQuery,
    const std::span<const JointStateT<T>> states,
    const T contactMargin) {
  const std::vector<PlaneCollisionContactPointT<T>> contacts =
      collisionQuery.getContactPointsByParentWithDetails(states, contactMargin);
  std::vector<SupportContactT<T>> result;
  result.reserve(contacts.size());
  for (const PlaneCollisionContactPointT<T>& contact : contacts) {
    MT_THROW_IF(
        contact.parentJoint >= states.size(),
        "Plane collision parent {} is outside skeleton state with {} joints",
        contact.parentJoint,
        states.size());
    result.push_back(
        {contact.parentJoint,
         contact.position,
         parentOffsetFromWorldPoint(states[contact.parentJoint], contact.position)});
  }
  return result;
}

template <typename T>
SupportContactListT<T> computeFloorLocatorSupportContacts(
    const std::span<const JointStateT<T>> states,
    const std::span<const Locator> locators,
    const T contactHeight,
    const SupportPlaneT<T>& supportPlane) {
  MT_THROW_IF(
      !std::isfinite(contactHeight) || contactHeight < T(0),
      "Floor locator contact height must be finite and non-negative, got {}",
      contactHeight);

  SupportContactListT<T> result;
  result.contacts.reserve(locators.size());
  result.floorLocatorPositions.reserve(locators.size());
  visitFloorLocatorPositions<T>(
      states,
      locators,
      supportPlane,
      [&](const Locator& locator,
          const JointStateT<T>& parentState,
          const Vector3<T>& position,
          const T signedDistance) {
        result.floorLocatorPositions.push_back(position);
        if (signedDistance > contactHeight) {
          return;
        }
        result.contacts.push_back(
            {locator.parent, position, parentOffsetFromWorldPoint(parentState, position)});
      });
  return result;
}

template <typename T>
std::vector<Vector3<T>> computeSupportContactPositions(
    const Character& character,
    const std::span<const JointStateT<T>> states,
    const T contactHeight,
    PlaneCollisionQueryT<T>* collisionQuery,
    const SupportPlaneT<T>& supportPlane) {
  MT_THROW_IF(
      !std::isfinite(contactHeight) || contactHeight < T(0),
      "Support contact height must be finite and non-negative, got {}",
      contactHeight);

  std::vector<Vector3<T>> result;
  result.reserve(character.locators.size());
  visitFloorLocatorPositions<T>(
      states,
      character.locators,
      supportPlane,
      [&](const Locator& /*locator*/,
          const JointStateT<T>& /*parentState*/,
          const Vector3<T>& position,
          const T signedDistance) {
        if (signedDistance <= contactHeight) {
          result.push_back(position);
        }
      });

  withSupportCollisionQuery<T>(
      character, collisionQuery, supportPlane, [&](PlaneCollisionQueryT<T>& query) {
        const std::vector<Vector3<T>> collisionPositions =
            query.getContactPointsByParent(states, contactHeight);
        result.insert(result.end(), collisionPositions.begin(), collisionPositions.end());
      });
  return result;
}

template <typename T>
SupportContactListT<T> computeSupportContacts(
    const Character& character,
    const std::span<const JointStateT<T>> states,
    const T contactHeight,
    PlaneCollisionQueryT<T>* collisionQuery,
    const SupportPlaneT<T>& supportPlane) {
  SupportContactListT<T> result =
      computeFloorLocatorSupportContacts(states, character.locators, contactHeight, supportPlane);

  // TODO: Floor locators and collision primitive support points are not identical contact
  // signals. Compute a proper contact set instead of concatenating these proxies.
  withSupportCollisionQuery<T>(
      character, collisionQuery, supportPlane, [&](PlaneCollisionQueryT<T>& query) {
        const std::vector<SupportContactT<T>> collisionContacts =
            computePlaneCollisionSupportContacts(query, states, contactHeight);
        result.contacts.insert(
            result.contacts.end(), collisionContacts.begin(), collisionContacts.end());
      });
  return result;
}

template std::vector<SupportContactT<float>> computePlaneCollisionSupportContacts<float>(
    PlaneCollisionQueryT<float>& collisionQuery,
    std::span<const JointStateT<float>> states,
    float contactMargin);
template std::vector<SupportContactT<double>> computePlaneCollisionSupportContacts<double>(
    PlaneCollisionQueryT<double>& collisionQuery,
    std::span<const JointStateT<double>> states,
    double contactMargin);

template SupportContactListT<float> computeFloorLocatorSupportContacts<float>(
    std::span<const JointStateT<float>> states,
    std::span<const Locator> locators,
    float contactHeight,
    const SupportPlaneT<float>& supportPlane);
template SupportContactListT<double> computeFloorLocatorSupportContacts<double>(
    std::span<const JointStateT<double>> states,
    std::span<const Locator> locators,
    double contactHeight,
    const SupportPlaneT<double>& supportPlane);
template std::vector<Vector3<float>> computeSupportContactPositions<float>(
    const Character& character,
    std::span<const JointStateT<float>> states,
    float contactHeight,
    PlaneCollisionQueryT<float>* collisionQuery,
    const SupportPlaneT<float>& supportPlane);
template std::vector<Vector3<double>> computeSupportContactPositions<double>(
    const Character& character,
    std::span<const JointStateT<double>> states,
    double contactHeight,
    PlaneCollisionQueryT<double>* collisionQuery,
    const SupportPlaneT<double>& supportPlane);
template SupportContactListT<float> computeSupportContacts<float>(
    const Character& character,
    std::span<const JointStateT<float>> states,
    float contactHeight,
    PlaneCollisionQueryT<float>* collisionQuery,
    const SupportPlaneT<float>& supportPlane);
template SupportContactListT<double> computeSupportContacts<double>(
    const Character& character,
    std::span<const JointStateT<double>> states,
    double contactHeight,
    PlaneCollisionQueryT<double>* collisionQuery,
    const SupportPlaneT<double>& supportPlane);

} // namespace momentum
