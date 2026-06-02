/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/collision_geometry_state.h"

#include "momentum/character/skeleton_state.h"
#include "momentum/character/types.h"

namespace momentum {

template <typename T>
void CollisionGeometryStateT<T>::update(
    const SkeletonStateT<T>& skeletonState,
    const CollisionGeometry& collisionGeometry) {
  const size_t numElements = collisionGeometry.size();
  origin.resize(numElements);
  direction.resize(numElements);
  radius.resize(numElements);
  delta.resize(numElements);
  type.resize(numElements);
  orientation.resize(numElements);
  ellipsoidRadii.resize(numElements);
  boxHalfExtents.resize(numElements);

  // TODO: This per-primitive loop is independent across iterations and could be parallelized.
  for (size_t i = 0; i < numElements; ++i) {
    const auto& primitive = collisionGeometry[i];
    const TransformT<T> parentTransform = (primitive.parent == kInvalidIndex)
        ? TransformT<T>()
        : skeletonState.jointState[primitive.parent].transform;
    const TransformT<T> transform = parentTransform * primitive.transformation.cast<T>();
    type[i] = primitive.type;
    origin[i] = transform.translation;
    orientation[i] = transform.rotation;

    if (primitive.type == CollisionPrimitiveType::TaperedCapsule) {
      direction[i].noalias() = transform.getAxisX() * transform.scale * primitive.length;
      radius[i].noalias() = primitive.radius.cast<T>() * parentTransform.scale;
      delta[i] = radius[i][1] - radius[i][0];
      ellipsoidRadii[i].setZero();
      boxHalfExtents[i].setZero();
      continue;
    }

    if (primitive.type == CollisionPrimitiveType::Ellipsoid) {
      direction[i].setZero();
      radius[i].setZero();
      delta[i] = T(0);
      ellipsoidRadii[i].noalias() = primitive.ellipsoidRadii.cast<T>() * parentTransform.scale;
      boxHalfExtents[i].setZero();
      continue;
    }

    if (primitive.type == CollisionPrimitiveType::Box) {
      direction[i].setZero();
      radius[i].setZero();
      delta[i] = T(0);
      ellipsoidRadii[i].setZero();
      boxHalfExtents[i].noalias() = primitive.boxHalfExtents.cast<T>() * parentTransform.scale;
      continue;
    }
  }
}

template struct CollisionGeometryStateT<float>;
template struct CollisionGeometryStateT<double>;

} // namespace momentum
