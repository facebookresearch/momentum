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
void CollisionGeometryStateT<T>::resize(const size_t numElements) {
  origin.resize(numElements);
  direction.resize(numElements);
  radius.resize(numElements);
  delta.resize(numElements);
  type.resize(numElements);
  orientation.resize(numElements);
  ellipsoidRadii.resize(numElements);
  boxHalfExtents.resize(numElements);
}

template <typename T>
void CollisionGeometryStateT<T>::updatePrimitive(
    const SkeletonStateT<T>& skeletonState,
    const CollisionGeometry& collisionGeometry,
    const size_t primitiveIndex) {
  const auto& primitive = collisionGeometry[primitiveIndex];
  const TransformT<T> parentTransform = (primitive.parent == kInvalidIndex)
      ? TransformT<T>()
      : skeletonState.jointState[primitive.parent].transform;
  const TransformT<T> transform = parentTransform * primitive.transformation.cast<T>();
  type[primitiveIndex] = primitive.type;
  origin[primitiveIndex] = transform.translation;
  orientation[primitiveIndex] = transform.rotation;

  if (primitive.type == CollisionPrimitiveType::TaperedCapsule) {
    direction[primitiveIndex].noalias() = transform.getAxisX() * transform.scale * primitive.length;
    radius[primitiveIndex].noalias() = primitive.radius.cast<T>() * parentTransform.scale;
    delta[primitiveIndex] = radius[primitiveIndex][1] - radius[primitiveIndex][0];
    ellipsoidRadii[primitiveIndex].setZero();
    boxHalfExtents[primitiveIndex].setZero();
    return;
  }

  if (primitive.type == CollisionPrimitiveType::Ellipsoid) {
    direction[primitiveIndex].setZero();
    radius[primitiveIndex].setZero();
    delta[primitiveIndex] = T(0);
    ellipsoidRadii[primitiveIndex].noalias() =
        primitive.ellipsoidRadii.cast<T>() * parentTransform.scale;
    boxHalfExtents[primitiveIndex].setZero();
    return;
  }

  if (primitive.type == CollisionPrimitiveType::Box) {
    direction[primitiveIndex].setZero();
    radius[primitiveIndex].setZero();
    delta[primitiveIndex] = T(0);
    ellipsoidRadii[primitiveIndex].setZero();
    boxHalfExtents[primitiveIndex].noalias() =
        primitive.boxHalfExtents.cast<T>() * parentTransform.scale;
    return;
  }
}

template <typename T>
void CollisionGeometryStateT<T>::update(
    const SkeletonStateT<T>& skeletonState,
    const CollisionGeometry& collisionGeometry) {
  const size_t numElements = collisionGeometry.size();
  resize(numElements);

  // TODO: This per-primitive loop is independent across iterations and could be parallelized.
  for (size_t i = 0; i < numElements; ++i) {
    updatePrimitive(skeletonState, collisionGeometry, i);
  }
}

template struct CollisionGeometryStateT<float>;
template struct CollisionGeometryStateT<double>;

} // namespace momentum
