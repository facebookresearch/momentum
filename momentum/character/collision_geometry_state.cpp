/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/collision_geometry_state.h>

#include <momentum/character/skeleton_state.h>

namespace momentum {

template <typename T>
void CollisionGeometryStateT<T>::update(
    const SkeletonStateT<T>& skeletonState,
    const CollisionGeometry& collisionGeometry) {
  // resize all elements
  const size_t numElements = collisionGeometry.size();
  origin.resize(numElements);
  direction.resize(numElements);
  radius.resize(numElements);
  delta.resize(numElements);

  // calculate data from the geometry (can be parallelized)
  for (size_t i = 0; i < numElements; ++i) {
    const auto& tc = collisionGeometry[i];
    const auto& js = skeletonState.jointState[tc.parent];
    const Affine3<T> transform = js.transformation * tc.transformation.cast<T>();
    origin[i] = transform.translation();
    direction[i].noalias() = transform.linear().col(0) * tc.length;
    radius[i].noalias() = tc.radius.cast<T>() * js.scale();
    delta[i] = radius[i][1] - radius[i][0];
  }
}

template struct CollisionGeometryStateT<float>;
template struct CollisionGeometryStateT<double>;

} // namespace momentum
