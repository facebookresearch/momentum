/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/collision_geometry.h>
#include <momentum/character/locator.h>
#include <momentum/character/skeleton.h>

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdSkel/skeleton.h>

PXR_NAMESPACE_USING_DIRECTIVE

namespace momentum {

/// Load a skeleton from a USD stage.
Skeleton loadSkeletonFromUsd(const UsdStageRefPtr& stage);

/// Save a skeleton to a USD stage.
void saveSkeletonToUsd(const Skeleton& skeleton, UsdSkelSkeleton& skelPrim);

/// Save collision geometry as child prims under the skeleton.
void saveCollisionGeometryToUsd(
    const CollisionGeometry& collision,
    const Skeleton& skeleton,
    const UsdStageRefPtr& stage,
    const SdfPath& skelRootPath);

/// Load collision geometry from child prims.
CollisionGeometry loadCollisionGeometryFromUsd(
    const UsdStageRefPtr& stage,
    const Skeleton& skeleton);

/// Save locators as child prims under the skeleton.
void saveLocatorsToUsd(
    const LocatorList& locators,
    const Skeleton& skeleton,
    const UsdStageRefPtr& stage,
    const SdfPath& skelRootPath);

/// Load locators from child prims.
LocatorList loadLocatorsFromUsd(const UsdStageRefPtr& stage, const Skeleton& skeleton);

} // namespace momentum
