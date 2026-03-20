/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/skeleton.h>

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdSkel/skeleton.h>

PXR_NAMESPACE_USING_DIRECTIVE

namespace momentum {

/// Load a skeleton from a USD stage.
///
/// Reads joint names, parent topology, and rest/bind transforms from the first
/// UsdSkelSkeleton prim found on the stage.
///
/// @param[in] stage The USD stage to read from.
/// @return The loaded Skeleton.
Skeleton loadSkeletonFromUsd(const UsdStageRefPtr& stage);

/// Save a skeleton to a USD stage.
///
/// Writes joint names, parent topology, local rest transforms, and world-space
/// bind transforms to the given UsdSkelSkeleton prim.
///
/// @param[in] skeleton The Skeleton to save.
/// @param[in] skelPrim The UsdSkelSkeleton prim to write to.
void saveSkeletonToUsd(const Skeleton& skeleton, UsdSkelSkeleton& skelPrim);

} // namespace momentum
