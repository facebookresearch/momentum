/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/types.h>
#include <momentum/math/types.h>

#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdSkel/skeleton.h>

#include <span>
#include <string>
#include <tuple>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace momentum {

/// Save skeleton states as a UsdSkelAnimation on the stage.
///
/// Creates a UsdSkelAnimation prim with time-sampled translations,
/// rotations, and scales per joint per frame.
///
/// @param[in] stage The USD stage to write to.
/// @param[in] skelPrim The skeleton prim to bind the animation to.
/// @param[in] skeleton The skeleton definition (for joint names).
/// @param[in] skeletonStates The skeleton states to save (one per frame).
/// @param[in] fps Frames per second for the animation.
void saveSkeletonStatesToUsd(
    const UsdStageRefPtr& stage,
    UsdSkelSkeleton& skelPrim,
    const Skeleton& skeleton,
    std::span<const SkeletonState> skeletonStates,
    float fps);

/// Load skeleton states from a USD stage.
///
/// Reads UsdSkelAnimation time-sampled data and reconstructs
/// SkeletonState objects for each frame.
///
/// @param[in] stage The USD stage to read from.
/// @param[in] skeleton The skeleton definition.
/// @return A tuple of (skeleton states, frame times in seconds).
std::tuple<std::vector<SkeletonState>, std::vector<float>> loadSkeletonStatesFromUsd(
    const UsdStageRefPtr& stage,
    const Skeleton& skeleton);

/// Save model-parameter motion as custom attributes on a USD prim.
void saveMotionToUsd(
    const UsdPrim& prim,
    float fps,
    const MotionParameters& motion,
    const IdentityParameters& offsets);

/// Load model-parameter motion from custom attributes on a USD prim.
std::tuple<MotionParameters, IdentityParameters, float> loadMotionFromUsd(const UsdPrim& prim);

} // namespace momentum
