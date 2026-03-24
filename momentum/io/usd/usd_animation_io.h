/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>
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

/// Save marker sequence as animated Xform prims under a Markers scope.
///
/// Creates Xform prims for each unique marker with time-sampled translations
/// and occlusion attributes.
///
/// @param[in] stage The USD stage to write to.
/// @param[in] skelRootPath The path to the SkelRoot prim under which to create the Markers scope.
/// @param[in] fps Frames per second for the animation.
/// @param[in] markerSequence The marker data to save (one vector of markers per frame).
void saveMarkerSequenceToUsd(
    const UsdStageRefPtr& stage,
    const SdfPath& skelRootPath,
    float fps,
    std::span<const std::vector<Marker>> markerSequence);

/// Load marker sequence from animated Xform prims.
///
/// Reads Xform prims under a Markers scope and reconstructs marker data
/// for each frame from time-sampled translations and occlusion attributes.
///
/// @param[in] stage The USD stage to read from.
/// @return The loaded MarkerSequence containing per-frame marker data and fps.
MarkerSequence loadMarkerSequenceFromUsd(const UsdStageRefPtr& stage);

} // namespace momentum
