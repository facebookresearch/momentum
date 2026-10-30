/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace momentum {

/// Names of the custom boolean properties that mark the nodes of a marker sequence in an FBX file:
/// the node that holds all marker nodes, and each marker node. The FBX writer sets them, and the
/// loader finds the markers by them.
inline constexpr const char* kMomentumMarkersRootProperty = "Momentum_Markers_Root";
inline constexpr const char* kMomentumMarkerProperty = "Momentum_Marker";

} // namespace momentum
