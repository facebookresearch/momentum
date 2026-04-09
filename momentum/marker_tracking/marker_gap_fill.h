/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>

#include <vector>

namespace momentum {

/// Configuration for marker gap filling and blend-off.
struct GapFillConfig {
  /// Master switch.  When false, preprocessMarkerGaps() is a no-op.
  /// Default is false (off); callers that want gap filling must opt in.
  bool enabled = false;

  /// Maximum gap length (in frames) to interpolate using cubic Hermite.
  /// Gaps longer than this are treated as permanent dropouts.
  /// Set to 0 to disable gap interpolation.
  /// Default 30 (~500ms at 60Hz).
  size_t maxGapFrames = 30;

  /// Extended maximum gap length (in frames) for near-stationary markers.
  /// When a marker's displacement across a gap is below
  /// maxGapDisplacement, gaps up to this length are interpolated.
  /// Linearly blends to maxGapFrames as displacement approaches
  /// maxGapDisplacement. Set equal to maxGapFrames to disable.
  /// Default 90 (~1.5s at 60Hz).
  size_t maxGapFramesStationary = 90;

  /// Displacement threshold (cm) above which the standard maxGapFrames
  /// applies. Below this, the effective max gap is linearly interpolated
  /// toward maxGapFramesStationary. Default 5.0 cm.
  double maxGapDisplacement = 5.0;

  /// Minimum number of visible frames for a segment to be considered reliable.
  /// If a marker is visible for fewer than this many frames between two
  /// occluded regions, those visible frames are marked occluded, merging
  /// the surrounding gaps. Set to 0 to disable. Default 60 (~1s at 60Hz).
  size_t minVisibleFrames = 60;

  /// Number of frames over which to blend off permanently dropped markers.
  /// Uses a cosine ramp from confidence 1.0 to 0.0. Set to 0 to disable blend-off.
  size_t blendOffFrames = 10;

  /// Number of visible frames to use for velocity estimation when
  /// extrapolating permanent dropouts.
  size_t velocityWindowFrames = 5;
};

/// Pre-process marker data to fill temporary gaps and blend off permanent dropouts.
///
/// For temporary gaps (marker disappears then reappears within maxGapFrames),
/// the gap is filled using cubic Hermite interpolation with position and velocity
/// at the gap boundaries. Filled markers get occluded=false and confidence=1.0.
///
/// For permanent dropouts (marker disappears and never returns, or appears late),
/// the marker position is extrapolated using linear velocity estimated from the
/// nearest visible frames, and the confidence ramps from 1.0 to 0.0 over blendOffFrames
/// using a cosine curve. Extrapolated markers get occluded=false.
///
/// For long gaps with visible frames on both sides (exceeding maxGapFrames),
/// the marker is blended off from each side independently, preventing a hard pop
/// when the marker reappears.
///
/// @param[in,out] markerData Marker data to process in-place. Each frame must have
///   the same markers in the same order.
/// @param[in] config Gap fill configuration.
void preprocessMarkerGaps(
    std::vector<std::vector<Marker>>& markerData,
    const GapFillConfig& config = {});

} // namespace momentum
