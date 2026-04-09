/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/marker_tracking/marker_gap_fill.h"

#include "momentum/math/constants.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace momentum {

namespace {

/// Ensure every frame has the same markers in the same canonical order.
///
/// When marker data comes from concatenated sequences (e.g. body ROM + hand
/// ROM), different frames may have different marker sets.  The gap fill code
/// requires positional indexing to be consistent across frames, so this
/// function pads each frame to the union of all marker names.
///
/// Ordering: frame[0]'s markers keep their original order; markers that first
/// appear in later frames are appended in the order they are encountered.
/// Markers absent from a given frame are inserted as occluded with zero confidence.
///
/// This is a no-op when all frames already have the same marker count.
void normalizeMarkerFrames(std::vector<std::vector<Marker>>& markerData) {
  if (markerData.empty()) {
    return;
  }

  // Quick check: if all frames have the same count as frame[0], assume
  // they are already consistent (the common case).
  const size_t firstFrameCount = markerData[0].size();
  bool allSame = true;
  for (const auto& frame : markerData) {
    if (frame.size() != firstFrameCount) {
      allSame = false;
      break;
    }
  }
  if (allSame) {
    return;
  }

  // Build canonical name ordering: frame[0] first, then append new names.
  std::vector<std::string> canonicalNames;
  std::unordered_map<std::string, size_t> nameToIndex;
  canonicalNames.reserve(firstFrameCount + 16);

  for (const auto& frame : markerData) {
    for (const auto& marker : frame) {
      if (nameToIndex.find(marker.name) == nameToIndex.end()) {
        nameToIndex[marker.name] = canonicalNames.size();
        canonicalNames.push_back(marker.name);
      }
    }
  }

  const size_t totalMarkers = canonicalNames.size();

  // Rebuild each frame in canonical order.
  for (auto& frame : markerData) {
    if (frame.size() == totalMarkers) {
      // Check if already in canonical order.
      bool inOrder = true;
      for (size_t i = 0; i < totalMarkers; ++i) {
        if (frame[i].name != canonicalNames[i]) {
          inOrder = false;
          break;
        }
      }
      if (inOrder) {
        continue;
      }
    }

    // Build new frame with occluded placeholders.
    std::vector<Marker> normalized(totalMarkers);
    for (size_t i = 0; i < totalMarkers; ++i) {
      normalized[i].name = canonicalNames[i];
      normalized[i].occluded = true;
      normalized[i].confidence = 0.0f;
    }

    // Fill in markers that exist in this frame.
    for (auto& marker : frame) {
      const size_t idx = nameToIndex.at(marker.name);
      normalized[idx] = std::move(marker);
    }

    frame = std::move(normalized);
  }
}

/// Estimate linear velocity from a window of visible marker frames.
/// Returns velocity in units/frame. If fewer than 2 visible frames are available,
/// returns zero velocity.
///
/// @param markerData The full marker data.
/// @param markerIdx Index of the marker within each frame's marker list.
/// @param anchorFrame The frame at the boundary (first or last visible frame of a gap).
/// @param windowSize Number of visible frames to use for estimation.
/// @param forward If true, look forward from anchorFrame; if false, look backward.
Vector3d estimateVelocity(
    const std::vector<std::vector<Marker>>& markerData,
    size_t markerIdx,
    size_t anchorFrame,
    size_t windowSize,
    bool forward) {
  // Collect visible frames in the requested direction
  std::vector<std::pair<size_t, Vector3d>> samples;
  const auto numFrames = markerData.size();

  if (forward) {
    for (size_t f = anchorFrame; f < numFrames && samples.size() < windowSize; ++f) {
      const auto& m = markerData[f][markerIdx];
      if (!m.occluded) {
        samples.emplace_back(f, m.pos);
      }
    }
  } else {
    // Walk backward; use signed arithmetic carefully
    for (size_t f = anchorFrame + 1; f > 0 && samples.size() < windowSize; --f) {
      const auto& m = markerData[f - 1][markerIdx];
      if (!m.occluded) {
        samples.emplace_back(f - 1, m.pos);
      }
    }
    // Reverse so samples are in chronological order
    std::reverse(samples.begin(), samples.end());
  }

  if (samples.size() < 2) {
    return Vector3d::Zero();
  }

  // Least-squares linear fit: average finite differences weighted equally
  Vector3d totalVel = Vector3d::Zero();
  size_t count = 0;
  for (size_t i = 1; i < samples.size(); ++i) {
    const double dt =
        static_cast<double>(samples[i].first) - static_cast<double>(samples[i - 1].first);
    if (dt > 0) {
      totalVel += (samples[i].second - samples[i - 1].second) / dt;
      ++count;
    }
  }

  if (count == 0) {
    return Vector3d::Zero();
  }
  return totalVel / static_cast<double>(count);
}

/// Cubic Hermite interpolation between two endpoints.
///
/// @param p0 Position at t=0
/// @param v0 Velocity at t=0 (units/frame, scaled by gap length)
/// @param p1 Position at t=1
/// @param v1 Velocity at t=1 (units/frame, scaled by gap length)
/// @param t Parameter in [0, 1]
Vector3d
hermite(const Vector3d& p0, const Vector3d& v0, const Vector3d& p1, const Vector3d& v1, double t) {
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
  const double h10 = t3 - 2.0 * t2 + t;
  const double h01 = -2.0 * t3 + 3.0 * t2;
  const double h11 = t3 - t2;
  return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1;
}

/// Cosine blend-off weight: ramps from 1.0 at step=0 to 0.0 at step=totalSteps.
float cosineBlendWeight(size_t step, size_t totalSteps) {
  if (totalSteps == 0) {
    return 0.0f;
  }
  const double t = static_cast<double>(step) / static_cast<double>(totalSteps);
  // Cosine ramp: 1 at t=0, 0 at t=1
  return static_cast<float>(0.5 * (1.0 + std::cos(t * pi<double>())));
}

/// Represents a contiguous run of occluded frames for a single marker.
struct OcclusionGap {
  size_t startFrame; // first occluded frame
  size_t endFrame; // one past last occluded frame (exclusive)
  bool hasVisibleBefore; // is there a visible frame before startFrame?
  bool hasVisibleAfter; // is there a visible frame at or after endFrame?
};

/// Find all occlusion gaps for a single marker.
std::vector<OcclusionGap> findGaps(
    const std::vector<std::vector<Marker>>& markerData,
    size_t markerIdx) {
  const auto numFrames = markerData.size();
  std::vector<OcclusionGap> gaps;

  size_t f = 0;
  while (f < numFrames) {
    // Skip visible frames
    if (!markerData[f][markerIdx].occluded) {
      ++f;
      continue;
    }

    // Found start of an occluded run
    const size_t gapStart = f;
    while (f < numFrames && markerData[f][markerIdx].occluded) {
      ++f;
    }

    gaps.push_back({
        gapStart,
        f,
        gapStart > 0 && !markerData[gapStart - 1][markerIdx].occluded,
        f < numFrames && !markerData[f][markerIdx].occluded,
    });
  }

  return gaps;
}

/// Fill a temporary gap using cubic Hermite interpolation.
void fillTemporaryGap(
    std::vector<std::vector<Marker>>& markerData,
    size_t markerIdx,
    const OcclusionGap& gap,
    const GapFillConfig& config) {
  const size_t lastVisibleBefore = gap.startFrame - 1;
  const size_t firstVisibleAfter = gap.endFrame;
  const size_t gapLen = gap.endFrame - gap.startFrame;

  const auto& mBefore = markerData[lastVisibleBefore][markerIdx];
  const auto& mAfter = markerData[firstVisibleAfter][markerIdx];

  // Estimate velocity at gap boundaries (in units/frame)
  const Vector3d velBefore = estimateVelocity(
      markerData, markerIdx, lastVisibleBefore, config.velocityWindowFrames, false);
  const Vector3d velAfter =
      estimateVelocity(markerData, markerIdx, firstVisibleAfter, config.velocityWindowFrames, true);

  // Scale velocities by gap length for Hermite (tangent = velocity * interval)
  const auto gapLenD = static_cast<double>(gapLen + 1);
  const Vector3d scaledVelBefore = velBefore * gapLenD;
  const Vector3d scaledVelAfter = velAfter * gapLenD;

  for (size_t f = gap.startFrame; f < gap.endFrame; ++f) {
    const double t = static_cast<double>(f - lastVisibleBefore) / gapLenD;
    auto& m = markerData[f][markerIdx];
    m.pos = hermite(mBefore.pos, scaledVelBefore, mAfter.pos, scaledVelAfter, t);
    m.occluded = false;
    m.confidence = 1.0f;
  }
}

/// Blend off a trailing dropout (marker was visible, then permanently occluded).
void blendOffTrailingDropout(
    std::vector<std::vector<Marker>>& markerData,
    size_t markerIdx,
    const OcclusionGap& gap,
    const GapFillConfig& config) {
  const size_t lastVisibleBefore = gap.startFrame - 1;
  const auto& mBefore = markerData[lastVisibleBefore][markerIdx];

  const Vector3d vel = estimateVelocity(
      markerData, markerIdx, lastVisibleBefore, config.velocityWindowFrames, false);

  const size_t blendFrames = std::min(config.blendOffFrames, gap.endFrame - gap.startFrame);

  for (size_t i = 0; i < blendFrames; ++i) {
    const size_t f = gap.startFrame + i;
    const auto dt = static_cast<double>(f - lastVisibleBefore);
    auto& m = markerData[f][markerIdx];
    m.pos = mBefore.pos + vel * dt;
    m.occluded = false;
    m.confidence = cosineBlendWeight(i + 1, blendFrames + 1);
  }
}

/// Blend off a leading dropout (marker starts occluded, then appears).
void blendOffLeadingDropout(
    std::vector<std::vector<Marker>>& markerData,
    size_t markerIdx,
    const OcclusionGap& gap,
    const GapFillConfig& config) {
  const size_t firstVisibleAfter = gap.endFrame;
  const auto& mAfter = markerData[firstVisibleAfter][markerIdx];

  const Vector3d vel =
      estimateVelocity(markerData, markerIdx, firstVisibleAfter, config.velocityWindowFrames, true);

  const size_t gapLen = gap.endFrame - gap.startFrame;
  const size_t blendFrames = std::min(config.blendOffFrames, gapLen);

  // Fill from end of gap backward
  for (size_t i = 0; i < blendFrames; ++i) {
    const size_t f = gap.endFrame - 1 - i;
    const double dt = static_cast<double>(f) - static_cast<double>(firstVisibleAfter);
    auto& m = markerData[f][markerIdx];
    m.pos = mAfter.pos + vel * dt;
    m.occluded = false;
    m.confidence = cosineBlendWeight(i + 1, blendFrames + 1);
  }
}

/// Compute effective max gap length for interpolation.
/// When the marker barely moved across the gap, we can safely interpolate
/// over longer windows by blending between maxGapFrames and
/// maxGapFramesStationary based on displacement.
size_t computeEffectiveMaxGap(
    const std::vector<std::vector<Marker>>& markerData,
    size_t markerIdx,
    const OcclusionGap& gap,
    const GapFillConfig& config) {
  if (gap.hasVisibleBefore && gap.hasVisibleAfter &&
      config.maxGapFramesStationary > config.maxGapFrames) {
    const double displacement =
        (markerData[gap.endFrame][markerIdx].pos - markerData[gap.startFrame - 1][markerIdx].pos)
            .norm();
    if (config.maxGapDisplacement > 0.0 && displacement < config.maxGapDisplacement) {
      const double t = displacement / config.maxGapDisplacement;
      return static_cast<size_t>(std::round(
          static_cast<double>(config.maxGapFramesStationary) * (1.0 - t) +
          static_cast<double>(config.maxGapFrames) * t));
    }
  }
  return config.maxGapFrames;
}

/// Suppress short visible segments between occluded regions.
/// After gap filling, if a marker is only visible for fewer than
/// minVisibleFrames between two remaining occluded regions, mark those
/// frames as occluded (weight=0) so the solver ignores them.
void suppressShortVisibleSegments(
    std::vector<std::vector<Marker>>& markerData,
    size_t markerIdx,
    size_t minVisibleFrames) {
  if (minVisibleFrames == 0) {
    return;
  }
  const auto numFrames = markerData.size();
  size_t f = 0;
  while (f < numFrames) {
    if (markerData[f][markerIdx].occluded) {
      ++f;
      continue;
    }

    const size_t visStart = f;
    while (f < numFrames && !markerData[f][markerIdx].occluded) {
      ++f;
    }
    const size_t visLen = f - visStart;

    // Only suppress if bounded by occluded frames on both sides
    // (not at sequence start/end).
    if (visLen < minVisibleFrames && visStart > 0 && f < numFrames) {
      for (size_t i = visStart; i < f; ++i) {
        markerData[i][markerIdx].occluded = true;
        markerData[i][markerIdx].confidence = 0.0f;
      }
    }
  }
}

} // namespace

void preprocessMarkerGaps(
    std::vector<std::vector<Marker>>& markerData,
    const GapFillConfig& config) {
  if (markerData.empty() || !config.enabled) {
    return;
  }

  // Normalize so all frames have the same markers in the same order.
  // This handles concatenated sequences with different marker sets.
  normalizeMarkerFrames(markerData);

  const size_t numMarkers = markerData[0].size();
  if (numMarkers == 0) {
    return;
  }

  for (size_t mi = 0; mi < numMarkers; ++mi) {
    const auto gaps = findGaps(markerData, mi);

    for (const auto& gap : gaps) {
      const size_t gapLen = gap.endFrame - gap.startFrame;
      const size_t effectiveMaxGap = computeEffectiveMaxGap(markerData, mi, gap, config);

      if (gap.hasVisibleBefore && gap.hasVisibleAfter && gapLen <= effectiveMaxGap) {
        // Temporary gap: interpolate with cubic Hermite
        fillTemporaryGap(markerData, mi, gap, config);
      } else if (gap.hasVisibleBefore && gap.hasVisibleAfter) {
        // Long gap with visible frames on both sides: blend off from each side.
        // This prevents a hard pop when the marker reappears.
        if (config.blendOffFrames > 0) {
          blendOffTrailingDropout(markerData, mi, gap, config);
          blendOffLeadingDropout(markerData, mi, gap, config);
        }
      } else if (gap.hasVisibleBefore && config.blendOffFrames > 0) {
        // Trailing dropout: extrapolate forward and blend off
        blendOffTrailingDropout(markerData, mi, gap, config);
      } else if (gap.hasVisibleAfter && config.blendOffFrames > 0) {
        // Leading dropout: extrapolate backward and blend off
        blendOffLeadingDropout(markerData, mi, gap, config);
      }
      // else: both-sided permanent dropout or disabled — leave as-is
    }

    // Second pass: suppress short visible segments that remain after
    // gap filling. These are brief appearances that are unreliable.
    suppressShortVisibleSegments(markerData, mi, config.minVisibleFrames);
  }
}

} // namespace momentum
