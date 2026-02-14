/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <rerun.hpp>

#include <string>

// Rerun SDK version compatibility layer
// This header provides compatibility between rerun 0.23.x (legacy) and 0.29.x+ (latest) APIs.
//
// Minimum supported versions:
// - rerun-latest: >=0.29.0
// - rerun-legacy: >=0.23.3,<0.24
//
// Key API changes between versions:
// - 0.24+: Scalar -> Scalars (takes batch of values)
// - 0.24+: SeriesLine -> SeriesLines
// - 0.24+: SeriesPoint -> SeriesPoints
// - 0.26+: set_time_seconds deprecated -> use set_time_duration_secs
// - 0.28+: Transform3D::with_axis_length removed (use Axes3D archetype instead)
// - 0.29+: No breaking API changes affecting this codebase
//
// The RERUN_VERSION_GE macro is available since rerun 0.18.

namespace momentum {

// Helper function to log a single scalar value
// In rerun 0.24+, Scalar was renamed to Scalars and takes a batch of values
// This wrapper provides a consistent API across versions
template <typename T>
inline auto makeScalar([[maybe_unused]] T value) {
#if defined(RERUN_VERSION_GE) && RERUN_VERSION_GE(0, 24, 0)
  // Rerun 0.24+: Use Scalars with a single-element batch
  return rerun::Scalars({static_cast<double>(value)});
#else
  // Rerun 0.23.x: Use Scalar with a single value
  return rerun::Scalar(value);
#endif
}

// Helper function to create a SeriesLine configuration
// In rerun 0.24+, SeriesLine was renamed to SeriesLines
inline auto makeSeriesLine() {
#if defined(RERUN_VERSION_GE) && RERUN_VERSION_GE(0, 24, 0)
  return rerun::SeriesLines();
#else
  return rerun::SeriesLine();
#endif
}

// Helper function to create a SeriesLine with a name
inline auto makeSeriesLineWithName(const std::string& name) {
#if defined(RERUN_VERSION_GE) && RERUN_VERSION_GE(0, 24, 0)
  return rerun::SeriesLines().with_names({name});
#else
  return rerun::SeriesLine().with_name(name);
#endif
}

// Helper function to set time in seconds on a recording stream
// In rerun 0.26+, set_time_seconds was deprecated in favor of set_time_duration_secs
inline void
setTimeSeconds(const rerun::RecordingStream& rec, const std::string& timelineName, double seconds) {
#if defined(RERUN_VERSION_GE) && RERUN_VERSION_GE(0, 26, 0)
  rec.set_time_duration_secs(timelineName, seconds);
#else
  rec.set_time_seconds(timelineName, seconds);
#endif
}

// Helper function to create a Transform3D with axis visualization
// In rerun 0.28+, with_axis_length was removed from Transform3D
// Instead, Axes3D archetype should be used for axis visualization
// This wrapper returns the transform without axis_length for 0.28+
// Callers should log a separate Axes3D archetype if axis visualization is needed
template <typename... Args>
inline auto makeTransform3D(Args&&... args) {
  return rerun::Transform3D(std::forward<Args>(args)...);
}

// Log coordinate axes at a given entity path
// In rerun 0.28+, Transform3D::with_axis_length was removed
// The axis visualization is omitted in newer versions
inline void logAxes3D(
    [[maybe_unused]] const rerun::RecordingStream& rec,
    [[maybe_unused]] const std::string& path,
    [[maybe_unused]] float axisLength) {
#if !defined(RERUN_VERSION_GE) || !RERUN_VERSION_GE(0, 24, 0)
  (void)rec;
  (void)path;
  (void)axisLength;
  // In 0.23.x, axis_length is set directly on Transform3D (handled by caller)
#endif
  // In 0.24+, axis visualization on transforms is not supported
  // To visualize axes, users should use Arrows3D archetype manually
}

} // namespace momentum
