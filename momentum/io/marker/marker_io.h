/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>
#include <momentum/io/marker/coordinate_system.h>

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace momentum {

/// Loads all actor sequences from a marker file (c3d, trc, glb) and converts the positions
/// to the target coordinate system. Default target coordinate system is: (Y-up, right-handed,
/// centimeter units).
///
/// @param[in] filename The marker file to load data from.
/// @param[in] up (Optional) The up-vector convention of the input marker file (default:
/// UpVector::Y).
/// @return std::vector<MarkerSequence> A vector of MarkerSequences containing the marker data from
/// the file.
[[nodiscard]] std::vector<MarkerSequence> loadMarkers(
    const std::string& filename,
    UpVector up = UpVector::Y,
    std::span<const std::string> validMarkerNames = {});

/// Loads the main subject's marker data from a marker file (c3d, trc, glb) and converts the
/// positions to the target coordinate system. Default target coordinate system is: (Y-up,
/// right-handed, centimeter units). The main subject is determined as the one with the most number
/// of visible markers in the first frame.
///
/// @param[in] filename The marker file to load data from.
/// @param[in] up (Optional) The up-vector convention of the input marker file (default:
/// UpVector::Y).
/// @return std::optional<MarkerSequence> A MarkerSequence containing the main subject's marker
/// data, or an empty optional if no main subject is found.
[[nodiscard]] std::optional<MarkerSequence> loadMarkersForMainSubject(
    const std::string& filename,
    UpVector up = UpVector::Y,
    std::span<const std::string> validMarkerNames = {});

/// Finds the "main subject" from a vector of MarkerSequences. The main subject is currently defined
/// as a named actor with the maximum number of visible markers in the sequence.
///
/// @param[in] markerSequences A vector of MarkerSequences to search for the main subject.
/// @return int The main subject's index in the input vector. If no main subject found, -1 is
/// returned.
/// @note This function is exposed mainly for unit tests.
[[nodiscard]] int findMainSubjectIndex(std::span<const MarkerSequence> markerSequences);

/// Loads all actor sequences from an in-memory marker file buffer (c3d, trc, glb, fbx).
///
/// Same semantics as the filename overload, but takes a byte buffer plus an explicit format
/// hint (since bytes don't carry an extension). All four formats use a true-bytes path with no
/// temporary file: c3d via the patched ezc3d istream constructor, trc via std::istringstream,
/// glb via fx::gltf's byte-span loader, and fbx via OpenFBX's native buffer API.
///
/// @param[in] bytes Marker file contents.
/// @param[in] format File extension including the leading dot (".c3d", ".trc", ".glb", ".fbx").
/// @param[in] up (Optional) Up-vector convention of the input data.
/// @param[in] validMarkerNames (Optional) Marker-name allowlist (c3d only).
/// @return Vector of MarkerSequences. Empty on unknown format or read error.
[[nodiscard]] std::vector<MarkerSequence> loadMarkers(
    std::span<const std::byte> bytes,
    const std::string& format,
    UpVector up = UpVector::Y,
    std::span<const std::string> validMarkerNames = {});

/// Loads the main subject's marker data from an in-memory marker file buffer.
///
/// Convenience wrapper around the buffer-based loadMarkers + findMainSubjectIndex.
///
/// @param[in] bytes Marker file contents.
/// @param[in] format File extension including the leading dot (".c3d", ".trc", ".glb", ".fbx").
/// @param[in] up (Optional) Up-vector convention of the input data.
/// @param[in] validMarkerNames (Optional) Marker-name allowlist (c3d only).
/// @return The main subject's sequence, or empty optional if no main subject is found.
[[nodiscard]] std::optional<MarkerSequence> loadMarkersForMainSubject(
    std::span<const std::byte> bytes,
    const std::string& format,
    UpVector up = UpVector::Y,
    std::span<const std::string> validMarkerNames = {});

} // namespace momentum
