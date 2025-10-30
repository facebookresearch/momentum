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

} // namespace momentum
