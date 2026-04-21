/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>
#include <momentum/io/marker/coordinate_system.h>

#include <span>

namespace momentum {

/// Loads all the subjects from a C3D file into a vector of MarkerSequences.
///
/// The function reads the given C3D file and stores the marker data as a MarkerSequence. The unit
/// of the input file will be read from the file and converted to the momentum unit (i.e., cm). The
/// UpVector parameter 'up' is the up-vector convention of the input marker file.
///
/// @param[in] filename The C3D file to load marker data from.
/// @param[in] up (Optional) The up-vector convention of the input marker file (default:
/// UpVector::Y).
/// @return std::vector<MarkerSequence> A vector containing the marker sequences from the C3D file.
[[nodiscard]] std::vector<MarkerSequence> loadC3d(
    const std::string& filename,
    UpVector up = UpVector::Y,
    std::span<const std::string> validMarkerNames = {});

#ifdef MOMENTUM_WITH_EZC3D_ISTREAM
/// Loads all the subjects from an in-memory C3D byte buffer into a vector of MarkerSequences.
///
/// Same semantics as the filename overload, but reads from a buffer instead of a file.
///
/// @param[in] bytes Buffer holding the C3D file contents.
/// @param[in] up (Optional) The up-vector convention of the input data.
/// @param[in] validMarkerNames (Optional) Marker-name allowlist.
/// @return std::vector<MarkerSequence> A vector containing the marker sequences from the buffer.
[[nodiscard]] std::vector<MarkerSequence> loadC3d(
    std::span<const std::byte> bytes,
    UpVector up = UpVector::Y,
    std::span<const std::string> validMarkerNames = {});
#endif

} // namespace momentum
