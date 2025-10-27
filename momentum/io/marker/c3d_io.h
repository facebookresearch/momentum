/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>
#include <momentum/io/marker/coordinate_system.h>

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

} // namespace momentum
