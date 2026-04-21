/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>
#include <momentum/io/marker/coordinate_system.h>

#include <istream>
#include <span>

namespace momentum {

/// Loads marker data from a TRC file into a MarkerSequence.
///
/// The function reads the given TRC file and stores the marker data as a MarkerSequence. The unit
/// of the input file will be read from the file and converted to the momentum unit (i.e., cm). The
/// UpVector parameter 'up' is the up-vector convention of the input marker file.
///
/// @param[in] filename The TRC file to load marker data from.
/// @param[in] up (Optional) The up-vector convention of the input marker file (default:
/// UpVector::Y).
/// @return MarkerSequence A MarkerSequence containing the marker data from the TRC file.
[[nodiscard]] MarkerSequence loadTrc(const std::string& filename, UpVector up = UpVector::Y);

/// Loads marker data from a TRC input stream into a MarkerSequence.
///
/// Same as the filename overload but reads from any std::istream (e.g., a file, a memory-backed
/// stream, etc.). The caller owns the stream and is responsible for opening it in text mode.
///
/// @param[in,out] stream Open input stream positioned at the start of the TRC content.
/// @param[in] up (Optional) The up-vector convention of the input marker file.
/// @return MarkerSequence A MarkerSequence containing the marker data from the stream.
[[nodiscard]] MarkerSequence loadTrc(std::istream& stream, UpVector up = UpVector::Y);

/// Loads marker data from an in-memory TRC byte buffer into a MarkerSequence.
///
/// Wraps the buffer in an ispanstream and forwards to the istream overload, so no copy is made.
///
/// @param[in] bytes Buffer holding the TRC file contents.
/// @param[in] up (Optional) The up-vector convention of the input marker file.
/// @return MarkerSequence A MarkerSequence containing the marker data from the buffer.
[[nodiscard]] MarkerSequence loadTrc(std::span<const std::byte> bytes, UpVector up = UpVector::Y);

} // namespace momentum
