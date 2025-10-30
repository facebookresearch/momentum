/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gsl/gsl>

#include <istream>
#include <streambuf>
#include <string>

namespace momentum {

/// Stream buffer based on a span.
/// Used to avoid copying the span when creating a std::istream:
///   spanstreambuf sbuf(str);
///   std::istream inputStream(&sbuf);
struct spanstreambuf : std::streambuf {
  /// Construct a spanstreambuf with an optional buffer.
  ///
  /// @param buffer A std::span of const std::byte (default: empty span).
  spanstreambuf(std::span<const std::byte> buffer = {});

  /// Destructor.
  virtual ~spanstreambuf() override;

 protected:
  /// Write a sequence of characters to the output buffer.
  ///
  /// @param s A pointer to the character sequence to be written.
  /// @param n The number of characters to write.
  /// @return The number of characters successfully written.
  std::streamsize xsputn(const char_type* s, std::streamsize n) override;
};

/// istream that takes a span of const bytes as input.
struct ispanstream : std::istream {
  /// Construct an ispanstream with a buffer.
  ///
  /// @param buffer A std::span of const std::byte.
  explicit ispanstream(std::span<const std::byte> buffer);

  /// Destructor.
  virtual ~ispanstream();

 private:
  /// Internal spanstreambuf instance.
  spanstreambuf sbuf_;
};

/// Reads a line from the input stream, handling line endings for different platforms.
///
/// This function reads a line from the given input stream and stores it in the 'line' parameter.
/// It properly handles line endings for different platforms (e.g., LF for Unix, CRLF for Windows).
///
/// @param[in,out] is Reference to the input stream from which the line should be read.
/// @param[in,out] line Reference to the string variable where the line will be stored.
/// @return Reference to the input stream after reading the line.
std::istream& GetLineCrossPlatform(std::istream& is, std::string& line);

} // namespace momentum
