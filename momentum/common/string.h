/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace momentum {

/// Returns a copy of the string with leading and trailing characters in @p whiteChars removed.
///
/// Comparison is byte-wise: each byte of @p text is checked against @p whiteChars via
/// `std::strchr`. Multi-byte UTF-8 sequences are not interpreted as code points, so only
/// single-byte (ASCII) whitespace characters can be specified in @p whiteChars. If @p text
/// is empty or consists entirely of @p whiteChars, an empty string is returned.
///
/// @tparam StringType The type of the string to trim (e.g., std::string, std::string_view).
///   For std::string_view inputs the returned view aliases the same underlying buffer as
///   @p text and remains valid only as long as that buffer does.
/// @param text The string to trim.
/// @param whiteChars Null-terminated set of single-byte characters to strip from both ends.
/// @return The trimmed string (or view).
template <typename StringType>
[[nodiscard]] StringType trim(const StringType& text, const char* whiteChars = " \t");

/// Returns a copy of the null-terminated C string with leading and trailing characters in
/// @p whiteChars removed.
///
/// Allocates a new std::string (constructed from @p text) before trimming, so this overload
/// always allocates regardless of whether any characters are stripped. See the templated
/// overload for the byte-wise comparison semantics.
///
/// @param text The string to trim (must be null-terminated, must not be null).
/// @param whiteChars Null-terminated set of single-byte characters to strip from both ends.
/// @return The trimmed string.
[[nodiscard]] std::string trim(const char* text, const char* whiteChars = " \t");

/// Tokenizes a string using the specified single-byte delimiter characters.
///
/// Splits @p inputString at every occurrence of any character in @p delimiters. Comparison
/// is byte-wise (via `std::string::find_first_of`), so multi-byte UTF-8 sequences in
/// @p delimiters are treated as a set of independent bytes rather than code points.
///
/// When @p inputString is empty an empty vector is returned. Otherwise the result has at
/// least one element, possibly empty. Each returned token is a freshly allocated copy of a
/// substring of @p inputString.
///
/// @param inputString The string to tokenize.
/// @param delimiters The set of single-byte delimiter characters.
/// @param trim When true (the default), empty tokens (produced by leading, trailing, or
///   consecutive delimiters) are dropped from the result. When false, empty tokens are
///   preserved. Note: this parameter does not strip whitespace from non-empty tokens
///   despite its name.
/// @return A vector of tokenized strings.
std::vector<std::string> tokenize(
    const std::string& inputString,
    const std::string& delimiters = " \t\r\n",
    bool trim = true);

/// Tokenizes a string_view using the specified single-byte delimiter characters.
///
/// Behaves identically to the std::string overload, except that the returned tokens are
/// std::string_view instances aliasing @p inputString's underlying buffer. The caller must
/// ensure that buffer outlives the returned vector.
///
/// @param inputString The string_view to tokenize.
/// @param delimiters The set of single-byte delimiter characters.
/// @param trim When true (the default), empty tokens (produced by leading, trailing, or
///   consecutive delimiters) are dropped from the result. When false, empty tokens are
///   preserved. Note: this parameter does not strip whitespace from non-empty tokens
///   despite its name.
/// @return A vector of tokenized string_views aliasing @p inputString.
std::vector<std::string_view>
tokenize(std::string_view inputString, std::string_view delimiters = " \t\r\n", bool trim = true);

} // namespace momentum
