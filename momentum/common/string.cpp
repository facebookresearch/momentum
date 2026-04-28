/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/string.h"

#include <algorithm>
#include <cstring>

namespace momentum {

template <typename StringType>
StringType trim(const StringType& text, const char* whiteChars) {
  size_t end = text.length();
  while (end > 0 && std::strchr(whiteChars, text[end - 1]) != nullptr) {
    end--;
  }
  if (end == 0) {
    return {};
  }
  size_t start = 0;
  while (start < end && std::strchr(whiteChars, text[start]) != nullptr) {
    start++;
  }
  if (start > 0 || end < text.length()) {
    return text.substr(start, end - start);
  }
  return text;
}

template std::string trim(const std::string& text, const char* whiteChars);
template std::string_view trim(const std::string_view& text, const char* whiteChars);

std::string trim(const char* text, const char* whiteChars) {
  return trim(std::string(text), whiteChars);
}

std::vector<std::string>
tokenize(const std::string& inputString, const std::string& delimiters, const bool trim) {
  if (inputString.empty()) {
    return {};
  }

  std::vector<std::string> results;

  size_t pos = 0;
  size_t lastPos = 0;
  while ((pos = inputString.find_first_of(delimiters, lastPos)) != std::string::npos) {
    const std::string res = inputString.substr(lastPos, pos - lastPos);
    // When `trim` is true, drop empty tokens (collapsing runs of consecutive delimiters);
    // otherwise preserve them so callers can distinguish empty fields.
    if (!res.empty() || !trim) {
      results.push_back(res);
    }
    lastPos = pos + 1;
  }
  const std::string res = inputString.substr(lastPos);
  if (!res.empty() || !trim) {
    results.push_back(res);
  }

  return results;
}

std::vector<std::string_view>
tokenize(std::string_view inputString, const std::string_view delimiters, const bool trim) {
  if (inputString.empty()) {
    return {};
  }

  std::vector<std::string_view> results;

  size_t pos = 0;
  size_t lastPos = 0;
  while ((pos = inputString.find_first_of(delimiters, lastPos)) != std::string::npos) {
    auto res = inputString.substr(lastPos, pos - lastPos);
    // See `tokenize(std::string, ...)` above for the empty-token handling rationale.
    if (!res.empty() || !trim) {
      results.push_back(res);
    }
    lastPos = pos + 1;
  }
  auto res = inputString.substr(lastPos);
  if (!res.empty() || !trim) {
    results.push_back(res);
  }

  return results;
}

} // namespace momentum
