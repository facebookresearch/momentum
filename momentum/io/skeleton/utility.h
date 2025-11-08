/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace momentum {
namespace io_detail {

/// Represents a single segment of a section (when sections are split across the file)
struct SectionSegment {
  std::string content;
  size_t startLine;
};

/// Holds all segments of a section and provides line-aware iteration
/// This is used internally to track line numbers across potentially duplicate sections
class SectionContent {
 public:
  void addSegment(std::string_view content, size_t startLine) {
    segments_.push_back({std::string(content), startLine});
  }

  [[nodiscard]] bool empty() const {
    return segments_.empty();
  }

  /// Iterator that walks through all segments while tracking line numbers
  class LineIterator {
   public:
    explicit LineIterator(const std::vector<SectionSegment>& segments);

    /// Get the next line from the sections
    /// @return true if a line was read, false if end of all segments reached
    bool getline(std::string& line);

    /// Get the current line number in the original file
    [[nodiscard]] size_t currentLine() const;

   private:
    const std::vector<SectionSegment>& segments_;
    size_t segmentIndex_;
    size_t lineInSegment_;
    std::istringstream currentStream_;
  };

  [[nodiscard]] LineIterator begin() const {
    return LineIterator(segments_);
  }

  /// Get all content as a single concatenated string (for backward compatibility)
  [[nodiscard]] std::string toString() const;

 private:
  std::vector<SectionSegment> segments_;
};

} // namespace io_detail
} // namespace momentum
