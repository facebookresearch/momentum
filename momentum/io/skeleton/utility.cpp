/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/skeleton/utility.h"

namespace momentum::io_detail {

// LineIterator implementation
SectionContent::LineIterator::LineIterator(const std::vector<SectionSegment>& segments)
    : segments_(segments), segmentIndex_(0), lineInSegment_(0) {
  if (!segments_.empty()) {
    currentStream_.str(segments_[0].content);
  }
}

bool SectionContent::LineIterator::getline(std::string& line) {
  while (segmentIndex_ < segments_.size()) {
    if (std::getline(currentStream_, line)) {
      ++lineInSegment_;
      return true;
    }

    // Move to next segment
    ++segmentIndex_;
    lineInSegment_ = 0;
    if (segmentIndex_ < segments_.size()) {
      currentStream_.clear();
      currentStream_.str(segments_[segmentIndex_].content);
    }
  }
  return false;
}

size_t SectionContent::LineIterator::currentLine() const {
  if (segmentIndex_ >= segments_.size()) {
    return 0;
  }
  return segments_[segmentIndex_].startLine + lineInSegment_ - 1;
}

std::string SectionContent::toString() const {
  std::string result;
  for (const auto& segment : segments_) {
    result += segment.content;
  }
  return result;
}

} // namespace momentum::io_detail
