/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/progress_bar.h"

namespace momentum {

ProgressBar::ProgressBar(const std::string& prefix, const size_t numOperations) {
  using namespace indicators;

  bar_.set_option(option::BarWidth(kMaxWidth - prefix.size() - 9));
  bar_.set_option(option::MaxProgress(numOperations));
  bar_.set_option(option::Start{"["});
  bar_.set_option(option::Fill{"="});
  bar_.set_option(option::Lead{">"});
  bar_.set_option(option::Remainder{" "});
  bar_.set_option(option::End{"]"});
  bar_.set_option(option::PrefixText{prefix});
  bar_.set_option(option::ShowPercentage{true});
  bar_.set_option(option::FontStyles{std::vector<FontStyle>{FontStyle::bold}});
}

void ProgressBar::increment(size_t count) {
  bar_.set_progress(bar_.current() + count);
}

void ProgressBar::set(size_t count) {
  bar_.set_progress(count);
}

size_t ProgressBar::getCurrentProgress() {
  return bar_.current();
}

} // namespace momentum
