/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/progress_bar.h"

#include <stdexcept>
#include <string>

namespace momentum {

ProgressBar::ProgressBar(const std::string& prefix, const size_t numOperations) {
  using namespace indicators;

  // Reserve 9 columns for the brackets, leading space, and percentage suffix
  // (e.g., " 100%") so the total rendered line stays within kMaxWidth.
  constexpr size_t kReservedColumns = 9;
  // Throw `std::invalid_argument` directly rather than via `MT_THROW_IF` and avoid pulling in
  // any momentum or third-party headers — `:common` already depends on `:progress_bar`, so
  // reaching back through `momentum/common/exception.h` would create a cyclic dependency, and
  // `<fmt>` is not in the `:progress_bar` BUCK target's deps either.
  if (prefix.size() + kReservedColumns >= kMaxWidth) {
    throw std::invalid_argument(
        "ProgressBar prefix is too long: prefix.size()=" + std::to_string(prefix.size()) +
        " leaves no room within kMaxWidth=" + std::to_string(kMaxWidth));
  }
  bar_.set_option(option::BarWidth(kMaxWidth - prefix.size() - kReservedColumns));
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
