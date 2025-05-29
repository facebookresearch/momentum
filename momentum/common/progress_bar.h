/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <indicators/progress_bar.hpp>

#include <string>

namespace momentum {

// A simple progress bar that prints hash marks (e.g., "Name [===>  ] 60%")
class ProgressBar {
 public:
  /// @param prefix Displayed prefix
  /// @param numOperations Total operations (determines progress ratio)
  ProgressBar(const std::string& prefix, size_t numOperations);

  /// Increments the progress by the given count (default 1)
  void increment(size_t count = 1);

  /// Sets the progress to the given count
  void set(size_t count);

  /// Returns the current progress in the range [0, 100]
  [[nodiscard]] size_t getCurrentProgress();

 private:
  static constexpr size_t kMaxWidth = 80;
  indicators::ProgressBar bar_;
};

} // namespace momentum
