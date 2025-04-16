/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// clang-format off
#include <cstdint>  // Must come first. See https://github.com/p-ranav/indicators/pull/124
#include <indicators/progress_bar.hpp>
// clang-format on

#include <string>

namespace momentum {

// A simple progress bar that prints hash marks (e.g., "[===>  ] 60% Name")
class ProgressBar {
  static constexpr int64_t kMaxWidth = 80;

 public:
  /// @param name Displayed prefix (visible=true only)
  /// @param numOperations Total operations (determines hash ratio)
  ProgressBar(const std::string& name, int64_t numOperations);
  void increment(int64_t count = 1);
  void set(int64_t count);

 private:
  indicators::ProgressBar bar_;
};

} // namespace momentum
