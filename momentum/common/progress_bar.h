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

/// A terminal progress bar rendered as `Name [===>   ] 60%`.
///
/// Wraps `indicators::ProgressBar` with a fixed visual style (bold, percentage
/// shown). All mutating operations write to the terminal (stdout) as a side
/// effect. Not thread-safe.
class ProgressBar {
 public:
  /// Constructs the progress bar and renders its initial (zero-progress) state.
  ///
  /// @param prefix Text shown before the bar.
  /// @param numOperations Total operation count corresponding to 100%.
  /// @pre `prefix.size() + 9 < kMaxWidth`, otherwise the computed bar width
  ///      underflows `size_t`.
  // TODO: Validate that `prefix.size() + 9 < kMaxWidth` to avoid a size_t
  // underflow when computing the bar width in the constructor.
  ProgressBar(const std::string& prefix, size_t numOperations);

  /// Advances progress by `count` (default 1) and redraws the bar.
  void increment(size_t count = 1);

  /// Sets progress to the absolute value `count` and redraws the bar.
  void set(size_t count);

  /// Returns the current absolute progress count (not a percentage).
  ///
  /// The value is in the range [0, numOperations] as supplied to the
  /// constructor.
  [[nodiscard]] size_t getCurrentProgress();

 private:
  static constexpr size_t kMaxWidth = 80;
  indicators::ProgressBar bar_;
};

} // namespace momentum
