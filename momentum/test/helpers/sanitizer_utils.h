/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace momentum {

/// Detect sanitizer builds at compile time.
///
/// Clang uses __has_feature, GCC defines __SANITIZE_ADDRESS__ /
/// __SANITIZE_THREAD__.
constexpr bool isSanitizerBuild() {
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
  return true;
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
  return true;
#endif
  return false;
}

} // namespace momentum
