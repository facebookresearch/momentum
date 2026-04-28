/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/// Provides the ::filesystem alias used throughout Momentum.
///
/// When @c MOMENTUM_WITH_PORTABILITY is defined the alias resolves to the
/// internal portability shim; otherwise it resolves to @c std::filesystem.
/// Including this header (rather than @c <filesystem> directly) lets call
/// sites work unchanged on platforms that lack a complete C++17 filesystem
/// implementation.
#if defined(MOMENTUM_WITH_PORTABILITY)

#include <portability/Filesystem.h>

#else

#include <filesystem>

namespace filesystem = std::filesystem;

#endif
