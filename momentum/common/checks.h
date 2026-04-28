/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/// Runtime check macros with two interchangeable backends:
/// - When @c MOMENTUM_WITH_XR_LOGGER is defined, failures are reported via the XR logger's
///   @c XR_CHECK family (which typically logs and aborts).
/// - Otherwise, failures throw a @c std::runtime_error via @c MT_THROW_IF.
/// The trailing variadic arguments are an optional fmt-style message and its format arguments.

#if defined(MOMENTUM_WITH_XR_LOGGER)

#include <logging/Checks.h>

/// Asserts that @p condition is true; on failure, reports the optional fmt-style message.
#define MT_CHECK(condition, ...) XR_CHECK(condition, __VA_ARGS__)

/// Asserts that @p val1 < @p val2; on failure, reports the optional fmt-style message
/// along with both operand values.
#define MT_CHECK_LT(val1, val2, ...) XR_CHECK_DETAIL_OP1(val1, val2, <, ##__VA_ARGS__)

/// Asserts that @p ptr is not null; on failure, reports the optional fmt-style message.
#define MT_CHECK_NOTNULL(ptr, ...) XR_CHECK(ptr != nullptr, ##__VA_ARGS__)

#else

#include <momentum/common/exception.h>

/// Asserts that @p condition is true; on failure, throws @c std::runtime_error with the
/// optional fmt-style message.
#define MT_CHECK(condition, ...) MT_THROW_IF(!(condition), ##__VA_ARGS__)

/// Asserts that @p val1 < @p val2; on failure, throws @c std::runtime_error with the
/// optional fmt-style message. Note: unlike the XR logger backend, operand values are not
/// auto-included in the message.
#define MT_CHECK_LT(val1, val2, ...) MT_THROW_IF(!((val1) < (val2)), ##__VA_ARGS__)

/// Asserts that @p ptr is not null; on failure, throws @c std::runtime_error with the
/// optional fmt-style message.
#define MT_CHECK_NOTNULL(ptr, ...) MT_THROW_IF((ptr) == nullptr, ##__VA_ARGS__)

#endif
