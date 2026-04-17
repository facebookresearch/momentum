/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(MOMENTUM_WITH_XR_LOGGER)

#include <logging/Checks.h>

#define MT_CHECK(condition, ...) XR_CHECK(condition, __VA_ARGS__)
#define MT_CHECK_LT(val1, val2, ...) XR_CHECK_DETAIL_OP1(val1, val2, <, ##__VA_ARGS__)
#define MT_CHECK_NOTNULL(ptr, ...) XR_CHECK(ptr != nullptr, ##__VA_ARGS__)

#else

#include <momentum/common/exception.h>

#define MT_CHECK(condition, ...) MT_THROW_IF(!(condition), ##__VA_ARGS__)
#define MT_CHECK_LT(val1, val2, ...) MT_THROW_IF(!((val1) < (val2)), ##__VA_ARGS__)
#define MT_CHECK_NOTNULL(ptr, ...) MT_THROW_IF((ptr) == nullptr, ##__VA_ARGS__)

#endif
