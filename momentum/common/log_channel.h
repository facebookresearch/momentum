/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(MOMENTUM_WITH_XR_LOGGER)

#include <logging/Log.h>

/// XR logger channel name used by all Momentum logging macros.
///
/// Pass this as the channel argument to XR_LOGC* macros, e.g.
/// `XR_LOGCI(MOMENTUM_LOG_CHANNEL, "message")`. Only defined when
/// `MOMENTUM_WITH_XR_LOGGER` is set at build time.
#define MOMENTUM_LOG_CHANNEL "MOMENTUM"

#endif
