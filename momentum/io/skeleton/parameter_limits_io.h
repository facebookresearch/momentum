/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/parameter_limits.h>

namespace momentum {

namespace io_detail {
// Forward declaration
class SectionContent;
} // namespace io_detail

// Internal overload for use within momentum parsing
ParameterLimits parseParameterLimits(
    const io_detail::SectionContent& content,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform);

// Public API for external use
ParameterLimits parseParameterLimits(
    const std::string& data,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform,
    size_t lineOffset = 0);

std::string writeParameterLimits(
    const ParameterLimits& parameterLimits,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform);

} // namespace momentum
