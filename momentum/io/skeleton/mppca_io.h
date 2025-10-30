/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/fwd.h>

#include <span>

#include <istream>

namespace momentum {

std::shared_ptr<const Mppca> loadMppca(std::istream& inputStream);

std::shared_ptr<const Mppca> loadMppca(const std::string& name);

std::shared_ptr<const Mppca> loadMppca(std::span<const unsigned char> posePriorDataRaw);

void saveMppca(const Mppca& mppca, const std::string& name);

} // namespace momentum
