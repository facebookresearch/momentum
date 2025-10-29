/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/locator.h>
#include <momentum/character/skinned_locator.h>

#include <pybind11/pybind11.h>

namespace pymomentum {

void registerLocatorBindings(
    pybind11::class_<momentum::Locator>& locatorClass,
    pybind11::class_<momentum::SkinnedLocator>& skinnedLocatorClass);

} // namespace pymomentum
