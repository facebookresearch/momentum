/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/joint.h>
#include <momentum/character/skeleton.h>

#include <pybind11/pybind11.h>

namespace pymomentum {

void registerJointBindings(pybind11::class_<momentum::Joint>& jointClass);

void registerSkeletonBindings(pybind11::class_<momentum::Skeleton>& skeletonClass);

} // namespace pymomentum
