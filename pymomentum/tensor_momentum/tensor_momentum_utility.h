/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>

#include <ATen/ATen.h>

namespace pymomentum {

// Verifies that the bone indices are empty or all valid.
// Throws runtime_error when one index is out of range.
void checkValidBoneIndex(at::Tensor idx, const momentum::Character& character, const char* name);

// allow_missing means -1 is allowed:
void checkValidParameterIndex(
    at::Tensor idx,
    const momentum::Character& character,
    const char* name,
    bool allow_missing);

// allow_missing means -1 is allowed:
void checkValidVertexIndex(
    at::Tensor idx,
    const momentum::Character& character,
    const char* name,
    bool allow_missing);

// If the user passes an empty tensor for a joint set, what kind of
// value to return.  This is different for different cases: sometimes we
// should include all joints, sometimes none, and sometimes no reasonable
// default is possible.
enum class DefaultJointSet { ALL_ONES, ALL_ZEROS, NO_DEFAULT };

std::vector<bool> tensorToJointSet(
    const momentum::Skeleton& skeleton,
    at::Tensor jointSet,
    DefaultJointSet defaultValue);

} // namespace pymomentum
