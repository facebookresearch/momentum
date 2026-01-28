/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace pymomentum {

// Enum for controlling default behavior when converting empty arrays/tensors to ParameterSet.
// If the user passes an empty tensor/array for a parameter set, what kind of value to return.
// This is different for different cases: sometimes we should include all parameters,
// sometimes none, and sometimes no reasonable default is possible.
// clang-format off
enum class DefaultParameterSet { AllOnes, AllZeros, NoDefault };
// clang-format on

} // namespace pymomentum
