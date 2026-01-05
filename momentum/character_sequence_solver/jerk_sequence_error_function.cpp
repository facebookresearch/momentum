/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/jerk_sequence_error_function.h"

#include "momentum/character/character.h"

namespace momentum {

template <typename T>
JerkSequenceErrorFunctionT<T>::JerkSequenceErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : FiniteDifferenceSequenceErrorFunctionT<T>(skel, pt, {T(1), T(-3), T(3), T(-1)}) {}

template <typename T>
JerkSequenceErrorFunctionT<T>::JerkSequenceErrorFunctionT(const Character& character)
    : FiniteDifferenceSequenceErrorFunctionT<T>(
          character.skeleton,
          character.parameterTransform,
          {T(1), T(-3), T(3), T(-1)}) {}

template class JerkSequenceErrorFunctionT<float>;
template class JerkSequenceErrorFunctionT<double>;

} // namespace momentum
