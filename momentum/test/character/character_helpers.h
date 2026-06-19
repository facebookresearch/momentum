/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/math/mppca.h>

namespace momentum {

/// Creates a character with a customizable number of joints.
///
/// @param numJoints The number of joints in the resulting character.
/// @note: The `numJoints` parameter should be equal to 3 or greater.
[[nodiscard]] Character createTestCharacter(size_t numJoints = 3);
// TODO: Currently, the number of parameters is fixed to 10, affecting only the first 3 joints.
// Additional work is needed to scale the number of parameters with the number of joints.

template <typename T>
[[nodiscard]] std::shared_ptr<const MppcaT<T>> createDefaultPosePrior();

[[nodiscard]] Character withTestBlendShapes(const Character& character);

[[nodiscard]] Character withTestFaceExpressionBlendShapes(const Character& character);

/// Returns a copy of `character` with deterministic physical mass properties attached to its first
/// two joints, for exercising physical-properties I/O round trips.
///
/// @pre `character.skeleton.joints.size() >= 2`.
[[nodiscard]] Character withTestPhysicalProperties(const Character& character);

} // namespace momentum
