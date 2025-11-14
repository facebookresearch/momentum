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

// Matching methods
void compareMeshes(const Mesh_u& refMesh, const Mesh_u& mesh);
void compareChars(
    const Character& refChar,
    const Character& character,
    bool withMesh = true,
    bool withParameterTransform = true);

} // namespace momentum
