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
void compareCollisionGeometry(
    const CollisionGeometry_u& refCollision,
    const CollisionGeometry_u& collision);
/// Asserts that two sets of joint physical properties are equal, matching entries by joint name so
/// the comparison is independent of save/load ordering.
void comparePhysicalProperties(
    const PhysicalProperties& refProperties,
    const PhysicalProperties& properties);
void compareChars(
    const Character& refChar,
    const Character& character,
    bool withMesh = true,
    bool withParameterTransform = true,
    bool withPhysicalProperties = true);

} // namespace momentum
