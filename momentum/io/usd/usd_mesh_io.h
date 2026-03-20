/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/blend_shape.h>
#include <momentum/character/skin_weights.h>
#include <momentum/math/mesh.h>

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdSkel/skeleton.h>

PXR_NAMESPACE_USING_DIRECTIVE

namespace momentum {

/// Load a mesh from a USD stage.
///
/// Reads vertices, faces, and vertex colors from the first UsdGeomMesh prim
/// found on the stage.
///
/// @param[in] stage The USD stage to read from.
/// @return The loaded Mesh.
Mesh loadMeshFromUsd(const UsdStageRefPtr& stage);

/// Load skin weights from a USD stage.
///
/// Reads joint indices and weights from the first skinned UsdGeomMesh prim
/// found on the stage.
///
/// @param[in] stage The USD stage to read from.
/// @param[in] numVertices The expected number of vertices.
/// @return The loaded SkinWeights.
SkinWeights loadSkinWeightsFromUsd(const UsdStageRefPtr& stage, size_t numVertices);

/// Save a mesh to a USD stage.
///
/// Writes vertices and faces to the given UsdGeomMesh prim.
///
/// @param[in] mesh The Mesh to save.
/// @param[in] meshPrim The UsdGeomMesh prim to write to.
void saveMeshToUsd(const Mesh& mesh, UsdGeomMesh& meshPrim);

/// Save skin weights to a USD stage.
///
/// Writes joint indices and weights to the given UsdGeomMesh prim via
/// UsdSkelBindingAPI, and binds it to the specified skeleton.
///
/// @param[in] skinWeights The SkinWeights to save.
/// @param[in] meshPrim The UsdGeomMesh prim to write skinning data to.
/// @param[in] skelPrim The UsdSkelSkeleton prim to bind to.
void saveSkinWeightsToUsd(
    const SkinWeights& skinWeights,
    UsdGeomMesh& meshPrim,
    const UsdSkelSkeleton& skelPrim);

/// Save blend shapes to a USD stage.
///
/// Creates UsdSkelBlendShape prims under the mesh and binds them via
/// UsdSkelBindingAPI.
///
/// @param[in] blendShape The BlendShape to save.
/// @param[in] meshPrim The UsdGeomMesh prim to attach blend shapes to.
void saveBlendShapesToUsd(const BlendShape& blendShape, UsdGeomMesh& meshPrim);

/// Load blend shapes from a USD stage.
///
/// Reads UsdSkelBlendShape prims bound to the first mesh found.
///
/// @param[in] stage The USD stage to read from.
/// @param[in] numVertices The expected number of vertices.
/// @return The loaded BlendShape, or nullptr if none found.
std::shared_ptr<BlendShape> loadBlendShapesFromUsd(const UsdStageRefPtr& stage, size_t numVertices);

} // namespace momentum
