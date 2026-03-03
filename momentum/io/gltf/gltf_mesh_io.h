/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/blend_shape.h>
#include <momentum/character/skin_weights.h>
#include <momentum/character/types.h>
#include <momentum/math/mesh.h>

#include <fx/gltf.h>

#include <tuple>
#include <vector>

namespace momentum {

/// Add mesh data from a glTF primitive.
///
/// @param[in] model The glTF document.
/// @param[in] primitive The primitive to load mesh data from.
/// @param[in,out] mesh The mesh to append data to.
/// @return The number of vertices newly loaded.
size_t addMesh(const fx::gltf::Document& model, const fx::gltf::Primitive& primitive, Mesh_u& mesh);

/// Add blend shapes from a glTF primitive.
///
/// @param[in] model The glTF document.
/// @param[in] primitive The primitive to load blend shapes from.
/// @param[in,out] blendShape The blend shape to append data to (may be nullptr, will be created if
/// needed).
/// @return The number of blend shapes newly loaded.
size_t addBlendShapes(
    const fx::gltf::Document& model,
    const fx::gltf::Primitive& primitive,
    BlendShape_u& blendShape);

/// Load skin weights from a glTF accessor.
///
/// @param[in] model The glTF document.
/// @param[in] accessorIdx The accessor index containing weight data.
/// @return A vector of weight data (4 weights per vertex).
std::vector<Vector4f> loadWeightsFromAccessor(const fx::gltf::Document& model, int32_t accessorIdx);

/// Add skin weights from a glTF primitive.
///
/// @param[in] model The glTF document.
/// @param[in] skin The skin containing joint indices.
/// @param[in] primitive The primitive to load skin weights from.
/// @param[in] nodeToObjectMap Mapping from node indices to joint indices.
/// @param[in] kNumVertices The number of vertices to load weights for.
/// @param[in,out] skinWeights The skin weights to append data to.
void addSkinWeights(
    const fx::gltf::Document& model,
    const fx::gltf::Skin& skin,
    const fx::gltf::Primitive& primitive,
    const std::vector<size_t>& nodeToObjectMap,
    size_t kNumVertices,
    SkinWeights_u& skinWeights);

/// Load skinned mesh, skin weights, and blend shapes from the model.
///
/// @param[in] model The glTF document.
/// @param[in] meshNodes The node indices containing meshes.
/// @param[in] nodeToObjectMap Mapping from node indices to joint indices.
/// @return A tuple containing the mesh, skin weights, and blend shapes.
std::tuple<Mesh_u, SkinWeights_u, BlendShape_u> loadSkinnedMesh(
    const fx::gltf::Document& model,
    const std::vector<size_t>& meshNodes,
    const std::vector<size_t>& nodeToObjectMap);

/// Bind a mesh to a single joint.
///
/// @param[in] mesh The mesh to bind.
/// @param[in] jointId The joint index to bind to.
/// @return The skin weights binding all vertices to the specified joint.
SkinWeights_u bindMeshToJoint(const Mesh_u& mesh, size_t jointId);

} // namespace momentum
