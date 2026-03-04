/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>

#include <fx/gltf.h>

#include <vector>

namespace momentum {

/// Internal function to load a character from a glTF model with node-to-object mapping.
///
/// This function is used internally by other glTF I/O functions that need access to
/// the node-to-object mapping for resolving references between glTF nodes and momentum objects.
///
/// @param[in] model The glTF document to load from.
/// @param[out] nodeToObjectMap Output mapping from glTF node indices to momentum object indices.
/// @return The loaded Character object.
Character loadGltfCharacterInternal(
    const fx::gltf::Document& model,
    std::vector<size_t>& nodeToObjectMap);

} // namespace momentum
