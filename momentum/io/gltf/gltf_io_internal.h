/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/io/gltf/gltf_skeleton_io.h>

#include <fx/gltf.h>

#include <cstdint>
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

/// A character loaded from a glTF document with the data needed to resolve the document's nodes to
/// it.
struct CharacterWithNodeMap {
  /// The loaded character.
  Character character;
  /// Maps each node index to the character's joint, collision or locator index; kInvalidIndex for
  /// nodes that do not belong to the character.
  std::vector<size_t> nodeToObjectMap;
  /// The character's top-level scene node (the root node GltfBuilder writes for it), or
  /// kInvalidNodeId if there is none.
  uint32_t rootNode = kInvalidNodeId;
};

/// Load every character of a glTF document, in the order of loadGltfCharacters().
///
/// @param[in] model The glTF document to load from.
/// @return One entry per character.
std::vector<CharacterWithNodeMap> loadGltfCharactersInternal(const fx::gltf::Document& model);

} // namespace momentum
