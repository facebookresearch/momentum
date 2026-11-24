/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/io/gltf/gltf_skeleton_io.h>

#include <fx/gltf.h>

#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

namespace momentum {

/// Internal function to load a character from a glTF model with node-to-object mapping.
///
/// This function is used internally by other glTF I/O functions that need access to
/// the node-to-object mapping for resolving references between glTF nodes and momentum objects.
/// Like loadGltfCharacter(), it returns the character with the most joints when the document has
/// several, and warns.
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

/// The character the single-character loaders return: the one with the most joints, the first of
/// those on a tie. Warns when there is more than one character and throws when there is none.
///
/// @param[in] characters The result of loadGltfCharactersInternal().
/// @return A reference into @p characters.
CharacterWithNodeMap& selectGltfCharacter(std::vector<CharacterWithNodeMap>& characters);

/// The first animation of @p model with a translation, rotation or scale channel that targets a
/// joint of @p character, or nullptr.
///
/// @param[in] model The glTF document.
/// @param[in] character The character, as loaded from @p model.
/// @param[in] nodeToObjectMap The character's node-to-object map.
const fx::gltf::Animation* findAnimationForCharacter(
    const fx::gltf::Document& model,
    const Character& character,
    std::span<const size_t> nodeToObjectMap);

/// Read the channels of @p animation that target @p character's joints as skeleton states.
///
/// @param[in] model The glTF document.
/// @param[in] animation An animation of @p model.
/// @param[in] character The character, as loaded from @p model.
/// @param[in] nodeToObjectMap The character's node-to-object map.
/// @return One skeleton state per frame, and the frame times in seconds.
std::tuple<std::vector<SkeletonState>, std::vector<float>> loadSkeletonStatesForCharacter(
    const fx::gltf::Document& model,
    const fx::gltf::Animation& animation,
    const Character& character,
    std::span<const size_t> nodeToObjectMap);

/// The momentum "motion" object of the character whose root node is @p rootNode: the one stored on
/// that node; else the document-level one if @p isOnlyCharacter and no node stores a motion (a file
/// written before motion was stored per character); else an empty object.
///
/// @param[in] model The glTF document to read.
/// @param[in] rootNode Index of the character's root node in @p model.
/// @param[in] isOnlyCharacter Whether the character is the only one in @p model.
/// @return A reference to the motion object, or to an empty object when the character has none.
const nlohmann::json&
getCharacterMotionJson(const fx::gltf::Document& model, uint32_t rootNode, bool isOnlyCharacter);

} // namespace momentum
