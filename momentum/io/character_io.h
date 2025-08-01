/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/math/fwd.h>
#include <momentum/math/types.h>

#include <string>

namespace momentum {

/// Represents the supported character formats.
enum class CharacterFormat : uint8_t {
  Fbx, ///< FBX file format.
  Gltf, ///< glTF file format.
  Usd, ///< USD file format.
  Unknown ///< Unknown or unsupported file format.
};

/// High level function to load a character of any type, with a local path.
///
/// @param[in] characterPath The path to the character file.
/// @param[in] parametersPath The optional path to the file containing additional parameters for the
/// character.
/// @param[in] locatorsPath The optional path to the file containing additional locators for the
/// character.
/// @return The loaded Character object.
///
/// Currently, only supports .glb and .fbx. If you want to parse from a non-local path, you may need
/// to parse it using your favorite resource retriever into a buffer and use the buffer version of
/// this function.
[[nodiscard]] Character loadFullCharacter(
    const std::string& characterPath,
    const std::string& parametersPath = std::string(),
    const std::string& locatorsPath = std::string());

/// Buffer version of loadFullCharacter function, supports .glb and .fbx file formats.
///
/// @param[in] format The character file format.
/// @param[in] characterBuffer The buffer containing the character data.
/// @param[in] paramBuffer The optional buffer containing additional parameters for the character.
/// @param[in] locBuffer The optional buffer containing additional locators for the character.
/// @return The loaded Character object.
[[nodiscard]] Character loadFullCharacterFromBuffer(
    CharacterFormat format,
    gsl::span<const std::byte> characterBuffer,
    gsl::span<const std::byte> paramBuffer = gsl::span<const std::byte>(),
    gsl::span<const std::byte> locBuffer = gsl::span<const std::byte>());

} // namespace momentum
