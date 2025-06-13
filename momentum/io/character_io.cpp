/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/character_io.h"

#include "momentum/character/character.h"
#include "momentum/io/fbx/fbx_io.h"
#include "momentum/io/gltf/gltf_io.h"
#include "momentum/io/shape/pose_shape_io.h"
#include "momentum/io/skeleton/locator_io.h"
#include "momentum/io/skeleton/mppca_io.h"
#include "momentum/io/skeleton/parameter_transform_io.h"
#include "momentum/io/skeleton/parameters_io.h"
#include "momentum/math/fwd.h"

#include <algorithm>
#include <optional>

namespace momentum {

namespace {

/// Parses the character format from the given file path.
///
/// @param[in] filepath The file path to the character file.
/// @return The parsed CharacterFormat enumeration value.
[[nodiscard]] CharacterFormat parseCharacterFormat(const filesystem::path& filepath) noexcept {
  std::string ext = filepath.extension().string();
  if (ext.empty()) {
    return CharacterFormat::Unknown;
  }

  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  if ((ext == ".glb") || (ext == ".gltf")) {
    return CharacterFormat::Gltf;
  } else if (ext == ".fbx") {
    return CharacterFormat::Fbx;
  } else {
    return CharacterFormat::Unknown;
  }
}

[[nodiscard]] std::optional<Character> loadCharacterByFormat(
    const CharacterFormat format,
    const filesystem::path& filepath) {
  if (format == CharacterFormat::Gltf) {
    return loadGltfCharacter(filepath);
  } else if (format == CharacterFormat::Fbx) {
    return loadFbxCharacter(filepath, KeepLocators::Yes);
  } else {
    return {};
  }
}

[[nodiscard]] std::optional<Character> loadCharacterByFormatFromBuffer(
    const CharacterFormat format,
    const gsl::span<const std::byte> fileBuffer) {
  if (format == CharacterFormat::Gltf) {
    return loadGltfCharacter(fileBuffer);
  } else if (format == CharacterFormat::Fbx) {
    return loadFbxCharacter(fileBuffer, KeepLocators::Yes);
  } else {
    return {};
  }
}

} // namespace

Character loadFullCharacter(
    const std::string& characterPath,
    const std::string& parametersPath,
    const std::string& locatorsPath) {
  // Parse format
  const auto format = parseCharacterFormat(characterPath);
  MT_THROW_IF(
      format == CharacterFormat::Unknown, "Unknown character format for path: {}", characterPath);

  // Load character
  auto character = loadCharacterByFormat(format, characterPath);
  MT_THROW_IF(!character, "Failed to load buffered character");

  // load parameter transform
  if (!parametersPath.empty()) {
    auto def = loadMomentumModel(parametersPath);
    loadParameters(def, *character);
  }

  // Load locators
  if (!locatorsPath.empty()) {
    (*character).locators =
        loadLocators(locatorsPath, (*character).skeleton, (*character).parameterTransform);
  }

  return character.value();
}

Character loadFullCharacterFromBuffer(
    CharacterFormat format,
    const gsl::span<const std::byte> characterBuffer,
    const gsl::span<const std::byte> paramBuffer,
    const gsl::span<const std::byte> locBuffer) {
  // Load character
  auto character = loadCharacterByFormatFromBuffer(format, characterBuffer);
  MT_THROW_IF(!character, "Failed to load buffered character");

  // load parameter transform
  if (!paramBuffer.empty()) {
    auto def = loadMomentumModelFromBuffer(paramBuffer);
    loadParameters(def, *character);
  }

  // Load locators
  if (!locBuffer.empty()) {
    (*character).locators =
        loadLocatorsFromBuffer(locBuffer, (*character).skeleton, (*character).parameterTransform);
  }

  return character.value();
}

} // namespace momentum
