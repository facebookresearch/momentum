/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/character_io.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/io/fbx/fbx_io.h"
#include "momentum/io/gltf/gltf_io.h"
#include "momentum/io/shape/pose_shape_io.h"
#include "momentum/io/skeleton/locator_io.h"
#include "momentum/io/skeleton/mppca_io.h"
#include "momentum/io/skeleton/parameter_transform_io.h"
#include "momentum/io/skeleton/parameters_io.h"
#ifdef MOMENTUM_IO_WITH_USD
#include "momentum/io/usd/usd_io.h"
#endif
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
  } else if ((ext == ".usd") || (ext == ".usda") || (ext == ".usdc")) {
    return CharacterFormat::Usd;
  } else {
    return CharacterFormat::Unknown;
  }
}

[[nodiscard]] std::optional<Character> loadCharacterByFormat(
    const CharacterFormat format,
    const filesystem::path& filepath,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No) {
  if (format == CharacterFormat::Gltf) {
    return loadGltfCharacter(filepath);
  } else if (format == CharacterFormat::Fbx) {
    return loadFbxCharacter(filepath, KeepLocators::Yes, Permissive::No, loadBlendShapes);
  } else if (format == CharacterFormat::Usd) {
#ifdef MOMENTUM_IO_WITH_USD
    return loadUsdCharacter(filepath);
#else
    MT_THROW("USD format is not supported in this build");
#endif
  } else {
    return {};
  }
}

[[nodiscard]] std::optional<Character> loadCharacterByFormatFromBuffer(
    const CharacterFormat format,
    const std::span<const std::byte> fileBuffer,
    LoadBlendShapes loadBlendShapes = LoadBlendShapes::No) {
  if (format == CharacterFormat::Gltf) {
    return loadGltfCharacter(fileBuffer);
  } else if (format == CharacterFormat::Fbx) {
    return loadFbxCharacter(fileBuffer, KeepLocators::Yes, Permissive::No, loadBlendShapes);
  } else if (format == CharacterFormat::Usd) {
#ifdef MOMENTUM_IO_WITH_USD
    return loadUsdCharacter(fileBuffer);
#else
    MT_THROW("USD format is not supported in this build");
#endif
  } else {
    return {};
  }
}

} // namespace

Character loadFullCharacter(
    const std::string& characterPath,
    const std::string& parametersPath,
    const std::string& locatorsPath,
    LoadBlendShapes loadBlendShapes) {
  // Parse format
  const auto format = parseCharacterFormat(characterPath);
  MT_THROW_IF(
      format == CharacterFormat::Unknown, "Unknown character format for path: {}", characterPath);

  // Load character
  auto character = loadCharacterByFormat(format, characterPath, loadBlendShapes);
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
    const std::span<const std::byte> characterBuffer,
    const std::span<const std::byte> paramBuffer,
    const std::span<const std::byte> locBuffer,
    LoadBlendShapes loadBlendShapes) {
  // Load character
  auto character = loadCharacterByFormatFromBuffer(format, characterBuffer, loadBlendShapes);
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

void saveCharacter(
    const filesystem::path& filename,
    const Character& character,
    const float fps,
    const MatrixXf& motion,
    std::span<const std::vector<Marker>> markerSequence,
    const FileSaveOptions& options) {
  // Parse format from file extension
  const auto format = parseCharacterFormat(filename);
  MT_THROW_IF(
      format == CharacterFormat::Unknown,
      "Unknown character format for path: {}. Supported formats: .fbx, .glb, .gltf",
      filename.string());

  if (format == CharacterFormat::Gltf) {
    saveGltfCharacter(
        filename,
        character,
        fps,
        {character.parameterTransform.name, motion},
        {},
        markerSequence,
        options);
  } else if (format == CharacterFormat::Fbx) {
    // Save as FBX
    saveFbx(
        filename, character, motion, VectorXf(), static_cast<double>(fps), markerSequence, options);
  } else {
    MT_THROW(
        "{} is not a supported format. Supported formats: .fbx, .glb, .gltf", filename.string());
  }
}

void saveCharacter(
    const filesystem::path& filename,
    const Character& character,
    const float fps,
    std::span<const SkeletonState> skeletonStates,
    std::span<const std::vector<Marker>> markerSequence,
    const FileSaveOptions& options) {
  // Parse format from file extension
  const auto format = parseCharacterFormat(filename);
  MT_THROW_IF(
      format == CharacterFormat::Unknown,
      "Unknown character format for path: {}. Supported formats: .fbx, .glb, .gltf",
      filename.string());

  if (format == CharacterFormat::Gltf) {
    saveGltfCharacter(filename, character, fps, skeletonStates, markerSequence, options);
  } else if (format == CharacterFormat::Fbx) {
    // Save as FBX
    saveFbxWithSkeletonStates(
        filename, character, skeletonStates, static_cast<double>(fps), markerSequence, options);
  } else {
    MT_THROW(
        "{} is not a supported format. Supported formats: .fbx, .glb, .gltf", filename.string());
  }
}

} // namespace momentum
