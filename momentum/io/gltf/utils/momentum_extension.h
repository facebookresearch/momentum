/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <nlohmann/json.hpp>

namespace momentum {

/// The name of the glTF extension that stores momentum's own data (rig, motion, markers and other
/// metadata) in a document, node, mesh or primitive.
inline constexpr const char* kMomentumExtensionName = "FB_momentum";

/// Whether a glTF `extensionsAndExtras` object has the FB_momentum extension, even an empty one.
///
/// @param[in] extensionsAndExtras The `extensionsAndExtras` of a document, node, mesh or primitive.
/// @return True if the extension is present.
[[nodiscard]] inline bool hasMomentumExtension(const nlohmann::json& extensionsAndExtras) {
  const auto extensions = extensionsAndExtras.find("extensions");
  return extensions != extensionsAndExtras.end() && extensions->contains(kMomentumExtensionName);
}

/// The FB_momentum extension object in a glTF `extensionsAndExtras` object, or an empty object.
///
/// @param[in] extensionsAndExtras The `extensionsAndExtras` of a document, node, mesh or primitive.
/// @return A reference into @p extensionsAndExtras, or to a static empty object.
[[nodiscard]] inline const nlohmann::json& findMomentumExtension(
    const nlohmann::json& extensionsAndExtras) {
  static const nlohmann::json kEmpty = nlohmann::json::object();
  const auto extensions = extensionsAndExtras.find("extensions");
  if (extensions == extensionsAndExtras.end()) {
    return kEmpty;
  }
  const auto momentum = extensions->find(kMomentumExtensionName);
  return momentum == extensions->end() ? kEmpty : *momentum;
}

/// The FB_momentum extension object in a glTF `extensionsAndExtras` object, added as an empty
/// object if it is not there.
///
/// @param[in,out] extensionsAndExtras The `extensionsAndExtras` of a document, node, mesh or
/// primitive.
/// @return A reference to the extension object in @p extensionsAndExtras.
inline nlohmann::json& addMomentumExtension(nlohmann::json& extensionsAndExtras) {
  if (extensionsAndExtras.count("extensions") == 0 ||
      extensionsAndExtras["extensions"].count(kMomentumExtensionName) == 0) {
    extensionsAndExtras["extensions"][kMomentumExtensionName] = nlohmann::json::object();
  }
  return extensionsAndExtras["extensions"][kMomentumExtensionName];
}

} // namespace momentum
