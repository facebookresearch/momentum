/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace momentum {

// File format in which the character is saved
enum class GltfFileFormat {
  EXTENSION = 0, // The file extension is used for deduction (e.g. ".gltf" --> ASCII)
  GLTF_BINARY = 1, // Binary format (generally .glb)
  GLTF_ASCII = 2, // ASCII format (generally .gltf)
};

} // namespace momentum
