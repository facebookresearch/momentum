/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/blend_shape.h>
#include <momentum/common/filesystem.h>

#include <iosfwd>

namespace momentum {

/// Loads blend shape vectors from a file without base shape.
///
/// @param filename Path to the blend shape base data file
/// @param expectedShapes Limits number of shape vectors loaded if > 0, otherwise loads all
/// @param expectedVertices Limits number of vertices loaded if > 0, otherwise loads all
/// @return BlendShapeBase object with loaded shape vectors
BlendShapeBase loadBlendShapeBase(
    const filesystem::path& filename,
    int expectedShapes = -1,
    int expectedVertices = -1);

/// Loads blend shape vectors from a stream without base shape.
///
/// @param data Input stream containing blend shape base data
/// @param expectedShapes Limits number of shape vectors loaded if > 0, otherwise loads all
/// @param expectedVertices Limits number of vertices loaded if > 0, otherwise loads all
/// @return BlendShapeBase object with loaded shape vectors
BlendShapeBase
loadBlendShapeBase(std::istream& data, int expectedShapes = -1, int expectedVertices = -1);

/// Loads a blend shape from a file, including base shape and shape vectors.
///
/// @param filename Path to the blend shape data file
/// @param expectedShapes Limits number of shape vectors loaded if > 0, otherwise loads all
/// @param expectedVertices Limits number of vertices loaded if > 0, otherwise loads all
/// @return BlendShape object with loaded data
///
/// @note Only supports local files. For non-local paths, use the stream version.
BlendShape loadBlendShape(
    const filesystem::path& filename,
    int expectedShapes = -1,
    int expectedVertices = -1);

/// Loads a blend shape from a stream, including base shape and shape vectors.
///
/// @param data Input stream containing blend shape data
/// @param expectedShapes Limits number of shape vectors loaded if > 0, otherwise loads all
/// @param expectedVertices Limits number of vertices loaded if > 0, otherwise loads all
/// @return BlendShape object with loaded data
BlendShape loadBlendShape(std::istream& data, int expectedShapes = -1, int expectedVertices = -1);

/// Saves a blend shape to a file.
///
/// @param filename Output file path for the blend shape data
/// @param blendShape BlendShape object to save
void saveBlendShape(const filesystem::path& filename, const BlendShape& blendShape);

/// Saves a blend shape to a stream.
///
/// @param os Output stream to write blend shape data
/// @param blendShape BlendShape object to save
void saveBlendShape(std::ostream& os, const BlendShape& blendShape);

} // namespace momentum
