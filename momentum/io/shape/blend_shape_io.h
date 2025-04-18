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

BlendShapeBase loadBlendShapeBase(
    const filesystem::path& filename,
    int expectedShapes = -1,
    int expectedVertices = -1);

BlendShapeBase
loadBlendShapeBase(std::istream& data, int expectedShapes = -1, int expectedVertices = -1);

/// Loads a blend shape from a given filepath.
///
/// This function reads blend shape data from the provided file, including the mean shape and shape
/// vectors. If `expectedShapes` or `expectedVertices` are provided, the function will attempt to
/// limit the number of shape vectors or vertices read accordingly.
///
/// @param filename The file path containing the blend shape data.
/// @param[in] expectedShapes The expected number of shape vectors to read. If this value is greater
/// than 0, it will limit the number of shape vectors read. Defaults to 0, meaning all shape vectors
/// will be read.
/// @param[in] expectedVertices The expected number of vertices to read. If this value is greater
/// than 0, it will limit the number of vertices read. Defaults to 0, meaning all vertices will be
/// read.
/// @return A BlendShape object containing the loaded data.
///
/// @note This function only supports parsing from a local file. If you want to parse from a
/// non-local path, you may need to parse it using your favorite resource retriever into a stream
/// buffer and use the buffer version of this function.
BlendShape loadBlendShape(
    const filesystem::path& filename,
    int expectedShapes = -1,
    int expectedVertices = -1);

BlendShape loadBlendShape(std::istream& is, int expectedShapes = -1, int expectedVertices = -1);

void ReadShapeVectors(
    std::istream& data,
    MatrixXf& shapeVectors,
    uint64_t numRows,
    uint64_t numCols,
    int expectedShapes,
    int expectedVertices);

void saveBlendShape(const filesystem::path& filename, const BlendShape& blendShape);

void saveBlendShape(std::ostream& os, const BlendShape& blendShape);

} // namespace momentum
