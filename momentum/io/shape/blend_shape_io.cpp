/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/shape/blend_shape_io.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_base.h"
#include "momentum/common/exception.h"
#include "momentum/common/log.h"

#include <gsl/gsl>

#include <fstream>
#include <istream>
#include <vector>

namespace momentum {

namespace {

std::istringstream readOrThrow(const filesystem::path& filepath) {
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  MT_THROW_IF(!file, "Failed to open the file.");

  std::size_t size = static_cast<std::size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  std::string buffer(size, '\0');
  MT_THROW_IF(!file.read(buffer.data(), size), "Failed to read the file.");

  return std::istringstream(std::move(buffer));
}

void readShapeVectors(
    std::istream& data,
    MatrixXf& shapeVectors,
    uint64_t numRows,
    uint64_t numCols,
    int expectedShapes,
    int expectedVertices) {
  // load shapeVectors
  shapeVectors.resize(numRows, numCols);
  data.read((char*)shapeVectors.data(), sizeof(float) * numRows * numCols);

  // resize to few shapes
  if (expectedShapes > 0) {
    shapeVectors.conservativeResize(
        shapeVectors.rows(),
        std::min(gsl::narrow_cast<Eigen::Index>(expectedShapes), shapeVectors.cols()));
  }
  // resize vertices
  if (expectedVertices > 0) {
    shapeVectors.conservativeResize(
        std::min(shapeVectors.rows(), gsl::narrow_cast<Eigen::Index>(expectedVertices * 3)),
        shapeVectors.cols());
  }
}

} // namespace

BlendShapeBase
loadBlendShapeBase(const filesystem::path& filename, int expectedShapes, int expectedVertices) {
  std::istringstream istr = readOrThrow(filename);
  return loadBlendShapeBase(istr, expectedShapes, expectedVertices);
}

BlendShapeBase loadBlendShapeBase(std::istream& data, int expectedShapes, int expectedVertices) {
  MatrixXf shapeVectors;

  // read dimensions
  uint64_t numRows = 0;
  uint64_t numCols = 0;
  data.read((char*)&numRows, sizeof(numRows));
  data.read((char*)&numCols, sizeof(numCols));

  // read shape vectors
  readShapeVectors(data, shapeVectors, numRows, numCols, expectedShapes, expectedVertices);

  BlendShapeBase res(shapeVectors.rows() / 3, shapeVectors.cols());
  res.setShapeVectors(shapeVectors);
  return res;
}

BlendShape
loadBlendShape(const filesystem::path& filename, int expectedShapes, int expectedVertices) {
  std::istringstream istr = readOrThrow(filename);
  return loadBlendShape(istr, expectedShapes, expectedVertices);
}

BlendShape loadBlendShape(std::istream& data, int expectedShapes, int expectedVertices) {
  std::vector<Vector3f> baseShape;
  MatrixXf shapeVectors;

  // read dimensions
  uint64_t numRows = 0;
  uint64_t numCols = 0;
  data.read((char*)&numRows, sizeof(numRows));
  data.read((char*)&numCols, sizeof(numCols));

  // read mean shape
  baseShape.resize(numRows / 3);
  data.read((char*)baseShape.data(), sizeof(float) * numRows);

  if (expectedVertices > 0) {
    baseShape.resize(std::min(baseShape.size(), gsl::narrow<size_t>(expectedVertices)));
  }

  // read shape vectors
  readShapeVectors(data, shapeVectors, numRows, numCols, expectedShapes, expectedVertices);

  BlendShape res(baseShape, shapeVectors.cols());
  res.setShapeVectors(shapeVectors);
  return res;
}

void saveBlendShapeBase(const filesystem::path& filename, const BlendShapeBase& blendShapeBase) {
  std::ofstream data(filename, std::ios::out | std::ios::binary);
  if (!data.is_open()) {
    MT_LOGW("Failed to open file for writing: {}", filename.string());
    return;
  }
  saveBlendShapeBase(data, blendShapeBase);
}

void saveBlendShape(const filesystem::path& filename, const BlendShape& blendShape) {
  std::ofstream data(filename, std::ios::out | std::ios::binary);
  if (!data.is_open()) {
    MT_LOGW("Failed to open file for writing: {}", filename.string());
    return;
  }
  saveBlendShape(data, blendShape);
}

void saveBlendShapeBase(std::ostream& os, const BlendShapeBase& blendShapeBase) {
  // write dimensions
  const uint64_t numRows = blendShapeBase.getShapeVectors().rows();
  const uint64_t numCols = blendShapeBase.getShapeVectors().cols();

  os.write((char const*)&numRows, sizeof(numRows));
  os.write((char const*)&numCols, sizeof(numCols));

  // write coefficient matrix
  os.write((char const*)blendShapeBase.getShapeVectors().data(), sizeof(float) * numRows * numCols);
}

void saveBlendShape(std::ostream& os, const BlendShape& blendShape) {
  // write dimensions
  const uint64_t numRows = blendShape.getShapeVectors().rows();
  const uint64_t numCols = blendShape.getShapeVectors().cols();
  MT_CHECK(
      numRows == blendShape.getBaseShape().size() * 3,
      "{}, {}",
      numRows,
      blendShape.getBaseShape().size() * 3);

  os.write((char const*)&numRows, sizeof(numRows));
  os.write((char const*)&numCols, sizeof(numCols));

  // write mean shape
  os.write((char const*)blendShape.getBaseShape().data(), sizeof(float) * numRows);

  // write coefficient matrix
  os.write((char const*)blendShape.getShapeVectors().data(), sizeof(float) * numRows * numCols);
}

} // namespace momentum
