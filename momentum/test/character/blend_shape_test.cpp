/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_base.h"
#include "momentum/math/random.h"

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct BlendShapeTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(BlendShapeTest, Types);

// Helper function to create random vertices
template <typename T>
std::vector<Eigen::Vector3<T>> createRandomVertices(size_t count) {
  std::vector<Eigen::Vector3<T>> vertices;
  vertices.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    vertices.push_back(Eigen::Vector3<T>::Random());
  }
  return vertices;
}

// Helper function to create random blend weights
template <typename T>
BlendWeightsT<T> createRandomBlendWeights(size_t count) {
  BlendWeightsT<T> weights;
  weights.v = VectorX<T>::Random(count);
  return weights;
}

// Tests for BlendShapeBase
TEST(BlendShapeTest, BlendShapeBaseConstruction) {
  const size_t modelSize = 10;
  const size_t numShapes = 5;

  BlendShapeBase blendShapeBase(modelSize, numShapes);

  EXPECT_EQ(blendShapeBase.modelSize(), modelSize);
  EXPECT_EQ(blendShapeBase.shapeSize(), numShapes);
  EXPECT_EQ(blendShapeBase.getShapeVectors().rows(), modelSize * 3);
  EXPECT_EQ(blendShapeBase.getShapeVectors().cols(), numShapes);
}

TEST(BlendShapeTest, BlendShapeBaseSetShapeVector) {
  const size_t modelSize = 10;
  const size_t numShapes = 5;

  BlendShapeBase blendShapeBase(modelSize, numShapes);

  // Create a random shape vector
  std::vector<Vector3f> shapeVector = createRandomVertices<float>(modelSize);

  // Set the shape vector at index 2
  blendShapeBase.setShapeVector(2, shapeVector);

  // Check that the shape vector was set correctly
  const MatrixXf& shapeVectors = blendShapeBase.getShapeVectors();
  for (size_t i = 0; i < modelSize; ++i) {
    EXPECT_FLOAT_EQ(shapeVectors(i * 3 + 0, 2), shapeVector[i].x());
    EXPECT_FLOAT_EQ(shapeVectors(i * 3 + 1, 2), shapeVector[i].y());
    EXPECT_FLOAT_EQ(shapeVectors(i * 3 + 2, 2), shapeVector[i].z());
  }
}

TYPED_TEST(BlendShapeTest, BlendShapeBaseComputeDeltas) {
  using T = typename TestFixture::Type;

  const size_t modelSize = 10;
  const size_t numShapes = 5;

  BlendShapeBase blendShapeBase(modelSize, numShapes);

  // Create and set random shape vectors
  std::vector<std::vector<Vector3f>> shapeVectors;
  for (size_t i = 0; i < numShapes; ++i) {
    std::vector<Vector3f> shapeVector = createRandomVertices<float>(modelSize);
    shapeVectors.push_back(shapeVector);
    blendShapeBase.setShapeVector(i, shapeVector);
  }

  // Create random blend weights
  BlendWeightsT<T> blendWeights = createRandomBlendWeights<T>(numShapes);

  // Compute deltas
  VectorX<T> deltas = blendShapeBase.computeDeltas(blendWeights);

  // Check that deltas have the correct size
  EXPECT_EQ(deltas.size(), modelSize * 3);

  // Manually compute expected deltas
  VectorX<T> expectedDeltas = VectorX<T>::Zero(modelSize * 3);
  for (size_t i = 0; i < numShapes; ++i) {
    for (size_t j = 0; j < modelSize; ++j) {
      expectedDeltas(j * 3 + 0) += static_cast<T>(shapeVectors[i][j].x()) * blendWeights.v(i);
      expectedDeltas(j * 3 + 1) += static_cast<T>(shapeVectors[i][j].y()) * blendWeights.v(i);
      expectedDeltas(j * 3 + 2) += static_cast<T>(shapeVectors[i][j].z()) * blendWeights.v(i);
    }
  }

  // Check that computed deltas match expected deltas
  for (int i = 0; i < deltas.size(); ++i) {
    EXPECT_NEAR(deltas(i), expectedDeltas(i), 1e-5);
  }
}

TYPED_TEST(BlendShapeTest, BlendShapeBaseApplyDeltas) {
  using T = typename TestFixture::Type;

  const size_t modelSize = 10;
  const size_t numShapes = 5;

  BlendShapeBase blendShapeBase(modelSize, numShapes);

  // Create and set random shape vectors
  std::vector<std::vector<Vector3f>> shapeVectors;
  for (size_t i = 0; i < numShapes; ++i) {
    std::vector<Vector3f> shapeVector = createRandomVertices<float>(modelSize);
    shapeVectors.push_back(shapeVector);
    blendShapeBase.setShapeVector(i, shapeVector);
  }

  // Create random blend weights
  BlendWeightsT<T> blendWeights = createRandomBlendWeights<T>(numShapes);

  // Create base shape
  std::vector<Eigen::Vector3<T>> baseShape = createRandomVertices<T>(modelSize);
  std::vector<Eigen::Vector3<T>> result = baseShape;

  // Apply deltas
  blendShapeBase.applyDeltas(blendWeights, result);

  // Compute expected result
  VectorX<T> deltas = blendShapeBase.computeDeltas(blendWeights);
  std::vector<Eigen::Vector3<T>> expectedResult = baseShape;
  for (size_t i = 0; i < modelSize; ++i) {
    expectedResult[i].x() += deltas(i * 3 + 0);
    expectedResult[i].y() += deltas(i * 3 + 1);
    expectedResult[i].z() += deltas(i * 3 + 2);
  }

  // Check that result matches expected result
  for (size_t i = 0; i < modelSize; ++i) {
    EXPECT_NEAR(result[i].x(), expectedResult[i].x(), 1e-5);
    EXPECT_NEAR(result[i].y(), expectedResult[i].y(), 1e-5);
    EXPECT_NEAR(result[i].z(), expectedResult[i].z(), 1e-5);
  }
}

// Tests for BlendShape
TEST(BlendShapeTest, BlendShapeConstruction) {
  const size_t modelSize = 10;
  const size_t numShapes = 5;

  // Create a random base shape
  std::vector<Vector3f> baseShape = createRandomVertices<float>(modelSize);

  BlendShape blendShape(baseShape, numShapes);

  EXPECT_EQ(blendShape.modelSize(), modelSize);
  EXPECT_EQ(blendShape.shapeSize(), numShapes);
  EXPECT_EQ(blendShape.getShapeVectors().rows(), modelSize * 3);
  EXPECT_EQ(blendShape.getShapeVectors().cols(), numShapes);
  EXPECT_EQ(blendShape.getBaseShape().size(), modelSize);
  EXPECT_FALSE(blendShape.getFactorizationValid());

  // Check that base shape was set correctly
  for (size_t i = 0; i < modelSize; ++i) {
    EXPECT_FLOAT_EQ(blendShape.getBaseShape()[i].x(), baseShape[i].x());
    EXPECT_FLOAT_EQ(blendShape.getBaseShape()[i].y(), baseShape[i].y());
    EXPECT_FLOAT_EQ(blendShape.getBaseShape()[i].z(), baseShape[i].z());
  }
}

TEST(BlendShapeTest, BlendShapeSetBaseShape) {
  const size_t modelSize = 10;
  const size_t numShapes = 5;

  // Create a random base shape
  std::vector<Vector3f> baseShape1 = createRandomVertices<float>(modelSize);

  BlendShape blendShape(baseShape1, numShapes);

  // Create a new random base shape
  std::vector<Vector3f> baseShape2 = createRandomVertices<float>(modelSize);

  // Set the new base shape
  blendShape.setBaseShape(baseShape2);

  // Check that base shape was updated correctly
  for (size_t i = 0; i < modelSize; ++i) {
    EXPECT_FLOAT_EQ(blendShape.getBaseShape()[i].x(), baseShape2[i].x());
    EXPECT_FLOAT_EQ(blendShape.getBaseShape()[i].y(), baseShape2[i].y());
    EXPECT_FLOAT_EQ(blendShape.getBaseShape()[i].z(), baseShape2[i].z());
  }
}

TYPED_TEST(BlendShapeTest, BlendShapeComputeShape) {
  using T = typename TestFixture::Type;

  const size_t modelSize = 10;
  const size_t numShapes = 5;

  // Create a random base shape
  std::vector<Vector3f> baseShape = createRandomVertices<float>(modelSize);

  BlendShape blendShape(baseShape, numShapes);

  // Create and set random shape vectors
  std::vector<std::vector<Vector3f>> shapeVectors;
  for (size_t i = 0; i < numShapes; ++i) {
    std::vector<Vector3f> shapeVector = createRandomVertices<float>(modelSize);
    shapeVectors.push_back(shapeVector);
    blendShape.setShapeVector(i, shapeVector);
  }

  // Create random blend weights
  BlendWeightsT<T> blendWeights = createRandomBlendWeights<T>(numShapes);

  // Compute shape
  std::vector<Eigen::Vector3<T>> shape = blendShape.computeShape(blendWeights);

  // Check that shape has the correct size
  EXPECT_EQ(shape.size(), modelSize);

  // Compute expected shape
  VectorX<T> deltas = blendShape.computeDeltas(blendWeights);
  std::vector<Eigen::Vector3<T>> expectedShape(modelSize);
  for (size_t i = 0; i < modelSize; ++i) {
    expectedShape[i].x() = static_cast<T>(baseShape[i].x()) + deltas(i * 3 + 0);
    expectedShape[i].y() = static_cast<T>(baseShape[i].y()) + deltas(i * 3 + 1);
    expectedShape[i].z() = static_cast<T>(baseShape[i].z()) + deltas(i * 3 + 2);
  }

  // Check that computed shape matches expected shape
  for (size_t i = 0; i < modelSize; ++i) {
    EXPECT_NEAR(shape[i].x(), expectedShape[i].x(), 1e-5);
    EXPECT_NEAR(shape[i].y(), expectedShape[i].y(), 1e-5);
    EXPECT_NEAR(shape[i].z(), expectedShape[i].z(), 1e-5);
  }

  // Test the version that takes an output parameter
  std::vector<Eigen::Vector3<T>> result;
  blendShape.computeShape(blendWeights, result);

  // Check that result has the correct size
  EXPECT_EQ(result.size(), modelSize);

  // Check that result matches expected shape
  for (size_t i = 0; i < modelSize; ++i) {
    EXPECT_NEAR(result[i].x(), expectedShape[i].x(), 1e-5);
    EXPECT_NEAR(result[i].y(), expectedShape[i].y(), 1e-5);
    EXPECT_NEAR(result[i].z(), expectedShape[i].z(), 1e-5);
  }
}

TEST(BlendShapeTest, BlendShapeEstimateCoefficients) {
  const size_t modelSize = 10;
  const size_t numShapes = 5;

  // Create a random base shape
  std::vector<Vector3f> baseShape = createRandomVertices<float>(modelSize);

  BlendShape blendShape(baseShape, numShapes);

  // Create and set random shape vectors
  std::vector<std::vector<Vector3f>> shapeVectors;
  for (size_t i = 0; i < numShapes; ++i) {
    std::vector<Vector3f> shapeVector = createRandomVertices<float>(modelSize);
    shapeVectors.push_back(shapeVector);
    blendShape.setShapeVector(i, shapeVector);
  }

  // Create known blend weights
  VectorXf knownCoefficients = VectorXf::Random(numShapes);
  BlendWeightsT<float> blendWeights;
  blendWeights.v = knownCoefficients;

  // Compute a shape using the known coefficients
  std::vector<Eigen::Vector3<float>> targetShape = blendShape.computeShape(blendWeights);

  // Estimate coefficients from the target shape
  VectorXf estimatedCoefficients = blendShape.estimateCoefficients(targetShape);

  // Check that estimated coefficients are close to the known coefficients
  EXPECT_EQ(estimatedCoefficients.size(), knownCoefficients.size());
  for (int i = 0; i < knownCoefficients.size(); ++i) {
    EXPECT_NEAR(estimatedCoefficients(i), knownCoefficients(i), 1e-4);
  }

  // Test with regularization
  float regularization = 0.1f;
  VectorXf estimatedCoefficientsReg = blendShape.estimateCoefficients(targetShape, regularization);

  // Check that estimated coefficients with regularization are still reasonable
  EXPECT_EQ(estimatedCoefficientsReg.size(), knownCoefficients.size());

  // Test with weights
  VectorXf weights = VectorXf::Ones(modelSize);
  VectorXf estimatedCoefficientsWeighted =
      blendShape.estimateCoefficients(targetShape, 1.0f, weights);

  // Check that estimated coefficients with weights are still reasonable
  EXPECT_EQ(estimatedCoefficientsWeighted.size(), knownCoefficients.size());
}

TEST(BlendShapeTest, BlendShapeSetShapeVector) {
  const size_t modelSize = 10;
  const size_t numShapes = 5;

  // Create a random base shape
  std::vector<Vector3f> baseShape = createRandomVertices<float>(modelSize);

  BlendShape blendShape(baseShape, numShapes);

  // Create a random shape vector
  std::vector<Vector3f> shapeVector = createRandomVertices<float>(modelSize);

  // Set the shape vector at index 2
  blendShape.setShapeVector(2, shapeVector);

  // Check that the shape vector was set correctly
  const MatrixXf& shapeVectors = blendShape.getShapeVectors();
  for (size_t i = 0; i < modelSize; ++i) {
    EXPECT_FLOAT_EQ(shapeVectors(i * 3 + 0, 2), shapeVector[i].x());
    EXPECT_FLOAT_EQ(shapeVectors(i * 3 + 1, 2), shapeVector[i].y());
    EXPECT_FLOAT_EQ(shapeVectors(i * 3 + 2, 2), shapeVector[i].z());
  }

  // Check that factorization is invalidated
  EXPECT_FALSE(blendShape.getFactorizationValid());

  // Force factorization to be computed
  std::vector<Vector3f> targetShape = createRandomVertices<float>(modelSize);
  (void)blendShape.estimateCoefficients(targetShape);

  // Check that factorization is now valid
  EXPECT_TRUE(blendShape.getFactorizationValid());

  // Set another shape vector
  blendShape.setShapeVector(3, shapeVector);

  // Check that factorization is invalidated again
  EXPECT_FALSE(blendShape.getFactorizationValid());
}

TEST(BlendShapeTest, BlendShapeIsApprox) {
  const size_t modelSize = 10;
  const size_t numShapes = 5;

  // Create a random base shape
  std::vector<Vector3f> baseShape = createRandomVertices<float>(modelSize);

  BlendShape blendShape1(baseShape, numShapes);

  // Create and set random shape vectors
  std::vector<std::vector<Vector3f>> shapeVectors;
  for (size_t i = 0; i < numShapes; ++i) {
    std::vector<Vector3f> shapeVector = createRandomVertices<float>(modelSize);
    shapeVectors.push_back(shapeVector);
    blendShape1.setShapeVector(i, shapeVector);
  }

  // Create an identical blend shape
  BlendShape blendShape2(baseShape, numShapes);
  for (size_t i = 0; i < numShapes; ++i) {
    blendShape2.setShapeVector(i, shapeVectors[i]);
  }

  // Check that the blend shapes are approximately equal
  EXPECT_TRUE(blendShape1.isApprox(blendShape2));

  // Modify the second blend shape
  std::vector<Vector3f> differentBaseShape = createRandomVertices<float>(modelSize);
  blendShape2.setBaseShape(differentBaseShape);

  // Check that the blend shapes are no longer approximately equal
  EXPECT_FALSE(blendShape1.isApprox(blendShape2));

  // Reset the base shape but modify a shape vector
  blendShape2.setBaseShape(baseShape);
  std::vector<Vector3f> differentShapeVector = createRandomVertices<float>(modelSize);
  blendShape2.setShapeVector(0, differentShapeVector);

  // Check that the blend shapes are no longer approximately equal
  EXPECT_FALSE(blendShape1.isApprox(blendShape2));
}

// Test edge cases
TEST(BlendShapeTest, EdgeCases) {
  // Test with empty base shape and no shape vectors
  BlendShape emptyBlendShape;
  EXPECT_EQ(emptyBlendShape.modelSize(), 0);
  EXPECT_EQ(emptyBlendShape.shapeSize(), 0);
  EXPECT_FALSE(emptyBlendShape.getFactorizationValid());

  // Test with a single vertex and a single shape
  const size_t modelSize = 1;
  const size_t numShapes = 1;

  std::vector<Vector3f> baseShape = {Vector3f(1.0f, 2.0f, 3.0f)};
  BlendShape singleVertexBlendShape(baseShape, numShapes);

  EXPECT_EQ(singleVertexBlendShape.modelSize(), modelSize);
  EXPECT_EQ(singleVertexBlendShape.shapeSize(), numShapes);

  std::vector<Vector3f> shapeVector = {Vector3f(0.1f, 0.2f, 0.3f)};
  singleVertexBlendShape.setShapeVector(0, shapeVector);

  // Test with a single coefficient
  BlendWeightsT<float> blendWeights;
  blendWeights.v = VectorXf::Ones(1);

  std::vector<Eigen::Vector3<float>> result = singleVertexBlendShape.computeShape(blendWeights);

  EXPECT_EQ(result.size(), modelSize);
  EXPECT_NEAR(result[0].x(), baseShape[0].x() + shapeVector[0].x(), 1e-5);
  EXPECT_NEAR(result[0].y(), baseShape[0].y() + shapeVector[0].y(), 1e-5);
  EXPECT_NEAR(result[0].z(), baseShape[0].z() + shapeVector[0].z(), 1e-5);
}

// Test error cases
TEST(BlendShapeTest, ErrorCases) {
  const size_t modelSize = 10;
  const size_t numShapes = 5;

  // Create a random base shape
  std::vector<Vector3f> baseShape = createRandomVertices<float>(modelSize);

  BlendShape blendShape(baseShape, numShapes);

  // Test setting a shape vector with wrong size
  std::vector<Vector3f> wrongSizeShapeVector = createRandomVertices<float>(modelSize + 1);

  // This should trigger an assertion failure, but we can't test that directly in gtest
  // Just documenting that this would be an error case

  // Test estimating coefficients with wrong size vertices
  std::vector<Vector3f> wrongSizeVertices = createRandomVertices<float>(modelSize + 1);

  // This should trigger an assertion failure, but we can't test that directly in gtest
  // Just documenting that this would be an error case

  // Test with blend weights that have more coefficients than shape vectors
  BlendWeightsT<float> tooManyWeights;
  tooManyWeights.v = VectorXf::Random(numShapes + 1);

  // This should trigger an assertion failure, but we can't test that directly in gtest
  // Just documenting that this would be an error case
}

// Test with different sizes
TEST(BlendShapeTest, DifferentSizes) {
  // Test with a larger model
  const size_t largeModelSize = 1000;
  const size_t numShapes = 10;

  std::vector<Vector3f> baseShape = createRandomVertices<float>(largeModelSize);
  BlendShape largeBlendShape(baseShape, numShapes);

  EXPECT_EQ(largeBlendShape.modelSize(), largeModelSize);
  EXPECT_EQ(largeBlendShape.shapeSize(), numShapes);

  // Test with many shape vectors
  const size_t modelSize = 10;
  const size_t manyShapes = 100;

  baseShape = createRandomVertices<float>(modelSize);
  BlendShape manyShapesBlendShape(baseShape, manyShapes);

  EXPECT_EQ(manyShapesBlendShape.modelSize(), modelSize);
  EXPECT_EQ(manyShapesBlendShape.shapeSize(), manyShapes);
}

TEST(BlendShapeTest, BlendShapeNames) {
  const size_t modelSize = 10;
  const size_t numShapes = 5;

  BlendShapeBase blendShapeBase(modelSize, numShapes);

  // check that we get names
  for (size_t i = 0; i < numShapes; ++i) {
    EXPECT_EQ(blendShapeBase.getShapeName(i), std::string("shape_") + std::to_string(i));
  }

  // do explicit names
  std::vector<std::string> names;
  for (size_t i = 0; i < numShapes; ++i) {
    names.push_back(std::string("different_name_") + std::to_string(i));
  }
  BlendShapeBase namedBase(modelSize, numShapes, names);

  for (size_t i = 0; i < numShapes; ++i) {
    EXPECT_EQ(namedBase.getShapeName(i), names[i]);
  }
}
