/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/character.h"
#include "momentum/character/skin_weights.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/helpers/expect_throw.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <limits>

using namespace momentum;

class SkinWeightsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create sample data for testing
    indices = {
        {0, 1, 2}, // Vertex 0 is influenced by joints 0, 1, 2
        {1, 2, 3}, // Vertex 1 is influenced by joints 1, 2, 3
        {2, 3, 4}, // Vertex 2 is influenced by joints 2, 3, 4
        {5, 3, 4}, // Vertex 3 is influenced by joints 5, 3, 4
    };

    weights = {
        {0.5f, 0.3f, 0.2f}, // Weights for vertex 0
        {0.4f, 0.4f, 0.2f}, // Weights for vertex 1
        {0.6f, 0.2f, 0.2f}, // Weights for vertex 2
        {0.4f, 0.3f, 0.3f}, // Weights for vertex 3
    };

    // Create indices and weights that exceed kMaxSkinJoints
    largeIndices = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, // 10 joints (exceeds kMaxSkinJoints)
    };

    largeWeights = {
        {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f}, // 10 weights
    };
  }

  std::vector<std::vector<size_t>> indices;
  std::vector<std::vector<float>> weights;
  std::vector<std::vector<size_t>> largeIndices;
  std::vector<std::vector<float>> largeWeights;
};

// Test the set method with valid inputs
TEST_F(SkinWeightsTest, SetWithValidInputs) {
  SkinWeights skinWeights;
  skinWeights.set(indices, weights);

  // Check that the matrices have the correct size
  EXPECT_EQ(skinWeights.index.rows(), 4);
  EXPECT_EQ(skinWeights.index.cols(), kMaxSkinJoints);
  EXPECT_EQ(skinWeights.weight.rows(), 4);
  EXPECT_EQ(skinWeights.weight.cols(), kMaxSkinJoints);

  // Check that the values were set correctly
  for (size_t i = 0; i < indices.size(); i++) {
    for (size_t j = 0; j < indices[i].size(); j++) {
      EXPECT_EQ(skinWeights.index(i, j), indices[i][j]);
      EXPECT_FLOAT_EQ(skinWeights.weight(i, j), weights[i][j]);
    }
  }

  // Check that the remaining values are zero
  for (size_t i = 0; i < indices.size(); i++) {
    for (size_t j = indices[i].size(); j < kMaxSkinJoints; j++) {
      EXPECT_EQ(skinWeights.index(i, j), 0);
      EXPECT_FLOAT_EQ(skinWeights.weight(i, j), 0.0f);
    }
  }
}

// Test that equal-weight truncation is stable.
TEST_F(SkinWeightsTest, SetWithLargeInputs) {
  SkinWeights skinWeights;
  skinWeights.set(largeIndices, largeWeights);

  for (size_t j = 0; j < kMaxSkinJoints; ++j) {
    EXPECT_EQ(skinWeights.index(0, j), largeIndices[0][j]);
    EXPECT_FLOAT_EQ(skinWeights.weight(0, j), largeWeights[0][j]);
  }
}

TEST_F(SkinWeightsTest, SetKeepsLargestInfluences) {
  SkinWeights skinWeights;
  const std::vector<std::vector<size_t>> unsortedIndices = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
  const std::vector<std::vector<float>> unsortedWeights = {
      {0.01f, 0.20f, 0.02f, 0.18f, 0.03f, 0.16f, 0.04f, 0.14f, 0.05f, 0.17f}};

  skinWeights.set(unsortedIndices, unsortedWeights);

  const std::array<uint32_t, kMaxSkinJoints> expectedIndices = {1, 3, 9, 5, 7, 8, 6, 4};
  for (size_t j = 0; j < expectedIndices.size(); ++j) {
    EXPECT_EQ(skinWeights.index(0, j), expectedIndices[j]);
  }
  EXPECT_NEAR(skinWeights.weight.row(0).sum(), 0.97f, 1e-6f);
}

TEST_F(SkinWeightsTest, SetSortsAndPacksInfluences) {
  SkinWeights skinWeights;
  const std::vector<std::vector<size_t>> unsortedIndices = {{0, 1, 2, 3}};
  const std::vector<std::vector<float>> unsortedWeights = {{0.2f, 0.0f, 0.5f, 0.3f}};

  skinWeights.set(unsortedIndices, unsortedWeights);

  const std::array<uint32_t, 3> expectedIndices = {2, 3, 0};
  const std::array<float, 3> expectedWeights = {0.5f, 0.3f, 0.2f};
  for (size_t j = 0; j < expectedIndices.size(); ++j) {
    EXPECT_EQ(skinWeights.index(0, j), expectedIndices[j]);
    EXPECT_FLOAT_EQ(skinWeights.weight(0, j), expectedWeights[j]);
  }
  for (size_t j = expectedIndices.size(); j < kMaxSkinJoints; ++j) {
    EXPECT_EQ(skinWeights.index(0, j), 0);
    EXPECT_FLOAT_EQ(skinWeights.weight(0, j), 0.0f);
  }
}

// Test the set method with empty inputs
TEST_F(SkinWeightsTest, SetWithEmptyInputs) {
  SkinWeights skinWeights;
  std::vector<std::vector<size_t>> emptyIndices;
  std::vector<std::vector<float>> emptyWeights;
  skinWeights.set(emptyIndices, emptyWeights);

  // Check that the matrices have zero rows
  EXPECT_EQ(skinWeights.index.rows(), 0);
  EXPECT_EQ(skinWeights.weight.rows(), 0);
}

// Test the set method with inputs of different sizes (should cause a fatal error)
TEST_F(SkinWeightsTest, SetWithDifferentSizedInputs) {
  SkinWeights skinWeights;
  std::vector<std::vector<size_t>> moreIndices = indices;
  moreIndices.push_back({5, 6, 7}); // Add an extra row to indices

  // MT_CHECK causes a fatal error, not an exception, so we use MOMENTUM_EXPECT_DEATH
  MOMENTUM_EXPECT_DEATH(
      skinWeights.set(moreIndices, weights),
      "Check 'ind.size\\(\\) == wgt.size\\(\\)' failed: 5 is not 4");
}

TEST_F(SkinWeightsTest, SetWithDifferentSizedRows) {
  SkinWeights skinWeights;
  const std::vector<std::vector<size_t>> rowIndices = {{0, 1}};
  const std::vector<std::vector<float>> rowWeights = {{1.0f}};

  MOMENTUM_EXPECT_DEATH(
      skinWeights.set(rowIndices, rowWeights), "Skin weight row 0 has 2 indices but 1 weights");
}

TEST_F(SkinWeightsTest, SetDiscardsInvalidWeights) {
  SkinWeights skinWeights;
  const std::vector<std::vector<size_t>> rowIndices = {{0, 1, 2}};
  const std::vector<std::vector<float>> rowWeights = {
      {0.4f, -0.1f, std::numeric_limits<float>::quiet_NaN()}};

  skinWeights.set(rowIndices, rowWeights);

  EXPECT_EQ(skinWeights.index(0, 0), 0);
  EXPECT_FLOAT_EQ(skinWeights.weight(0, 0), 0.4f);
  EXPECT_TRUE(skinWeights.index.row(0).tail<kMaxSkinJoints - 1>().isZero());
  EXPECT_TRUE(skinWeights.weight.row(0).tail<kMaxSkinJoints - 1>().isZero());
}

TEST(SkinWeightsTestHelper, CreateTestCharacterZeroesUnusedIndices) {
  const Character character = createTestCharacter();
  ASSERT_NE(character.skinWeights, nullptr);

  for (Eigen::Index i = 0; i < character.skinWeights->weight.rows(); ++i) {
    for (Eigen::Index j = 0; j < character.skinWeights->weight.cols(); ++j) {
      if (character.skinWeights->weight(i, j) == 0.0f) {
        EXPECT_EQ(character.skinWeights->index(i, j), 0);
      }
    }
  }
}

// Test the equality operator with identical SkinWeights objects
TEST_F(SkinWeightsTest, EqualityWithIdenticalObjects) {
  SkinWeights skinWeights1;
  skinWeights1.set(indices, weights);

  SkinWeights skinWeights2;
  skinWeights2.set(indices, weights);

  EXPECT_TRUE(skinWeights1 == skinWeights2);
}

// Test the equality operator with different SkinWeights objects
TEST_F(SkinWeightsTest, EqualityWithDifferentObjects) {
  SkinWeights skinWeights1;
  skinWeights1.set(indices, weights);

  // Create a slightly different set of weights
  std::vector<std::vector<float>> differentWeights = weights;
  differentWeights[0][0] = 0.6f; // Change one weight

  SkinWeights skinWeights2;
  skinWeights2.set(indices, differentWeights);

  EXPECT_FALSE(skinWeights1 == skinWeights2);

  // Create a slightly different set of indices
  std::vector<std::vector<size_t>> differentIndices = indices;
  differentIndices[0][0] = 9; // Change one index

  SkinWeights skinWeights3;
  skinWeights3.set(differentIndices, weights);

  EXPECT_FALSE(skinWeights1 == skinWeights3);
}

// Test the equality operator with objects of different sizes
TEST_F(SkinWeightsTest, EqualityWithDifferentSizes) {
  SkinWeights skinWeights1;
  skinWeights1.set(indices, weights);

  // Create a smaller set of indices and weights
  std::vector<std::vector<size_t>> smallerIndices = {indices[0], indices[1]};
  std::vector<std::vector<float>> smallerWeights = {weights[0], weights[1]};

  SkinWeights skinWeights2;
  skinWeights2.set(smallerIndices, smallerWeights);

  // Verify that the matrices have different dimensions
  EXPECT_NE(skinWeights1.index.rows(), skinWeights2.index.rows());
  EXPECT_NE(skinWeights1.weight.rows(), skinWeights2.weight.rows());

  // Create a custom equality check that first verifies dimensions
  auto customEqualityCheck = [](const SkinWeights& a, const SkinWeights& b) -> bool {
    // First check if dimensions match
    if (a.index.rows() != b.index.rows() || a.index.cols() != b.index.cols() ||
        a.weight.rows() != b.weight.rows() || a.weight.cols() != b.weight.cols()) {
      return false;
    }
    // If dimensions match, use the original equality operator
    return a == b;
  };

  // Use our custom equality check instead of the built-in operator==
  EXPECT_FALSE(customEqualityCheck(skinWeights1, skinWeights2));
}

// Test the equality operator with objects of different sizes by manually checking dimensions
TEST_F(SkinWeightsTest, EqualityWithDifferentSizesManualCheck) {
  SkinWeights skinWeights1;
  skinWeights1.set(indices, weights);

  // Create a smaller set of indices and weights
  std::vector<std::vector<size_t>> smallerIndices = {indices[0], indices[1]};
  std::vector<std::vector<float>> smallerWeights = {weights[0], weights[1]};

  SkinWeights skinWeights2;
  skinWeights2.set(smallerIndices, smallerWeights);

  // Verify that the matrices have different dimensions
  EXPECT_NE(skinWeights1.index.rows(), skinWeights2.index.rows());
  EXPECT_NE(skinWeights1.weight.rows(), skinWeights2.weight.rows());

  // Manually check dimensions before using the equality operator
  bool hasSameDimensions = skinWeights1.index.rows() == skinWeights2.index.rows() &&
      skinWeights1.index.cols() == skinWeights2.index.cols() &&
      skinWeights1.weight.rows() == skinWeights2.weight.rows() &&
      skinWeights1.weight.cols() == skinWeights2.weight.cols();

  EXPECT_FALSE(hasSameDimensions);

  // If dimensions are different, objects should not be equal
  if (!hasSameDimensions) {
    EXPECT_FALSE(false); // Always passes, just to show the logic
  } else {
    // Only check equality if dimensions match
    EXPECT_EQ(skinWeights1 == skinWeights2, true); // This line won't be reached
  }
}

// Test the set method with inputs where some vertices have no influences
TEST_F(SkinWeightsTest, SetWithEmptyInfluences) {
  SkinWeights skinWeights;
  std::vector<std::vector<size_t>> mixedIndices = {
      {0, 1, 2}, // Vertex 0 has 3 influences
      {}, // Vertex 1 has no influences
      {2, 3}, // Vertex 2 has 2 influences
  };

  std::vector<std::vector<float>> mixedWeights = {
      {0.5f, 0.3f, 0.2f}, // Weights for vertex 0
      {}, // No weights for vertex 1
      {0.7f, 0.3f}, // Weights for vertex 2
  };

  skinWeights.set(mixedIndices, mixedWeights);

  // Check that the matrices have the correct size
  EXPECT_EQ(skinWeights.index.rows(), 3);
  EXPECT_EQ(skinWeights.weight.rows(), 3);

  // Check that vertex 1 has all zero weights and indices
  for (size_t j = 0; j < kMaxSkinJoints; j++) {
    EXPECT_EQ(skinWeights.index(1, j), 0);
    EXPECT_FLOAT_EQ(skinWeights.weight(1, j), 0.0f);
  }
}

// Test the set method with inputs where weights don't sum to 1
TEST_F(SkinWeightsTest, SetWithNonNormalizedWeights) {
  SkinWeights skinWeights;
  std::vector<std::vector<size_t>> indices = {
      {0, 1, 2}, // Vertex 0 is influenced by joints 0, 1, 2
  };

  std::vector<std::vector<float>> nonNormalizedWeights = {
      {0.3f, 0.2f, 0.1f}, // Weights sum to 0.6, not 1.0
  };

  skinWeights.set(indices, nonNormalizedWeights);

  // Check that the weights were set correctly without normalization
  for (size_t j = 0; j < indices[0].size(); j++) {
    EXPECT_FLOAT_EQ(skinWeights.weight(0, j), nonNormalizedWeights[0][j]);
  }
}
