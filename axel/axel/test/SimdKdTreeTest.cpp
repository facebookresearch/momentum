/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <momentum/math/random.h>

#include <array>

#include "axel/SimdKdTree.h"

#ifdef AXEL_ENABLE_AVX
constexpr int32_t kAvxFloatBlockSize = 8;
constexpr int32_t kAvxAlignment = kAvxFloatBlockSize * 4;
#endif

// Constants for test stability
constexpr uint32_t kTestRandomSeed = 42;
constexpr float kTestTolerance = 1e-3f;

namespace axel::test {

template <typename T, typename TreeType>
void validateKdTreeNearestNeighbor(
    const std::vector<Eigen::Matrix<T, 3, 1>>& points,
    const std::vector<Eigen::Matrix<T, 3, 1>>& normals,
    const std::vector<Eigen::Matrix<T, 4, 1>>& colors,
    const TreeType& kdTree,
    const Eigen::Matrix<T, 3, 1>& queryPoint) {
  using SizeType = SimdKdTree3f::SizeType;
  using Scalar = SimdKdTree3f::Scalar;

  if (kdTree.empty()) {
    return;
  }

  T tol = kTestTolerance;

  const auto [found, closestPoint, bestDistSqr] = kdTree.closestPoint(queryPoint);

  EXPECT_TRUE(found);
  EXPECT_FLOAT_EQ((points[closestPoint] - queryPoint).squaredNorm(), bestDistSqr);
  EXPECT_GE(closestPoint, 0);
  EXPECT_LT(closestPoint, points.size());

  // The distances of all the points should be equal to or greater than the nearest distance
  for (const auto& point : points) {
    EXPECT_GE((point - queryPoint).squaredNorm() + tol, bestDistSqr);
  }

  // The result should be the same when searching the larger distance of the nearest distance
  const auto [found2, index2, dist2] = kdTree.closestPoint(queryPoint, Scalar(2) * bestDistSqr);
  EXPECT_EQ(found2, true);
  EXPECT_EQ(index2, closestPoint);

  // No nearest neighbor should be found when searching the shorter distance of the nearest distance
  const auto [found3, index3, dist3] = kdTree.closestPoint(queryPoint, Scalar(0.5) * bestDistSqr);
  EXPECT_EQ(found3, false);

  const size_t nNormalsToTest = 5;
  std::array<Eigen::Vector3f, nNormalsToTest> queryNormals = {
      Eigen::Matrix<T, 3, 1>::UnitX(),
      -Eigen::Matrix<T, 3, 1>::UnitX(),
      Eigen::Matrix<T, 3, 1>::UnitY(),
      -Eigen::Matrix<T, 3, 1>::UnitY(),
      Eigen::Matrix<T, 3, 1>(5, 4, 3).normalized()};

  // Find the closest point with momentum::normal criteria
  for (size_t iNormalTest = 0; iNormalTest < nNormalsToTest; ++iNormalTest) {
    const Eigen::Matrix<T, 3, 1> queryNormal = queryNormals[iNormalTest];
    const auto [found4, index4, dist4] = kdTree.closestPoint(queryPoint, queryNormal);

    if (!found4) {
      for (SizeType i = 0; i < points.size(); ++i) {
        EXPECT_LT(normals[i].dot(queryNormal), 0.0f);
      }
      continue;
    }

    const SizeType closestPointIndex = index4;
    EXPECT_GE(normals[closestPointIndex].dot(queryNormal), 0.0f);
    for (SizeType i = 0; i < points.size(); ++i) {
      if (normals[i].dot(queryNormal) < 0.0f) {
        continue;
      }

      EXPECT_GE((points[i] - queryPoint).squaredNorm() + tol, bestDistSqr);
    }
  }

  const size_t nColorsToTest = 3;
  std::array<Eigen::Vector4f, nColorsToTest> queryColors = {
      Eigen::Vector4f(1, 0, 0, 1), Eigen::Vector4f(0, 1, 0, 1), Eigen::Vector4f(0, 0, 1, 1)};

  // Find he closest point with momentum::normal and color criteria
  const auto queryNormal = queryNormals[0];
  for (size_t iColorTest = 0; iColorTest < nColorsToTest; ++iColorTest) {
    const auto queryColor = queryColors[iColorTest];
    const auto [found5, index5, dist5] = kdTree.closestPoint(
        queryPoint, queryNormal, queryColor, std::numeric_limits<Scalar>::max(), 0.0f, 0.5f);

    if (!found5) {
      for (SizeType i = 0; i < points.size(); ++i) {
        EXPECT_TRUE(
            normals[i].dot(queryNormal) < 0.0f ||
            (colors[i] - queryColor).squaredNorm() > 0.5f - tol);
      }
      continue;
    }

    const SizeType closestPointIndex = index5;
    EXPECT_TRUE(
        normals[closestPointIndex].dot(queryNormal) >= 0.0f ||
        (colors[closestPointIndex] - queryColor).squaredNorm() <= 0.5f + tol);
    for (SizeType i = 0; i < points.size(); ++i) {
      if (normals[i].dot(queryNormal) < 0.0f) {
        continue;
      }

      if ((colors[i] - queryColor).squaredNorm() > 0.5f) {
        continue;
      }

      EXPECT_GE((points[i] - queryPoint).squaredNorm() + tol, bestDistSqr);
    }
  }
}

template <typename T, typename TreeType>
void validateKdTreeNearestNeighborWithAcceptance(
    const std::vector<Eigen::Matrix<T, 3, 1>>& points,
    const TreeType& kdTree,
    const Eigen::Matrix<T, 3, 1>& queryPoint) {
  using SizeType = SimdKdTree3f::SizeType;
  using Scalar = SimdKdTree3f::Scalar;

  if (kdTree.empty()) {
    return;
  }

  T tol = kTestTolerance;

  const auto [found, closestPoint, bestDistSqr] = kdTree.closestPoint(queryPoint);

  EXPECT_TRUE(found);
  EXPECT_FLOAT_EQ((points[closestPoint] - queryPoint).squaredNorm(), bestDistSqr);
  EXPECT_GE(closestPoint, 0);
  EXPECT_LT(closestPoint, points.size());

#ifdef AXEL_ENABLE_AVX
  if constexpr (std::is_same_v<TreeType, SimdKdTreeAvx3f>) {
    // Try it with a positive acceptance function:
    const auto [found5, index5, dist5] = kdTree.closestPointWithAcceptance(
        queryPoint, std::numeric_limits<Scalar>::max(), [&](__m256i indices_in) -> __m256i {
          // All zero bytes
          const int all_zeros = 0;

          // All one bytes
          const int all_ones = ~all_zeros;

          // Do AVX magic: vector => []
          alignas(kAvxAlignment) std::array<int32_t, kAvxFloatBlockSize> indices{};
          _mm256_store_si256((__m256i*)indices.data(), indices_in);

          // For all kAvxFloatBlockSize indices in the AVX vector
          alignas(kAvxAlignment) std::array<int32_t, kAvxFloatBlockSize> result{};
          for (int32_t j = 0; j < 8; ++j) {
            result[j] = all_ones;
          }

          // Return AVX vector
          return _mm256_load_si256((const __m256i*)result.data());
        });
    ASSERT_TRUE(found5);
    EXPECT_EQ(closestPoint, index5);

    // Try it with a negative acceptance function:
    const auto [found6, index6, dist6] = kdTree.closestPointWithAcceptance(
        queryPoint, std::numeric_limits<Scalar>::max(), [&](__m256i indices_in) -> __m256i {
          // All zero bytes
          const int all_zeros = 0;

          // Do AVX magic: vector => []
          alignas(kAvxAlignment) std::array<int32_t, kAvxFloatBlockSize> indices{};
          _mm256_store_si256((__m256i*)indices.data(), indices_in);

          // For all kAvxFloatBlockSize indices in the AVX vector
          alignas(kAvxAlignment) std::array<int32_t, kAvxFloatBlockSize> result{};
          for (int32_t j = 0; j < 8; ++j) {
            result[j] = all_zeros;
          }

          // Return AVX vector
          return _mm256_load_si256((const __m256i*)result.data());
        });
    EXPECT_FALSE(found6);

    // Try it with an acceptance function:
    const auto [found7, index7, dist7] = kdTree.closestPointWithAcceptance(
        queryPoint, std::numeric_limits<Scalar>::max(), [&](__m256i indices_in) -> __m256i {
          // All zero bytes
          const int all_zeros = 0;

          // All one bytes
          const int all_ones = ~all_zeros;

          // Do AVX magic: vector => []
          alignas(kAvxAlignment) std::array<int32_t, kAvxFloatBlockSize> indices{};
          _mm256_store_si256((__m256i*)indices.data(), indices_in);

          // For all kAvxFloatBlockSize indices in the AVX vector
          alignas(kAvxAlignment) std::array<int32_t, kAvxFloatBlockSize> result{};
          for (int32_t j = 0; j < 8; ++j) {
            result[j] = (indices[j] % 2 == 0) ? all_ones : all_zeros;
          }

          // Return AVX vector
          return _mm256_load_si256((const __m256i*)result.data());
        });
    ASSERT_TRUE(found7);
    EXPECT_EQ(0, index7 % 2);
    if (closestPoint % 2 == 0) {
      EXPECT_EQ(closestPoint, index7);
    } else {
      EXPECT_GE(dist7 + tol, bestDistSqr);
    }
    for (SizeType i = 0; i < points.size(); i += 2) {
      EXPECT_GE((points[i] - queryPoint).squaredNorm() + tol, bestDistSqr);
    }
  } else {
#endif
    // Try it with a positive acceptance function:
    const auto [found5, index5, dist5] = kdTree.closestPointWithAcceptance(
        queryPoint, std::numeric_limits<Scalar>::max(), [](SizeType) -> bool { return true; });
    ASSERT_TRUE(found5);
    EXPECT_EQ(closestPoint, index5);

    // Try it with a negative acceptance function:
    const auto [found6, index6, dist6] = kdTree.closestPointWithAcceptance(
        queryPoint, std::numeric_limits<Scalar>::max(), [](SizeType) -> bool { return false; });
    EXPECT_FALSE(found6);

    // Try it with an acceptance function:
    const auto [found7, index7, dist7] = kdTree.closestPointWithAcceptance(
        queryPoint, std::numeric_limits<Scalar>::max(), [](SizeType index) -> bool {
          return (index % 2 == 0);
        });
    ASSERT_TRUE(found7);
    EXPECT_EQ(0, index7 % 2);
    if (closestPoint % 2 == 0) {
      EXPECT_EQ(closestPoint, index7);
    } else {
      EXPECT_GE(dist7 + tol, bestDistSqr);
    }
    for (SizeType i = 0; i < points.size(); i += 2) {
      EXPECT_GE((points[i] - queryPoint).squaredNorm() + tol, bestDistSqr);
    }
#ifdef AXEL_ENABLE_AVX
  }
#endif
}

template <typename T, typename TreeType>
void validateKdTreeSphereQuery(
    const std::vector<Eigen::Matrix<T, 3, 1>>& points,
    const TreeType& kdTree,
    const Eigen::Matrix<T, 3, 1>& center,
    float radius) {
  using SizeType = SimdKdTree3f::SizeType;
  using Scalar = SimdKdTree3f::Scalar;

  std::vector<SizeType> treeResult;
  kdTree.pointsInNSphere(center, radius, treeResult);

  // The result points should be within the radius
  const Scalar tol = kTestTolerance;
  const Scalar radSqr = radius * radius;
  for (auto i = 0u; i < points.size(); ++i) {
    const auto& point = points[i];
    if (std::find(treeResult.begin(), treeResult.end(), i) != treeResult.end()) {
      EXPECT_LE((point - center).squaredNorm(), radSqr + tol);
    } else {
      EXPECT_GE((point - center).squaredNorm(), radSqr - tol);
    }
  }
}

template <typename T, typename TreeType>
void validateKdTree(
    const std::vector<Eigen::Matrix<T, 3, 1>>& points,
    const std::vector<Eigen::Matrix<T, 3, 1>>& normals,
    const std::vector<Eigen::Matrix<T, 4, 1>>& colors,
    const TreeType& kdTree) {
  using Scalar = SimdKdTree3f::Scalar;

  momentum::Random<>::GetSingleton().setSeed(kTestRandomSeed);

  kdTree.validate();
  validateKdTreeNearestNeighbor<T>(points, normals, colors, kdTree, Eigen::Matrix<T, 3, 1>::Zero());
  validateKdTreeNearestNeighborWithAcceptance<T>(points, kdTree, Eigen::Matrix<T, 3, 1>::Zero());

  // Test a bunch of random query points:
  const size_t nTests = 1000;

  for (size_t i = 0; i < nTests; ++i) {
    const auto queryPoint = momentum::normal<Eigen::Matrix<T, 3, 1>>(0, 3);
    validateKdTreeNearestNeighbor(points, normals, colors, kdTree, queryPoint);
    validateKdTreeNearestNeighborWithAcceptance(points, kdTree, queryPoint);
  }

  for (size_t i = 0; i < nTests; ++i) {
    const auto queryPoint = momentum::normal<Eigen::Matrix<T, 3, 1>>(0, 3);
    const Scalar radius = momentum::uniform<float>(0, 5);
    validateKdTreeSphereQuery(points, kdTree, queryPoint, radius);
  }
}

TEST(SimdKdTreeTest, StaticProperties) {
  // 3D
  EXPECT_EQ(SimdKdTree2f::kDim, 2);
  EXPECT_EQ(SimdKdTree2f::kColorDimensions, 4);

  // 3D
  EXPECT_EQ(SimdKdTree3f::kDim, 3);
  EXPECT_EQ(SimdKdTree3f::kColorDimensions, 4);
}

TEST(SimdKdTreeTest, EmptyTree) {
  auto kdTree = SimdKdTree3f();
  EXPECT_TRUE(kdTree.empty());
  EXPECT_EQ(kdTree.size(), 0);
  EXPECT_EQ(kdTree.depth(), 0);
  EXPECT_FALSE(kdTree.boundingBox().contains(Eigen::Vector3f::Zero()));
  kdTree.validate();

  {
    const auto [found, index, dist] = kdTree.closestPoint(Eigen::Vector3f::Random(), 1);
    EXPECT_FALSE(found);
    EXPECT_EQ(index, std::numeric_limits<SimdKdTree3f::SizeType>::max());
    EXPECT_FLOAT_EQ(dist, 1);
  }

  {
    const auto [found, index, dist] =
        kdTree.closestPoint(Eigen::Vector3f::Random(), Eigen::Vector3f::Random().normalized(), 1);
    EXPECT_FALSE(found);
    EXPECT_EQ(index, std::numeric_limits<SimdKdTree3f::SizeType>::max());
    EXPECT_FLOAT_EQ(dist, 1);
  }

  {
    const auto [found, index, dist] = kdTree.closestPoint(
        Eigen::Vector3f::Random(),
        Eigen::Vector3f::Random().normalized(),
        Eigen::Vector4f::Random(),
        1);
    EXPECT_FALSE(found);
    EXPECT_EQ(index, std::numeric_limits<SimdKdTree3f::SizeType>::max());
    EXPECT_FLOAT_EQ(dist, 1);
  }

  {
    const auto [found, index, dist] = kdTree.closestPointWithAcceptance(
        Eigen::Vector3f::Random(), 1, [=](SimdKdTree3f::SizeType) -> bool { return true; });
    EXPECT_FALSE(found);
    EXPECT_EQ(index, std::numeric_limits<SimdKdTree3f::SizeType>::max());
    EXPECT_FLOAT_EQ(dist, 1);
  }

#ifdef AXEL_ENABLE_AVX
  auto kdTreeAvx = SimdKdTreeAvx3f();
  EXPECT_TRUE(kdTreeAvx.empty());
  EXPECT_EQ(kdTreeAvx.size(), 0);
  EXPECT_EQ(kdTreeAvx.depth(), 0);
  const auto [foundAvx, indexAvx, distAvx] = kdTree.closestPoint(Eigen::Vector3f::Random(), 1);
  EXPECT_FALSE(foundAvx);
  EXPECT_EQ(indexAvx, std::numeric_limits<SimdKdTree3f::SizeType>::max());
  EXPECT_FLOAT_EQ(distAvx, 1);
  kdTreeAvx.validate();

  {
    const auto [found, index, dist] = kdTreeAvx.closestPoint(Eigen::Vector3f::Random(), 1);
    EXPECT_FALSE(found);
    EXPECT_EQ(index, std::numeric_limits<SimdKdTree3f::SizeType>::max());
    EXPECT_FLOAT_EQ(dist, 1);
  }

  {
    const auto [found, index, dist] = kdTreeAvx.closestPoint(
        Eigen::Vector3f::Random(), Eigen::Vector3f::Random().normalized(), 1);
    EXPECT_FALSE(found);
    EXPECT_EQ(index, std::numeric_limits<SimdKdTree3f::SizeType>::max());
    EXPECT_FLOAT_EQ(dist, 1);
  }

  {
    const auto [found, index, dist] = kdTreeAvx.closestPoint(
        Eigen::Vector3f::Random(),
        Eigen::Vector3f::Random().normalized(),
        Eigen::Vector4f::Random(),
        1);
    EXPECT_FALSE(found);
    EXPECT_EQ(index, std::numeric_limits<SimdKdTree3f::SizeType>::max());
    EXPECT_FLOAT_EQ(dist, 1);
  }

  {
    const auto [found, index, dist] = kdTreeAvx.closestPointWithAcceptance(
        Eigen::Vector3f::Random(), 1, [&](__m256i indices_in) -> __m256i {
          // All zero bytes
          const int all_zeros = 0;

          // All one bytes
          const int all_ones = ~all_zeros;

          // Do AVX magic: vector => []
          alignas(kAvxAlignment) std::array<int32_t, kAvxFloatBlockSize> indices{};
          _mm256_store_si256((__m256i*)indices.data(), indices_in);

          // For all kAvxFloatBlockSize indices in the AVX vector
          alignas(kAvxAlignment) std::array<int32_t, kAvxFloatBlockSize> result{};
          for (int32_t j = 0; j < 8; ++j) {
            if (indices[j] % 2 == 0) {
              result[j] = all_ones;
            } else {
              result[j] = all_zeros;
            }
          }

          // Return AVX vector
          return _mm256_load_si256((const __m256i*)result.data());
        });
    EXPECT_FALSE(found);
    EXPECT_EQ(index, std::numeric_limits<SimdKdTree3f::SizeType>::max());
    EXPECT_FLOAT_EQ(dist, 1);
  }
#endif
}

TEST(SimdKdTreeTest, SmallTrees) {
  using Vec = SimdKdTree3f::Vec;
  using Col = SimdKdTree3f::Col;

  momentum::Random<>::GetSingleton().setSeed(kTestRandomSeed);

  // Create a bunch of small-ish kd-trees to make sure there aren't any problem behaviors:
  for (size_t i = 1; i < 16; ++i) {
    std::vector<Vec> points;
    for (size_t j = 0; j < i; ++j) {
      points.push_back(momentum::normal<Vec>(0, 3));
    }

    std::vector<Vec> normals;
    for (size_t j = 0; j < i; ++j) {
      normals.push_back(momentum::normal<Vec>(0, 3).normalized());
    }

    std::vector<Col> colors;
    for (size_t j = 0; j < i; ++j) {
      Col color;
      if (j % 3 == 0) {
        color << 1, 0, 0, 1;
      } else if (j % 3 == 1) {
        color << 0, 1, 0, 1;
      } else if (j % 3 == 2) {
        color << 0, 0, 1, 1;
      } else {
        color << 0, 0, 0, 1;
      }
      colors.push_back(color);
    }

    SimdKdTree3f kdTree(points, normals, colors);
    validateKdTree(points, normals, colors, kdTree);

#ifdef AXEL_ENABLE_AVX
    SimdKdTreeAvx3f kdTreeAvx(points, normals, colors);
    validateKdTree(points, normals, colors, kdTreeAvx);
#endif
  }
}

TEST(SimdKdTreeTest, RepeatedPoint) {
  using Vec = SimdKdTree3f::Vec;
  using Col = SimdKdTree3f::Col;

  momentum::Random<>::GetSingleton().setSeed(kTestRandomSeed);

  // Make sure the corner case where we just have a lot of the same point doesn't break anything.
  for (size_t i = 1; i < 10; ++i) {
    const size_t nPoints = 100;

    std::vector<Vec> points(nPoints, momentum::normal<Vec>(0, 3));
    for (size_t j = 0; j < nPoints; ++j) {
      points.push_back(momentum::normal<Vec>(0, 3));
    }

    std::vector<Vec> normals;
    for (size_t j = 0; j < (2 * nPoints); ++j) {
      normals.push_back(momentum::normal<Vec>(0, 3).normalized());
    }

    std::vector<Col> colors(nPoints, momentum::normal<Col>(0, 3));
    for (size_t j = 0; j < nPoints; ++j) {
      Col color;
      if (j % 3 == 0) {
        color << 1, 0, 0, 1;
      } else if (j % 3 == 1) {
        color << 0, 1, 0, 1;
      } else if (j % 3 == 2) {
        color << 0, 0, 1, 1;
      } else {
        color << 0, 0, 0, 1;
      }
      colors.push_back(color);
    }

    SimdKdTree3f kdTree(points, normals, colors);
    validateKdTree(points, normals, colors, kdTree);

#ifdef AXEL_ENABLE_AVX
    SimdKdTreeAvx3f kdTreeAvx(points, normals, colors);
    validateKdTree(points, normals, colors, kdTreeAvx);
#endif
  }
}

TEST(SimdKdTreeTest, BigTree) {
  using Vec = SimdKdTree3f::Vec;
  using Col = SimdKdTree3f::Col;

  momentum::Random<>::GetSingleton().setSeed(kTestRandomSeed);

  // Create some really big kd-trees and validate that it works:
  for (size_t i = 0; i < 10; ++i) {
    const size_t numPoints = (size_t)1 << i;
    std::vector<Vec> points;
    for (size_t jPt = 0; jPt < numPoints; ++jPt) {
      points.push_back(momentum::normal<Vec>(0, 3));
    }

    std::vector<Vec> normals;
    for (size_t jPt = 0; jPt < numPoints; ++jPt) {
      normals.push_back(momentum::normal<Vec>(0, 3).normalized());
    }

    std::vector<Col> colors;
    for (size_t jPt = 0; jPt < numPoints; ++jPt) {
      Col color;
      if (jPt % 3 == 0) {
        color << 1, 0, 0, 1;
      } else if (jPt % 3 == 1) {
        color << 0, 1, 0, 1;
      } else if (jPt % 3 == 2) {
        color << 0, 0, 1, 1;
      } else {
        color << 0, 0, 0, 1;
      }
      colors.push_back(color);
    }

    SimdKdTree3f kdTree(points, normals, colors);
    validateKdTree(points, normals, colors, kdTree);

#ifdef AXEL_ENABLE_AVX
    SimdKdTreeAvx3f kdTreeAvx(points, normals, colors);
    validateKdTree(points, normals, colors, kdTreeAvx);
#endif
  }
}

} // namespace axel::test
