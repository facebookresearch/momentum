/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/test/character/character_helpers_gtest.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/character.h"
#include "momentum/character/collision_geometry.h"
#include "momentum/character/locator.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/pose_shape.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skin_weights.h"
#include "momentum/math/mesh.h"

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>

namespace momentum {

namespace {

MATCHER_P(FloatNearPointwise, tol, "Value mismatch") {
  for (int i = 0; i < std::get<0>(arg).size(); i++) {
    if (std::abs(std::get<0>(arg)[i] - std::get<1>(arg)[i]) > tol) {
      return false;
    }
  }
  return true;
}

MATCHER(IntExactPointwise, "Value mismatch") {
  for (int i = 0; i < std::get<0>(arg).size(); i++) {
    if (std::get<0>(arg)[i] != std::get<1>(arg)[i]) {
      return false;
    }
  }
  return true;
}

MATCHER(ElementsEq, "Elements mismatch") {
  return ::testing::get<0>(arg) == ::testing::get<1>(arg);
}

MATCHER(ElementsIsApprox, "Elements mismatch") {
  return ::testing::get<0>(arg).isApprox(::testing::get<1>(arg));
}

template <typename DerivedIndexMatrix, typename DerivedWeightMatrix>
void sortSkinWeightsRows(
    Eigen::MatrixBase<DerivedIndexMatrix>& indexMap,
    Eigen::MatrixBase<DerivedWeightMatrix>& weightMap) {
  using Index = typename DerivedIndexMatrix::Scalar;
  using Scalar = typename DerivedWeightMatrix::Scalar;

  for (int i = 0; i < indexMap.rows(); ++i) {
    std::vector<std::pair<Index, Scalar>> pairs;

    for (int j = 0; j < indexMap.cols(); ++j) {
      if (indexMap(i, j) == 0) { // Assuming 0 is the marker for unused elements
        break;
      }
      pairs.emplace_back(indexMap(i, j), weightMap(i, j));
    }

    // Sort pairs based on the first element of the pair (index value)
    std::sort(
        pairs.begin(),
        pairs.end(),
        [](const std::pair<Index, Scalar>& a, const std::pair<Index, Scalar>& b) {
          return a.first < b.first;
        });

    // Place sorted values back into the matrices
    for (Index k = 0; k < pairs.size(); ++k) {
      indexMap(i, k) = pairs[k].first;
      weightMap(i, k) = pairs[k].second;
    }
  }
}

} // namespace

void compareMeshes(const Mesh_u& refMesh, const Mesh_u& mesh) {
  ASSERT_TRUE((refMesh && mesh)) << "Mesh A (" << refMesh.get() << ") or Mesh B (" << mesh.get()
                                 << ") is null";
  EXPECT_THAT(refMesh->vertices, testing::Pointwise(FloatNearPointwise(0.0001), mesh->vertices));
  EXPECT_THAT(refMesh->normals, testing::Pointwise(FloatNearPointwise(0.01), mesh->normals));
  EXPECT_THAT(refMesh->faces, testing::Pointwise(IntExactPointwise(), mesh->faces));
  EXPECT_THAT(refMesh->colors, testing::Pointwise(IntExactPointwise(), mesh->colors));
  EXPECT_THAT(refMesh->texcoords, testing::Pointwise(FloatNearPointwise(0.0001), mesh->texcoords));
  EXPECT_THAT(
      refMesh->confidence, testing::Pointwise(testing::DoubleNear(0.0001), mesh->confidence));
  EXPECT_THAT(
      refMesh->texcoord_faces, testing::Pointwise(IntExactPointwise(), mesh->texcoord_faces));
}

void compareLocators(const LocatorList& refLocators, const LocatorList& locators) {
  EXPECT_EQ(refLocators.size(), locators.size());
  auto sortedRefLocators = refLocators;
  auto sortedLocators = locators;
  auto compareLocators = [](const Locator& l1, const Locator& l2) {
    return l1.parent != l2.parent ? l1.parent < l2.parent : l1.name < l2.name;
  };
  std::sort(sortedRefLocators.begin(), sortedRefLocators.end(), compareLocators);
  std::sort(sortedLocators.begin(), sortedLocators.end(), compareLocators);
  EXPECT_THAT(sortedRefLocators, testing::Pointwise(ElementsEq(), sortedLocators));
}

void compareCollisionGeometry(
    const CollisionGeometry_u& refCollision,
    const CollisionGeometry_u& collision) {
  if (refCollision == nullptr) {
    ASSERT_EQ(collision, nullptr);
  } else {
    ASSERT_NE(collision, nullptr);
    EXPECT_EQ(refCollision->size(), collision->size());
    auto sortedRefCollision = *refCollision;
    auto sortedCollision = *collision;
    auto compareCollisions = [](const TaperedCapsule& l1, const TaperedCapsule& l2) {
      if (l1.parent != l2.parent) {
        return l1.parent < l2.parent;
      }
      if (l1.length != l2.length) {
        return l1.length < l2.length;
      }
      if (!l1.radius.isApprox(l2.radius)) {
        return l1.radius.x() != l2.radius.x() ? l1.radius.x() < l2.radius.x()
                                              : l1.radius.y() < l2.radius.y();
      }
      const auto t1 = l1.transformation.toMatrix();
      const auto t2 = l2.transformation.toMatrix();
      EXPECT_EQ(t1.size(), t2.size());
      for (auto i = 0; i < t1.size(); i++) {
        if (std::abs(t1.data()[i] - t2.data()[i]) > std::numeric_limits<float>::epsilon()) {
          return t1.data()[i] < t2.data()[i];
        }
      }
      return false;
    };
    std::sort(sortedRefCollision.begin(), sortedRefCollision.end(), compareCollisions);
    std::sort(sortedCollision.begin(), sortedCollision.end(), compareCollisions);
    for (size_t i = 0; i < sortedCollision.size(); ++i) {
      const auto& collA = sortedRefCollision[i];
      const auto& collB = sortedCollision[i];

      EXPECT_TRUE(collA.isApprox(collB)) << "Collision geometry mismatch at index " << i << ":\n"
                                         << "- refCollision:\n"
                                         << "  - radius_0 : " << collA.radius.x() << "\n"
                                         << "  - radius_1 : " << collA.radius.y() << "\n"
                                         << "  - length   : " << collA.length << "\n"
                                         << "  - parent   : " << collA.parent << "\n"
                                         << "  - transform:\n"
                                         << collA.transformation.toMatrix() << "\n"
                                         << "- collision:\n"
                                         << "  - radius_0 : " << collB.radius.x() << "\n"
                                         << "  - radius_1 : " << collB.radius.y() << "\n"
                                         << "  - length   : " << collB.length << "\n"
                                         << "  - parent   : " << collB.parent << "\n"
                                         << "  - transform:\n"
                                         << collB.transformation.toMatrix() << std::endl;
    }
  }
}

void compareChars(const Character& refChar, const Character& character, const bool withMesh) {
  ASSERT_EQ(refChar.name, character.name);
  const auto& refJoints = refChar.skeleton.joints;
  const auto& joints = character.skeleton.joints;
  ASSERT_EQ(refJoints.size(), joints.size());
  for (size_t i = 0; i < refJoints.size(); ++i) {
    EXPECT_TRUE(refJoints[i].isApprox(joints[i]))
        << "Joint " << i << " is not equal:\n"
        << "- refJoint:\n"
        << "  - name: " << refJoints[i].name << "\n"
        << "  - parent: " << refJoints[i].parent << "\n"
        << "  - preRotation: " << refJoints[i].preRotation.coeffs().transpose() << "\n"
        << "  - translationOffset: " << refJoints[i].translationOffset.transpose() << "\n"
        << "- joint:\n"
        << "  - name: " << joints[i].name << "\n"
        << "  - parent: " << joints[i].parent << "\n"
        << "  - preRotation: " << joints[i].preRotation.coeffs().transpose() << "\n"
        << "  - translationOffset: " << joints[i].translationOffset.transpose() << "\n";
  }
  ASSERT_TRUE(refChar.parameterTransform.isApprox(character.parameterTransform));
  EXPECT_THAT(refChar.parameterLimits, testing::Pointwise(ElementsEq(), character.parameterLimits));
  compareLocators(refChar.locators, character.locators);
  compareCollisionGeometry(refChar.collision, character.collision);
  ASSERT_EQ(refChar.inverseBindPose.size(), character.inverseBindPose.size());
  for (size_t i = 0; i < refChar.inverseBindPose.size(); ++i) {
    EXPECT_TRUE(refChar.inverseBindPose[i].isApprox(character.inverseBindPose[i], 1e-4f))
        << "InverseBindPose " << i << " is not equal:\n"
        << "- Expected:\n"
        << refChar.inverseBindPose[i].matrix() << "\n"
        << "- Actual  :\n"
        << character.inverseBindPose[i].matrix() << std::endl;
  }
  EXPECT_EQ(refChar.jointMap, character.jointMap);

  if (withMesh) {
    auto ptrValEq = [](auto& l, auto& r) {
      if (l == nullptr && r == nullptr) {
        return true;
      }
      if (l == nullptr) {
        return (r == nullptr);
      }
      if (r == nullptr) {
        return false;
      }
      return true;
    };

    compareMeshes(refChar.mesh, character.mesh);

    auto refCharIndices = refChar.skinWeights->index;
    auto refCharWeights = refChar.skinWeights->weight;
    sortSkinWeightsRows(refCharIndices, refCharWeights);

    auto charIndices = refChar.skinWeights->index;
    auto charWeights = refChar.skinWeights->weight;
    sortSkinWeightsRows(charIndices, charWeights);

    ASSERT_EQ(refCharIndices.rows(), charIndices.rows());
    ASSERT_EQ(refCharIndices.cols(), charIndices.cols());
    for (auto i = 0u; i < refCharIndices.rows(); ++i) {
      EXPECT_EQ(refCharIndices.row(i), charIndices.row(i))
          << "SkinWeights index row " << i << " mismatch\n";
    }
    ASSERT_LT((refCharWeights - charWeights).lpNorm<Eigen::Infinity>(), 1e-4f);
    ASSERT_TRUE(ptrValEq(refChar.skinWeights, character.skinWeights));

    auto ptrValIsApprox = [](auto& l, auto& r) {
      if (l == nullptr) {
        return (r == nullptr);
      }
      if (r == nullptr) {
        return false;
      }
      return l->isApprox(*r);
    };
    ASSERT_TRUE(ptrValIsApprox(refChar.poseShapes, character.poseShapes));
    ASSERT_TRUE(ptrValIsApprox(refChar.blendShape, character.blendShape));
  }
}

} // namespace momentum
