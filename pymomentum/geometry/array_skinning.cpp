/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/array_skinning.h"

#include <pymomentum/array_utility/array_utility.h>
#include <pymomentum/array_utility/batch_accessor.h>
#include <pymomentum/array_utility/geometry_accessors.h>

#include <momentum/character/linear_skinning.h>
#include <momentum/character/skin_weights.h>
#include <momentum/common/exception.h>
#include <momentum/math/mesh.h>

#include <dispenso/parallel_for.h>

namespace pymomentum {

namespace {

template <typename T>
py::array_t<T> skinPointsImpl(
    const momentum::Character& character,
    const py::array& skelState,
    const std::optional<py::array>& restVertices,
    const LeadingDimensions& leadingDims) {
  MT_THROW_IF(
      !character.mesh || !character.skinWeights,
      "skin_points: character is missing a mesh or skin weights");

  const auto nJoints = static_cast<py::ssize_t>(character.skeleton.joints.size());
  const auto nVertices = static_cast<py::ssize_t>(character.mesh->vertices.size());
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array with shape [..., nVertices, 3]
  auto result = createOutputArray<T>(leadingDims, {nVertices, static_cast<py::ssize_t>(3)});

  // Create batch indexer
  BatchIndexer indexer(leadingDims);

  // Get inverse bind pose in the correct type
  const auto inverseBindPose = momentum::cast<T>(character.inverseBindPose);
  MT_THROW_IF(
      static_cast<py::ssize_t>(inverseBindPose.size()) != nJoints,
      "inverseBindPose has wrong size");

  // Create transform accessor (auto-detects skeleton state vs matrix format)
  TransformAccessor<T> transformAcc(skelState, leadingDims, nJoints);

  // Get rest vertices if provided
  // Handle broadcasting: rest_vertices may have fewer leading dims than skel_state
  std::optional<VectorArrayAccessor<T, 3>> restVerticesAcc;
  std::vector<momentum::Vector3<T>> restPoints;
  LeadingDimensions restLeadingDims;
  if (restVertices.has_value()) {
    py::buffer_info restInfo = restVertices.value().request();
    // Compute leading dims for rest vertices (total dims - 2 trailing dims)
    const auto restLeadingNDim = restInfo.ndim - 2;
    restLeadingDims.dims.clear();
    for (py::ssize_t i = 0; i < restLeadingNDim; ++i) {
      restLeadingDims.dims.push_back(restInfo.shape[i]);
    }
    restVerticesAcc.emplace(restVertices.value(), restLeadingDims, nVertices);
  } else {
    restPoints.reserve(nVertices);
    for (const auto& v : character.mesh->vertices) {
      restPoints.push_back(v.template cast<T>());
    }
  }

  // Create output accessor
  VectorArrayAccessor<T, 3> outputAcc(result, leadingDims, nVertices);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get world-space transforms from skeleton state or matrices
      const momentum::TransformListT<T> transforms = transformAcc.getTransforms(indices);

      // Get rest points (handle broadcasting)
      std::vector<momentum::Vector3<T>> restPts;
      if (restVerticesAcc.has_value()) {
        // If rest vertices has fewer leading dims, use appropriate subset of indices
        if (restLeadingDims.dims.empty()) {
          // Unbatched rest vertices - use empty indices
          std::vector<py::ssize_t> emptyIndices;
          restPts = restVerticesAcc->get(emptyIndices);
        } else {
          // Batched rest vertices - use corresponding indices
          restPts = restVerticesAcc->get(indices);
        }
      } else {
        restPts = restPoints;
      }

      // Apply linear blend skinning using the refactored applySSD function
      const auto skinnedPoints =
          momentum::applySSD<T>(inverseBindPose, *character.skinWeights, restPts, transforms);

      // Copy results to output array
      outputAcc.set(indices, skinnedPoints);
    });
  }

  return result;
}

template <typename T>
py::array_t<T> skinSkinnedLocatorsImpl(
    const momentum::Character& character,
    const py::array& skelState,
    const std::optional<py::array>& restPositions,
    const LeadingDimensions& leadingDims) {
  MT_THROW_IF(
      character.skinnedLocators.empty(),
      "skin_skinned_locators: character has no skinned locators");

  const auto nJoints = static_cast<py::ssize_t>(character.skeleton.joints.size());
  const auto nLocators = static_cast<py::ssize_t>(character.skinnedLocators.size());
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array with shape [..., nLocators, 3]
  auto result = createOutputArray<T>(leadingDims, {nLocators, static_cast<py::ssize_t>(3)});

  // Create batch indexer
  BatchIndexer indexer(leadingDims);

  // Get inverse bind pose in the correct type
  const auto inverseBindPose = momentum::cast<T>(character.inverseBindPose);
  MT_THROW_IF(inverseBindPose.size() != nJoints, "inverseBindPose has wrong size");

  // Create skeleton state accessor
  SkeletonStateAccessor<T> skelStateAcc(skelState, leadingDims, nJoints);

  // Get rest positions if provided
  std::optional<VectorArrayAccessor<T, 3>> restPositionsAcc;
  std::vector<momentum::Vector3<T>> restPts;
  if (restPositions.has_value()) {
    restPositionsAcc.emplace(restPositions.value(), leadingDims, nLocators);
  } else {
    // Use positions from skinned locators
    restPts.reserve(nLocators);
    for (const auto& locator : character.skinnedLocators) {
      restPts.push_back(locator.position.template cast<T>());
    }
  }

  // Create output accessor
  VectorArrayAccessor<T, 3> outputAcc(result, leadingDims, nLocators);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get world-space transforms from skeleton state
      const momentum::TransformListT<T> transforms = skelStateAcc.getTransforms(indices);

      // Get rest positions for this batch
      const auto& restPositions =
          restPositionsAcc.has_value() ? restPositionsAcc->get(indices) : restPts;

      // Compute skinned locator positions
      std::vector<momentum::Vector3<T>> skinnedPositions;
      skinnedPositions.reserve(nLocators);

      for (size_t iLoc = 0; iLoc < nLocators; ++iLoc) {
        const auto& locator = character.skinnedLocators[iLoc];
        const auto& restPos = restPositions[iLoc];

        momentum::Vector3<T> worldPos = momentum::Vector3<T>::Zero();
        T weightSum = T(0);

        // Apply linear blend skinning using locator's bone influences
        for (size_t k = 0; k < locator.skinWeights.size(); ++k) {
          const auto& weight = locator.skinWeights[k];
          if (weight == 0.0f) {
            break; // Weights are sorted, so we can stop here
          }

          const auto boneIndex = locator.parents[k];
          const auto& jointTransform = transforms[boneIndex];

          // Transform rest position: world = joint * invBindPose * rest
          worldPos += static_cast<T>(weight) *
              (jointTransform * (inverseBindPose[boneIndex] * restPos)).template cast<T>();
          weightSum += static_cast<T>(weight);
        }

        // Normalize by total weight
        if (weightSum > T(0)) {
          worldPos /= weightSum;
        }

        skinnedPositions.push_back(worldPos);
      }

      // Copy results to output array
      outputAcc.set(indices, skinnedPositions);
    });
  }

  return result;
}

} // namespace

py::array skinPointsArray(
    const momentum::Character& character,
    py::buffer skelState,
    std::optional<py::buffer> restVertices) {
  MT_THROW_IF(
      !character.mesh || !character.skinWeights,
      "skin_points: character is missing a mesh or skin weights");

  // Use ArrayChecker to validate inputs and detect format/dtype
  ArrayChecker checker("skin_points");

  // Validate transforms (auto-detects skeleton state vs matrix format)
  checker.validateTransforms(skelState, "skel_state", character);

  // Validate rest vertices if provided (supports broadcasting with fewer leading dims)
  if (restVertices.has_value()) {
    checker.validateVertices(restVertices.value(), "rest_vertices", character);
  }

  // Get validated arrays for processing
  py::array skelStateArr = py::array::ensure(skelState);
  std::optional<py::array> restVerticesArr;
  if (restVertices.has_value()) {
    restVerticesArr = py::array::ensure(restVertices.value());
  }

  // Get leading dimensions and dtype from checker
  const auto& leadingDims = checker.getLeadingDimensions();
  const bool isFloat64 = checker.isFloat64();

  if (isFloat64) {
    return skinPointsImpl<double>(character, skelStateArr, restVerticesArr, leadingDims);
  } else {
    return skinPointsImpl<float>(character, skelStateArr, restVerticesArr, leadingDims);
  }
}

py::array skinSkinnedLocatorsArray(
    const momentum::Character& character,
    py::buffer skelState,
    std::optional<py::buffer> restPositions) {
  // Early return for empty locators
  if (character.skinnedLocators.empty()) {
    // Return empty array with shape [0, 3]
    return py::array_t<float>(std::vector<py::ssize_t>{0, 3});
  }

  const auto nLocators = static_cast<py::ssize_t>(character.skinnedLocators.size());

  ArrayChecker checker("skin_skinned_locators");

  // Validate skeleton state: shape (..., nJoints, 8)
  checker.validateSkeletonState(skelState, "skel_state", character);

  // Get leading dimensions from validated skel_state
  const auto& leadingDims = checker.getLeadingDimensions();

  // Validate rest positions if provided
  // ArrayChecker will automatically validate that leading dimensions match skel_state
  if (restPositions.has_value()) {
    checker.validateBuffer(
        restPositions.value(),
        "rest_positions",
        {static_cast<int>(nLocators), 3},
        {"nLocators", "xyz"},
        false /* allowEmpty */);
  }

  // Get validated arrays for processing
  py::array skelStateArr = py::array::ensure(skelState);
  std::optional<py::array> restPositionsArr;
  if (restPositions.has_value()) {
    restPositionsArr = py::array::ensure(restPositions.value());
  }

  if (checker.isFloat64()) {
    return skinSkinnedLocatorsImpl<double>(character, skelStateArr, restPositionsArr, leadingDims);
  } else {
    return skinSkinnedLocatorsImpl<float>(character, skelStateArr, restPositionsArr, leadingDims);
  }
}

} // namespace pymomentum
