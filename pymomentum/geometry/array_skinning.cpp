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

} // namespace

py::array skinPointsArray(
    const momentum::Character& character,
    py::buffer skelState,
    std::optional<py::buffer> restVertices) {
  MT_THROW_IF(
      !character.mesh || !character.skinWeights,
      "skin_points: character is missing a mesh or skin weights");

  const auto nVertices = static_cast<py::ssize_t>(character.mesh->vertices.size());

  // Use ArrayChecker to validate transforms and detect format/dtype
  ArrayChecker checker("skin_points");

  // Validate transforms (auto-detects skeleton state vs matrix format)
  checker.validateTransforms(skelState, "skel_state", character);

  // Validate rest vertices if provided
  // Note: rest_vertices can have fewer leading dimensions than skel_state (broadcasting)
  // So we validate it separately with just shape checks, not updating leadingDims
  if (restVertices.has_value()) {
    py::buffer_info restInfo = restVertices.value().request();

    MT_THROW_IF(
        restInfo.ndim < 2,
        "In skin_points, rest_vertices must have at least 2 dimensions, got {}",
        restInfo.ndim);

    MT_THROW_IF(
        restInfo.shape[restInfo.ndim - 1] != 3,
        "In skin_points, rest_vertices last dimension must be 3, got {}",
        restInfo.shape[restInfo.ndim - 1]);

    MT_THROW_IF(
        restInfo.shape[restInfo.ndim - 2] != nVertices,
        "In skin_points, rest_vertices second-to-last dimension must be {} (numVertices), got {}",
        nVertices,
        restInfo.shape[restInfo.ndim - 2]);

    // Validate dtype matches skel_state
    bool isRestFloat64 = (restInfo.format == py::format_descriptor<double>::format());
    bool isRestFloat32 = (restInfo.format == py::format_descriptor<float>::format());
    MT_THROW_IF(
        !isRestFloat64 && !isRestFloat32,
        "In skin_points, rest_vertices has unsupported dtype {}, expected float32 or float64",
        restInfo.format);
    MT_THROW_IF(
        isRestFloat64 != checker.isFloat64(),
        "In skin_points, rest_vertices dtype ({}) must match skel_state dtype ({})",
        isRestFloat64 ? "float64" : "float32",
        checker.isFloat64() ? "float64" : "float32");
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

} // namespace pymomentum
