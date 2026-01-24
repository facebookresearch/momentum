/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/array_joint_parameters_to_positions.h"

#include "pymomentum/array_utility/array_utility.h"
#include "pymomentum/array_utility/batch_accessor.h"
#include "pymomentum/array_utility/geometry_accessors.h"
#include "pymomentum/geometry/array_parameter_transform.h"

#include <momentum/character/joint.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/common/exception.h>

#include <dispenso/parallel_for.h>

namespace pymomentum {

namespace {

template <typename T>
py::array_t<T> jointParametersToPositionsImpl(
    const momentum::Character& character,
    const py::buffer& jointParams,
    const IntScalarArrayAccessor& parentsAcc,
    const py::buffer& offsets,
    const LeadingDimensions& leadingDims,
    py::ssize_t nPoints,
    JointParamsShape shape) {
  const auto nJoints = static_cast<py::ssize_t>(character.skeleton.joints.size());
  const auto nBatch = leadingDims.totalBatchElements();

  // Validate all parent indices upfront using minmax()
  if (nPoints > 0) {
    const auto [minIdx, maxIdx] = parentsAcc.minmax();
    MT_THROW_IF(minIdx < 0, "Invalid parent joint index: minimum value {} is negative", minIdx);
    MT_THROW_IF(
        maxIdx >= static_cast<int>(nJoints),
        "Invalid parent joint index: maximum value {} >= number of joints ({})",
        maxIdx,
        nJoints);
  }

  // Create output array with shape [..., nPoints, 3]
  auto result = createOutputArray<T>(leadingDims, {nPoints, static_cast<py::ssize_t>(3)});

  // Create batch indexer for converting flat indices to multi-dimensional indices
  BatchIndexer indexer(leadingDims);

  // Create accessors using geometry accessors
  JointParametersAccessor<T> inputAcc(jointParams, leadingDims, nJoints, shape);
  VectorArrayAccessor<T, 3> outputAcc(result, leadingDims, nPoints);

  // Offsets are not batched, so use empty leading dimensions
  LeadingDimensions emptyLeadingDims;
  VectorArrayAccessor<T, 3> offsetsAcc(offsets, emptyLeadingDims, nPoints);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get joint parameters as JointParametersT<T>
      auto jp = inputAcc.get(indices);

      // Compute skeleton state (global transforms)
      const momentum::SkeletonStateT<T> skelState(jp, character.skeleton, /*computeDeriv=*/false);

      // Get output view for this batch element
      auto outView = outputAcc.view(indices);

      // Get offsets view (not batched, so use empty indices)
      std::vector<py::ssize_t> emptyIndices;
      auto offsetsView = offsetsAcc.view(emptyIndices);

      // Compute world-space position for each point
      for (py::ssize_t iPoint = 0; iPoint < nPoints; ++iPoint) {
        // Parent indices are already validated via minmax()
        const int parent = parentsAcc.get(iPoint);

        // Get the offset for this point using the accessor
        const Eigen::Vector3<T> offset = offsetsView.get(iPoint);

        // Transform the offset from local space to world space
        const Eigen::Vector3<T> p_world = skelState.jointState[parent].transform * offset;

        // Write the result
        outView.set(iPoint, p_world);
      }
    });
  }

  return result;
}

} // namespace

py::array jointParametersToPositionsArray(
    const momentum::Character& character,
    const py::buffer& jointParameters,
    const py::buffer& parents,
    const py::buffer& offsets) {
  ArrayChecker checker("joint_parameters_to_positions");
  JointParamsShape shape =
      checker.validateJointParameters(jointParameters, "joint_parameters", character);

  // Get nPoints from parents buffer
  py::buffer_info parentsBuf = parents.request();
  MT_THROW_IF(
      parentsBuf.ndim != 1,
      "joint_parameters_to_positions: parents must be a 1D array, got {}D",
      parentsBuf.ndim);
  const auto nPoints = static_cast<py::ssize_t>(parentsBuf.shape[0]);

  // Create type-erased accessor for parents (handles int32, int64, uint32, uint64)
  // This validates the dtype and provides on-the-fly conversion to int
  IntScalarArrayAccessor parentsAcc(parents, nPoints);

  // Validate offsets buffer: [nPoints, 3] array of floats (must match nPoints from parents)
  checker.validateBuffer(offsets, "offsets", {static_cast<int>(nPoints), 3}, {"numPoints", "3"});

  if (checker.isFloat64()) {
    return jointParametersToPositionsImpl<double>(
        character,
        jointParameters,
        parentsAcc,
        offsets,
        checker.getLeadingDimensions(),
        nPoints,
        shape);
  } else {
    return jointParametersToPositionsImpl<float>(
        character,
        jointParameters,
        parentsAcc,
        offsets,
        checker.getLeadingDimensions(),
        nPoints,
        shape);
  }
}

py::array modelParametersToPositionsArray(
    const momentum::Character& character,
    const py::buffer& modelParameters,
    const py::buffer& parents,
    const py::buffer& offsets) {
  // First apply parameter transform to get joint parameters
  py::array jointParams =
      applyParameterTransformArray(character.parameterTransform, modelParameters);

  // Then convert joint parameters to positions
  return jointParametersToPositionsArray(character, jointParams, parents, offsets);
}

} // namespace pymomentum
