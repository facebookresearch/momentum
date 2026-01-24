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
    const py::array& jointParams,
    const IntScalarArrayAccessor& parentsAcc,
    const py::array& offsets,
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

  // Cast offsets to numpy array for validation
  py::array offsetsArr = py::array::ensure(offsets);

  // Get parents array info for nPoints
  py::buffer_info parentsBuf = parents.request();
  MT_THROW_IF(parentsBuf.ndim != 1, "parents must be a 1D array");
  const auto nPoints = static_cast<py::ssize_t>(parentsBuf.shape[0]);

  // Create type-erased accessor for parents (handles int32, int64, uint32, uint64)
  IntScalarArrayAccessor parentsAcc(parents, nPoints);

  // Validate offsets
  py::buffer_info offsetsBuf = offsetsArr.request();
  MT_THROW_IF(offsetsBuf.ndim != 2, "offsets must be a 2D array of shape [nPoints, 3]");
  MT_THROW_IF(
      offsetsBuf.shape[0] != nPoints || offsetsBuf.shape[1] != 3,
      "offsets must have shape [nPoints, 3], got [{}, {}]",
      offsetsBuf.shape[0],
      offsetsBuf.shape[1]);

  if (checker.isFloat64()) {
    // Convert offsets to float64 if needed
    py::array_t<double> offsetsF64 = py::array_t<double, py::array::c_style>::ensure(offsetsArr);
    return jointParametersToPositionsImpl<double>(
        character,
        jointParameters,
        parentsAcc,
        offsetsF64,
        checker.getLeadingDimensions(),
        nPoints,
        shape);
  } else {
    // Convert offsets to float32 if needed
    py::array_t<float> offsetsF32 = py::array_t<float, py::array::c_style>::ensure(offsetsArr);
    return jointParametersToPositionsImpl<float>(
        character,
        jointParameters,
        parentsAcc,
        offsetsF32,
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
