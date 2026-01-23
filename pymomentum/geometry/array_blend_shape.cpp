/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/array_blend_shape.h"

#include <pymomentum/array_utility/array_utility.h>
#include <pymomentum/array_utility/batch_accessor.h>
#include <pymomentum/array_utility/geometry_accessors.h>

#include <momentum/character/blend_shape.h>
#include <momentum/character/blend_shape_base.h>
#include <momentum/common/exception.h>

#include <dispenso/parallel_for.h>

namespace pymomentum {

namespace {

template <typename T>
py::array_t<T> computeBlendShapeImpl(
    const momentum::BlendShapeBase& blendShapeBase,
    const py::array& coefficients,
    const LeadingDimensions& leadingDims) {
  const auto nCoeffs = static_cast<py::ssize_t>(coefficients.shape(coefficients.ndim() - 1));
  const auto nVertices = static_cast<py::ssize_t>(blendShapeBase.modelSize());
  const auto nBatch = leadingDims.totalBatchElements();

  MT_THROW_IF(
      nCoeffs > blendShapeBase.shapeSize(),
      "computeBlendShape: expected at most {} coefficients but got {}.",
      blendShapeBase.shapeSize(),
      nCoeffs);

  // Create output array with shape [..., nVertices, 3]
  auto result = createOutputArray<T>(leadingDims, {nVertices, static_cast<py::ssize_t>(3)});

  // Create batch indexer
  BatchIndexer indexer(leadingDims);

  // Create accessors
  BlendWeightsAccessor<T> inputAcc(coefficients, leadingDims, nCoeffs);
  VertexPositionsAccessor<T> outputAcc(result, leadingDims, nVertices);

  // Check if this is a BlendShape (has base shape) or just BlendShapeBase
  const auto* blendShape = dynamic_cast<const momentum::BlendShape*>(&blendShapeBase);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      // Convert flat batch index to multi-dimensional indices
      auto indices = indexer.decompose(iBatch);

      // Get blend weights
      auto blendWeights = inputAcc.get(indices);

      // Compute the shape
      std::vector<Eigen::Vector3<T>> positions;
      if (blendShape != nullptr) {
        // BlendShape: computes base_shape + weighted sum of shape vectors
        positions = blendShape->template computeShape<T>(blendWeights);
      } else {
        // BlendShapeBase: computes only the weighted sum of shape vectors (deltas)
        positions.resize(nVertices, Eigen::Vector3<T>::Zero());
        blendShapeBase.template applyDeltas<T>(blendWeights, positions);
      }

      // Write output
      outputAcc.set(indices, positions);
    });
  }

  return result;
}

} // namespace

py::array computeBlendShapeArray(
    const momentum::BlendShapeBase& blendShape,
    const py::buffer& coefficients) {
  const auto nShapes = static_cast<int>(blendShape.shapeSize());

  ArrayChecker checker("compute_shape");
  // Use -1 for variable number of coefficients (can be less than or equal to nShapes)
  checker.validateBuffer(coefficients, "coefficients", {-1}, {"numCoefficients"});

  const auto nCoeffs = checker.getBoundValue(-1);
  MT_THROW_IF(
      nCoeffs > nShapes,
      "compute_shape: expected at most {} coefficients but got {}.",
      nShapes,
      nCoeffs);

  if (checker.isFloat64()) {
    return computeBlendShapeImpl<double>(blendShape, coefficients, checker.getLeadingDimensions());
  } else {
    return computeBlendShapeImpl<float>(blendShape, coefficients, checker.getLeadingDimensions());
  }
}

} // namespace pymomentum
