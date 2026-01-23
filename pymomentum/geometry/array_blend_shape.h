/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/blend_shape.h>
#include <momentum/character/blend_shape_base.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pymomentum {

namespace py = pybind11;

/// Computes the blend shape output by applying blend shape coefficients.
///
/// For BlendShape objects, this computes: base_shape + sum(coefficients[i] * shape_vectors[i])
/// For BlendShapeBase objects, this computes: sum(coefficients[i] * shape_vectors[i])
///
/// Supports arbitrary leading dimensions with broadcasting and both float32/float64 dtypes.
///
/// @param blendShape A BlendShapeBase or BlendShape object
/// @param coefficients Numpy array containing blend shape coefficients.
///   Shape: [..., numCoefficients]
/// @return Numpy array containing the computed vertex positions.
///   Shape: [..., numVertices, 3]
py::array computeBlendShapeArray(
    const momentum::BlendShapeBase& blendShape,
    const py::buffer& coefficients);

} // namespace pymomentum
