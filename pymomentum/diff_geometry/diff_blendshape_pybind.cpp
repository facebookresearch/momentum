/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/diff_geometry/diff_blendshape_pybind.h"

#include "pymomentum/tensor_momentum/tensor_blend_shape.h"
#include "pymomentum/tensor_momentum/tensor_skinning.h"

#include <momentum/character/blend_shape.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>

namespace py = pybind11;

using namespace pymomentum;

void defBlendShapeOperations(pybind11::module_& m) {
  // compute_blend_shape(blend_shape, coeffs)
  m.def(
      "compute_blend_shape",
      [](py::object blendShape, at::Tensor coeffs) {
        return applyBlendShapeCoefficients(std::move(blendShape), std::move(coeffs));
      },
      R"(Apply the blend shape coefficients to compute the rest shape.

The resulting shape is equal to a linear combination of the shape vectors (plus the base shape, if it is a BlendShape object).

:param blend_shape: A BlendShape or BlendShapeBase object.
:param coeffs: A torch.Tensor of size [n_batch x n_shapes] containing blend shape coefficients.
:result: A [n_batch x n_vertices x 3] tensor containing the vertex positions.)",
      py::arg("blend_shape"),
      py::arg("coeffs"));

  // compute_vertex_normals(vertex_positions, triangles)
  m.def(
      "compute_vertex_normals",
      &computeVertexNormals,
      R"(
Computes vertex normals for a triangle mesh given its positions.

:param vertex_positions: [nBatch] x nVert x 3 Tensor of vertex positions.
:param triangles: nTriangles x 3 Tensor of triangle indices.
:return: Smooth per-vertex normals.
    )",
      py::arg("vertex_positions"),
      py::arg("triangles"));
}
