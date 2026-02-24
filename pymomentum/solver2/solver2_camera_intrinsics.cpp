/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/solver2/solver2_camera_intrinsics.h"

#include <momentum/camera/camera.h>
#include <momentum/character/character.h>
#include <momentum/character_solver/camera_intrinsics_parameters.h>

#include <pymomentum/array_utility/array_utility.h>
#include <pymomentum/array_utility/batch_accessor.h>
#include <pymomentum/array_utility/geometry_accessors.h>

#include <dispenso/parallel_for.h> // @manual
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

namespace {

// Template implementation for extracting camera intrinsics from model parameters.
// Supports batched arrays with shape (..., nModelParams).
template <typename T>
py::array_t<T> extractCameraIntrinsicsImpl(
    const mm::ParameterTransform& paramTransform,
    const py::buffer& modelParams,
    const LeadingDimensions& leadingDims,
    py::ssize_t nModelParams,
    const mm::IntrinsicsModel& intrinsicsModel) {
  const auto nIntrinsicParams = static_cast<py::ssize_t>(intrinsicsModel.numIntrinsicParameters());
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array with shape (..., nIntrinsicParams)
  auto result = createOutputArray<T>(leadingDims, {nIntrinsicParams});

  // Create batch indexer
  BatchIndexer indexer(leadingDims);

  // Create accessor for input model parameters
  ModelParametersAccessor<T> inputAcc(modelParams, leadingDims, nModelParams);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      auto indices = indexer.decompose(iBatch);

      // Get model parameters for this batch element
      auto mp = inputAcc.get(indices);

      // Extract camera intrinsics — returns Eigen::VectorXf regardless of T,
      // so we always call with float ModelParameters
      mm::ModelParameters mpFloat(mp.v.template cast<float>());
      Eigen::VectorXf intrinsics =
          mm::extractCameraIntrinsics(paramTransform, mpFloat, intrinsicsModel);

      // Write output element-by-element (output array is contiguous from createOutputArray)
      auto* outPtr = result.mutable_data();
      const auto outOffset = iBatch * nIntrinsicParams;
      for (py::ssize_t i = 0; i < nIntrinsicParams; ++i) {
        outPtr[outOffset + i] = static_cast<T>(intrinsics(i));
      }
    });
  }

  return result;
}

// Template implementation for setting camera intrinsics into model parameters.
// Returns a new array instead of mutating in-place.
// Supports batched arrays with shape (..., nModelParams).
template <typename T>
py::array_t<T> setCameraIntrinsicsImpl(
    const mm::ParameterTransform& paramTransform,
    const mm::IntrinsicsModel& intrinsicsModel,
    const py::buffer& modelParams,
    const LeadingDimensions& leadingDims,
    py::ssize_t nModelParams) {
  const auto nBatch = leadingDims.totalBatchElements();

  // Create output array with same shape as input (..., nModelParams)
  auto result = createOutputArray<T>(leadingDims, {nModelParams});

  // Create batch indexer
  BatchIndexer indexer(leadingDims);

  // Create accessors
  ModelParametersAccessor<T> inputAcc(modelParams, leadingDims, nModelParams);
  ModelParametersAccessor<T> outputAcc(result, leadingDims, nModelParams);

  // Release GIL for parallel computation
  {
    py::gil_scoped_release release;
    dispenso::parallel_for(0, static_cast<int64_t>(nBatch), [&](int64_t iBatch) {
      auto indices = indexer.decompose(iBatch);

      // Get input model parameters
      auto mp = inputAcc.get(indices);

      // setCameraIntrinsics works with float ModelParameters
      mm::ModelParameters mpFloat(mp.v.template cast<float>());
      mm::setCameraIntrinsics(paramTransform, intrinsicsModel, mpFloat);

      // Convert back and write to output
      mm::ModelParametersT<T> mpOut(mpFloat.v.template cast<T>());
      outputAcc.set(indices, mpOut);
    });
  }

  return result;
}

} // namespace

void addCameraIntrinsicsBindings(pybind11::module_& m) {
  m.def(
      "add_camera_intrinsics_parameters",
      [](const mm::Character& character,
         const mm::IntrinsicsModel& intrinsicsModel) -> mm::Character {
        auto [paramTransform, paramLimits] = mm::addCameraIntrinsicsParameters(
            character.parameterTransform, character.parameterLimits, intrinsicsModel);
        mm::Character result = character;
        result.parameterTransform = std::move(paramTransform);
        result.parameterLimits = std::move(paramLimits);
        return result;
      },
      R"(Add camera intrinsics parameters to the character's parameter transform.

Creates a new :class:`~pymomentum.geometry.Character` with additional model parameters for the
given camera's intrinsic parameters (e.g., fx, fy, cx, cy, distortion coefficients). The parameter
names follow the convention "intrinsics_{camera_name}_{param_name}".

The intrinsics model must have a non-empty name set via the ``name`` property.

This operation is idempotent: calling it twice with the same camera name will
replace the existing parameters rather than duplicating them.

:param character: The character whose parameter transform to augment.
:param intrinsics_model: The camera intrinsics model whose parameters to add (must have a name set).
:return: A new Character with the augmented parameter transform.)",
      py::arg("character"),
      py::arg("intrinsics_model"));

  m.def(
      "extract_camera_intrinsics",
      [](const mm::Character& character,
         const py::buffer& modelParameters,
         const mm::IntrinsicsModel& intrinsicsModel) -> py::array {
        const auto& pt = character.parameterTransform;
        const auto nModelParams = static_cast<int>(pt.numAllModelParameters());

        ArrayChecker checker("extract_camera_intrinsics");
        checker.validateBuffer(
            modelParameters, "model_parameters", {nModelParams}, {"nModelParams"});

        const auto& leadingDims = checker.getLeadingDimensions();

        if (checker.isFloat64()) {
          return extractCameraIntrinsicsImpl<double>(
              pt, modelParameters, leadingDims, nModelParams, intrinsicsModel);
        } else {
          return extractCameraIntrinsicsImpl<float>(
              pt, modelParameters, leadingDims, nModelParams, intrinsicsModel);
        }
      },
      R"(Extract camera intrinsic parameter values from model parameters.

Reads the values corresponding to the camera's intrinsics from the model parameter
vector. Parameters not present in the parameter transform retain the current values
from the intrinsics model (useful when working with a downselected parameter set).

Supports batched input with shape ``(..., n_model_params)``.

:param character: The character with camera intrinsics in its parameter transform.
:param model_parameters: The model parameter array with shape ``(..., n_model_params)``.
:param intrinsics_model: The intrinsics model (provides name and default values).
:return: Array of intrinsic parameters with shape ``(..., n_intrinsic_params)`` suitable
    for :meth:`~pymomentum.camera.IntrinsicsModel.set_intrinsic_parameters`.)",
      py::arg("character"),
      py::arg("model_parameters"),
      py::arg("intrinsics_model"));

  m.def(
      "set_camera_intrinsics",
      [](const mm::Character& character,
         const mm::IntrinsicsModel& intrinsicsModel,
         const py::buffer& modelParameters) -> py::array {
        const auto& pt = character.parameterTransform;
        const auto nModelParams = static_cast<int>(pt.numAllModelParameters());

        ArrayChecker checker("set_camera_intrinsics");
        checker.validateBuffer(
            modelParameters, "model_parameters", {nModelParams}, {"nModelParams"});

        const auto& leadingDims = checker.getLeadingDimensions();

        if (checker.isFloat64()) {
          return setCameraIntrinsicsImpl<double>(
              pt, intrinsicsModel, modelParameters, leadingDims, nModelParams);
        } else {
          return setCameraIntrinsicsImpl<float>(
              pt, intrinsicsModel, modelParameters, leadingDims, nModelParams);
        }
      },
      R"(Write camera intrinsic parameter values into model parameters.

Copies the intrinsics model's current parameter values into the corresponding
slots of the model parameter vector, returning a new array. Parameters not
present in the parameter transform are silently skipped.

Supports batched input with shape ``(..., n_model_params)``.

:param character: The character with camera intrinsics in its parameter transform.
:param intrinsics_model: The intrinsics model to read values from.
:param model_parameters: The model parameter array with shape ``(..., n_model_params)``.
:return: A new model parameter array with the intrinsics values written in.)",
      py::arg("character"),
      py::arg("intrinsics_model"),
      py::arg("model_parameters"));
}

} // namespace pymomentum
