/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/diff_geometry/diff_transform_pybind.h"

#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"
#include "pymomentum/torch_bridge.h"

#include <momentum/character/character.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

namespace py = pybind11;

using namespace pymomentum;

void defParameterTransforms(pybind11::module_& m) {
  // apply_parameter_transform(character, model_parameters)
  m.def(
      "apply_parameter_transform",
      [](py::object character, at::Tensor modelParameters) {
        return applyParamTransform(std::move(character), std::move(modelParameters));
      },
      R"(Apply the parameter transform to a [nBatch x nParams] tensor of model parameters.
This is functionally identical to :meth:`ParameterTransform.apply` except that it allows
batching on the character.

:param character: A character or list of characters.
:type character: Union[Character, List[Character]]
:param model_parameters: A [nBatch x nParams] tensor of model parameters.

:return: a tensor of joint parameters.)",
      py::arg("character"),
      py::arg("model_parameters"));

  // apply_inverse_parameter_transform(inverse_parameter_transform, joint_parameters)
  m.def(
      "apply_inverse_parameter_transform",
      [](const momentum::InverseParameterTransform* invParamTransform, at::Tensor jointParameters) {
        return applyInverseParamTransform(invParamTransform, std::move(jointParameters));
      },
      R"(Apply the inverse parameter transform to a [nBatch x nJointParams] tensor of joint parameters.
This is functionally identical to :meth:`InverseParameterTransform.apply` except that it supports
automatic differentiation.

:param inverse_parameter_transform: An inverse parameter transform (from ParameterTransform.inverse()).
:type inverse_parameter_transform: InverseParameterTransform
:param joint_parameters: A [nBatch x nJointParams] tensor of joint parameters.

:return: a tensor of model parameters.)",
      py::arg("inverse_parameter_transform"),
      py::arg("joint_parameters"));

  // model_parameters_to_blend_shape_coefficients(character, model_parameters)
  m.def(
      "model_parameters_to_blend_shape_coefficients",
      &modelParametersToBlendShapeCoefficients,
      R"(Extract the model parameters that correspond to the blend shape coefficients, in the order
required to call `meth:BlendShape.compute_shape`.

:param character: A character.
:parameter model_parameters: A [nBatch x nParams] tensor of model parameters.

:return: A [nBatch x nBlendShape] torch.Tensor of blend shape coefficients.)",
      py::arg("character"),
      py::arg("model_parameters"));

  // model_parameters_to_face_expression_coefficients(character, model_parameters)
  m.def(
      "model_parameters_to_face_expression_coefficients",
      &modelParametersToFaceExpressionCoefficients,
      R"(Extract the model parameters that correspond to the face expression coefficients.

:param character: A character.
:parameter model_parameters: A [nBatch x nParams] tensor of model parameters.

:return: A [nBatch x nFaceExpressions] torch.Tensor of face expression coefficients.)",
      py::arg("character"),
      py::arg("model_parameters"));

  // uniform_random_to_model_parameters(character, unif_noise)
  m.def(
      "uniform_random_to_model_parameters",
      &uniformRandomToModelParameters,
      R"(Convert a uniform noise vector into a valid body pose.

:parameter character: The character to use.
:parameter unif_noise: A uniform noise tensor, with dimensions (nBatch x nModelParams).
:return: A torch.Tensor with dimensions (nBatch x nModelParams).)",
      py::arg("character"),
      py::arg("unif_noise"));

  // map_model_parameters(motion_data, source_parameter_names, target_character)
  m.def(
      "map_model_parameters",
      &mapModelParameters_names,
      R"(Remap model parameters from one character to another.

:param motion_data: The source motion data as a nFrames x nParams torch.Tensor.
:param source_parameter_names: The source parameter names as a list of strings (e.g. c.parameterTransform.name).
:param target_character: The target character to remap onto.

:return: The motion with the parameters remapped to match the passed-in Character. The fields with no match are filled with zero.
      )",
      py::arg("motion_data"),
      py::arg("source_parameter_names"),
      py::arg("target_character"),
      py::arg("verbose") = false);

  // map_model_parameters(motion_data, source_character, target_character)
  m.def(
      "map_model_parameters",
      &mapModelParameters,
      R"(Remap model parameters from one character to another.

:param motion_data: The source motion data as a nFrames x nParams torch.Tensor.
:param source_character: The source character.
:param target_character: The target character to remap onto.
:param verbose: If true, print out warnings about missing parameters.

:return: The motion with the parameters remapped to match the passed-in Character. The fields with no match are filled with zero.
      )",
      py::arg("motion_data"),
      py::arg("source_character"),
      py::arg("target_character"),
      py::arg("verbose") = true);

  // map_joint_parameters(motion_data, source_character, target_character)
  m.def(
      "map_joint_parameters",
      &mapJointParameters,
      R"(Remap joint parameters from one character to another.

:param motion_data: The source motion data as a [nFrames x (nBones * 7)] torch.Tensor.
:param source_character: The source character.
:param target_character: The target character to remap onto.

:return: The motion with the parameters remapped to match the passed-in Character. The fields with no match are filled with zero.
      )",
      py::arg("motion_data"),
      py::arg("source_character"),
      py::arg("target_character"));
}
