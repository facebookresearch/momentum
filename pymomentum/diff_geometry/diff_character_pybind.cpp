/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/diff_geometry/diff_character_pybind.h"

#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"
#include "pymomentum/tensor_momentum/tensor_skinning.h"

#include <momentum/character/character.h>
#include <momentum/character/parameter_transform.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>

#include <utility>

namespace py = pybind11;
namespace mm = momentum;

using namespace pymomentum;

void defCharacterOperations(pybind11::module_& m) {
  // skin_points(character, skel_state, rest_vertices)
  m.def(
      "skin_points",
      &skinPoints,
      R"(Skins the points using the character's linear blend skinning.

:param character: The Character to use for skinning.
:param skel_state: A torch.Tensor containing either a [nBatch x nJoints x 8] skeleton state or a [nBatch x nJoints x 4 x 4] transforms.
:param rest_vertices: An optional torch.Tensor containing the rest points; if not passed, the ones stored inside the character are used.
:return: The vertex positions in worldspace.
      )",
      py::arg("character"),
      py::arg("skel_state"),
      py::arg("rest_vertices") = std::optional<at::Tensor>{});

  // skin_skinned_locators(character, skel_state, rest_positions)
  m.def(
      "skin_skinned_locators",
      &skinSkinnedLocators,
      R"(Apply linear blend skinning to compute the world-space positions of the character's skinned locators.

This function uses the character's built-in skinned locators and applies linear blend skinning
to compute their world-space positions given a skeleton state.

:param character: The Character to use.
:param skel_state: Skeleton state tensor with shape [nJoints x 8] or [nBatch x nJoints x 8].
:param rest_positions: Optional rest positions tensor with shape [nLocators x 3] or [nBatch x nLocators x 3]. If not provided, uses the position stored in each SkinnedLocator.
:return: Tensor of shape [nLocators x 3] or [nBatch x nLocators x 3] containing the world-space positions of the skinned locators.
)",
      py::arg("character"),
      py::arg("skel_state"),
      py::arg("rest_positions") = std::optional<at::Tensor>{});

  // apply_model_param_limits(character, model_params)
  m.def(
      "apply_model_param_limits",
      &applyModelParameterLimits,
      R"(Clamp model parameters by parameter limits stored in Character.

Note the function is differentiable.

:param character: The Character to use.
:param model_params: the (can be batched) body model parameters.
:return: clamped model parameters. Same tensor shape as the input.)",
      py::arg("character"),
      py::arg("model_params"));

  // reduce_to_selected_model_parameters(character, active_parameters)
  m.def(
      "reduce_to_selected_model_parameters",
      [](const mm::Character& character, at::Tensor activeParameters) {
        return character.simplifyParameterTransform(
            tensorToParameterSet(character.parameterTransform, std::move(activeParameters)));
      },
      R"(Strips out unused parameters from the parameter transform.

:param character: Full-body character.
:param active_parameters: A boolean tensor marking which parameters should be retained.
:return: A new character whose parameter transform only includes the marked parameters.)",
      py::arg("character"),
      py::arg("active_parameters"));
}
