/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/math/mppca.h>
#include <momentum/test/character/character_helpers.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace mm = momentum;

namespace {
mm::Character createTestCharacter(int num_joints, bool with_blendshapes) {
  auto character = momentum::createTestCharacter<float>(num_joints);
  if (with_blendshapes) {
    character = momentum::withTestBlendShapes(character);
  }
  return character;
}
} // namespace

PYBIND11_MODULE(geometry_test_utils, m) {
  m.doc() =
      "Test utilities for pymomentum.  Provides simple test characters and "
      "pose priors for writing unit tests that don't depend on external files.";
  m.attr("__name__") = "pymomentum.geometry_test_utils";

  m.def(
      "create_test_character",
      &createTestCharacter,
      R"(Create a simple 3-joint test character with blendshapes.  This is useful for writing
confidence tests that execute quickly and don't rely on outside files.

The mesh is made by a few vertices on the line segment from (1,0,0) to (1,1,0) and a few dummy
faces. The skeleton has three joints: root at (0,0,0), joint1 parented by root, at world-space
(0,1,0), and joint2 parented by joint1, at world-space (0,2,0).
The character has only one parameter limit: min-max type [-0.1, 0.1] for root.

:parameter numJoints: The number of joints in the resulting character.
:return: A simple character with 3 joints, 10 model parameters, and 5 blendshapes.
      )",
      py::arg("num_joints") = 3,
      py::arg("with_blendshapes") = false);

  m.def(
      "create_test_mppca",
      &momentum::createDefaultPosePrior<float>,
      R"(Create a pose prior that acts on the simple 3-joint test character.

:return: A simple pose prior.)");
}
