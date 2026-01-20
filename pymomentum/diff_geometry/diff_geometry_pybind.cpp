/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/diff_geometry/diff_geometry_pybind.h"
#include "pymomentum/diff_geometry/diff_blendshape_pybind.h"
#include "pymomentum/diff_geometry/diff_character_pybind.h"
#include "pymomentum/diff_geometry/diff_transform_pybind.h"

#include "pymomentum/tensor_momentum/tensor_joint_parameters_to_positions.h"
#include "pymomentum/tensor_momentum/tensor_kd_tree.h"
#include "pymomentum/tensor_momentum/tensor_skeleton_state.h"
#include "pymomentum/torch_bridge.h"

#include <momentum/character/character.h>
#include <momentum/character/character_utility.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>

namespace py = pybind11;

using namespace pymomentum;

void defForwardKinematics(pybind11::module_& m) {
  // modelParametersToPositions(character, modelParameters, parents, offsets)
  m.def(
      "model_parameters_to_positions",
      &modelParametersToPositions,
      R"(Convert model parameters to 3D positions relative to skeleton joints using forward kinematics.  You can use this (for example) to
supervise a model to produce the correct 3D ground truth.

Working directly from modelParameters is preferable to mapping to jointParameters first because it does a better job exploiting the
sparsity in the model and therefore can be made somewhat faster.

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param model_parameters: Model parameter tensor, with dimension (nBatch x nModelParams).
:param parents: Joint parents, on for each target position.
:param offsets: 3-d offset in each joint's local space.
:return: Tensor of size (nBatch x nParents x 3), representing the world-space position of each point.
)",
      py::arg("character"),
      py::arg("model_parameters"),
      py::arg("parents"),
      py::arg("offsets"));

  // jointParametersToPositions(character, jointParameters, parents, offsets)
  m.def(
      "joint_parameters_to_positions",
      &jointParametersToPositions,
      R"(Convert joint parameters to 3D positions relative to skeleton joints using forward kinematics.  You can use this (for example) to
supervise a model to produce the correct 3D ground truth.

You should prefer :meth:`model_parameters_to_positions` when working from modelParameters because it is better able to exploit sparsity; this
function is provided as a convenience because motion read from external files generally uses jointParameters.

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param joint_parameters: Joint parameter tensor, with dimension (nBatch x (7*nJoints)).
:param parents: Joint parents, on for each target position.
:param offsets: 3-d offset in each joint's local space.
:return: Tensor of size (nBatch x nParents x 3), representing the world-space position of each point.
)",
      py::arg("character"),
      py::arg("joint_parameters"),
      py::arg("parents"),
      py::arg("offsets"));

  // modelParametersToSkeletonState(characters, modelParameters)
  m.def(
      "model_parameters_to_skeleton_state",
      &modelParametersToSkeletonState,
      R"(Map from the k modelParameters to the 8*nJoints global skeleton state.

The skeletonState is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to worldspace.
Rotations are Quaternions in the ((x, y, z), w) format.  This is deliberately identical to the representation used in a legacy format.)

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param model_parameters: torch.Tensor containing the (nBatch x nModelParameters) model parameters.

:return: torch.Tensor of size (nBatch x nJoints x 8) containing the skeleton state; should be also compatible with a legacy format's skeleton state representation.)",
      py::arg("character"),
      py::arg("model_parameters"));

  // modelParametersToLocalSkeletonState(characters, modelParameters)
  m.def(
      "model_parameters_to_local_skeleton_state",
      &modelParametersToLocalSkeletonState,
      R"(Map from the k modelParameters to the 8*nJoints local skeleton state.

The skeletonState is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to its parent joint space.
Rotations are Quaternions in the ((x, y, z), w) format.  This is deliberately identical to the representation used in a legacy format.)

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param model_parameters: torch.Tensor containing the (nBatch x nModelParameters) model parameters.

:return: torch.Tensor of size (nBatch x nJoints x 8) containing the skeleton state; should be also compatible with a legacy format's skeleton state representation.)",
      py::arg("character"),
      py::arg("model_parameters"));

  // jointParametersToSkeletonState(character, jointParameters)
  m.def(
      "joint_parameters_to_skeleton_state",
      &jointParametersToSkeletonState,
      R"(Map from the 7*nJoints jointParameters to the 8*nJoints global skeleton state.

The skeletonState is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to worldspace.
Rotations are Quaternions in the ((x, y, z), w) format.  This is deliberately identical to the representation used in a legacy format.)

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param joint_parameters: torch.Tensor containing the (nBatch x nJointParameters) joint parameters.

:return: torch.Tensor of size (nBatch x nJoints x 8) containing the skeleton state; should be also compatible with a legacy format's skeleton state representation.)",
      py::arg("character"),
      py::arg("joint_parameters"));

  // jointParametersToLocalSkeletonState(character, jointParameters)
  m.def(
      "joint_parameters_to_local_skeleton_state",
      &jointParametersToLocalSkeletonState,
      R"(Map from the 7*nJoints jointParameters (representing transforms to the parent joint) to the 8*nJoints local skeleton state.

The skeletonState is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to its parent joint space.
Rotations are Quaternions in the ((x, y, z), w) format.  This is deliberately identical to the representation used in a legacy format.)

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param joint_parameters: torch.Tensor containing the (nBatch x nJointParameters) joint parameters.

:return: torch.Tensor of size (nBatch x nJoints x 8) containing the skeleton state; should be also compatible with a legacy format's skeleton state representation.)",
      py::arg("character"),
      py::arg("joint_parameters"));

  // skeletonStateToJointParameters(character, skelState)
  m.def(
      "skeleton_state_to_joint_parameters",
      &skeletonStateToJointParameters,
      R"(Map from the 8*nJoints skeleton state (representing transforms to world-space) to the 7*nJoints jointParameters.  This performs the following operations:

* Removing the translation offset.
* Inverting out the pre-rotation.
* Converting to Euler angles.

The skeleton state is stored (tx, ty, tz; rx, ry, rz, rw; s) and transforms from the joint's local space to world-space.
The joint parameters are stored (tx, ty, tz; ry, rz, ry; s) where rotations are in Euler angles and are relative to the parent joint.

:param character: Character to use.
:param skel_state: torch.Tensor containing the ([nBatch] x nJoints x 8) skeleton state.

:return: torch.Tensor of size ([nBatch] x nJoints x 7) containing the joint parameters.)",
      py::arg("character"),
      py::arg("skel_state"));

  // localSkeletonStateToJointParameters(character, skelState)
  m.def(
      "local_skeleton_state_to_joint_parameters",
      &localSkeletonStateToJointParameters,
      R"(Map from the 8*nJoints local skeleton state to the 7*nJoints jointParameters.  This performs the following operations:

* Removing the translation offset.
* Inverting out the pre-rotation.
* Converting to Euler angles.

The local skeleton state is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to its parent joint space.
The joint parameters are stored (tx, ty, tz; ry, rz, ry; s) where rotations are in Euler angles and are relative to the parent joint.

:param character: Character to use.
:param local_skel_state: torch.Tensor containing the ([nBatch] x nJoints x 8) skeleton state.

:return: torch.Tensor of size ([nBatch] x nJoints x 7) containing the joint parameters.)",
      py::arg("character"),
      py::arg("local_skel_state"));
}

void defUtilityFunctions(pybind11::module_& m) {
  m.def(
      "find_closest_points",
      &findClosestPoints,
      pybind11::call_guard<py::gil_scoped_release>(),
      R"(For each point in the points_source tensor, find the closest point in the points_target tensor.  This version of find_closest points supports both 2- and 3-dimensional point sets.

:param points_source: [nBatch x nPoints x dim] tensor of source points (dim must be 2 or 3).
:param points_target: [nBatch x nPoints x dim] tensor of target points (dim must be 2 or 3).
:param max_dist: Maximum distance to search, can be used to speed up the method by allowing the search to return early.  Defaults to FLT_MAX.
:return: A tuple of three tensors.  The first is [nBatch x nPoints x dim] and contains the closest point for each point in the target set.
         The second is [nBatch x nPoints] and contains the index of each closest point in the target set (or -1 if none).
         The third is [nBatch x nPoints] and is a boolean tensor indicating whether a valid closest point was found for each source point.
      )",
      py::arg("points_source"),
      py::arg("points_target"),
      py::arg("max_dist") = std::numeric_limits<float>::max());

  m.def(
      "find_closest_points",
      &findClosestPointsWithNormals,
      pybind11::call_guard<py::gil_scoped_release>(),
      R"(For each point in the points_source tensor, find the closest point in the points_target tensor whose normal is compatible (n_source . n_target > max_normal_dot).
Using the normal is a good way to avoid certain kinds of bad matches, such as matching the front of the body against depth values from the back of the body.

:param points_source: [nBatch x nPoints x 3] tensor of source points.
:param normals_source: [nBatch x nPoints x 3] tensor of source normals (must be normalized).
:param points_target: [nBatch x nPoints x 3] tensor of target points.
:param normals_target: [nBatch x nPoints x 3] tensor of target normals (must be normalized).
:param max_dist: Maximum distance to search, can be used to speed up the method by allowing the search to return early.  Defaults to FLT_MAX.
:param max_normal_dot: Maximum dot product allowed between the source and target normal.  Defaults to 0.
:return: A tuple of three tensors.  The first is [nBatch x nPoints x dim] and contains the closest point for each point in the target set.
         The second is [nBatch x nPoints] and contains the index of each closest point in the target set (or -1 if none).
         The third is [nBatch x nPoints] and is a boolean tensor indicating whether a valid closest point was found for each source point.
      )",
      py::arg("points_source"),
      py::arg("normals_source"),
      py::arg("points_target"),
      py::arg("normals_target"),
      py::arg("max_dist") = std::numeric_limits<float>::max(),
      py::arg("max_normal_dot") = 0.0f);

  m.def(
      "find_closest_points_on_mesh",
      &findClosestPointsOnMesh,
      pybind11::call_guard<py::gil_scoped_release>(),
      R"(For each point in the points_source tensor, find the closest point in the target mesh.

:param points_source: [nBatch x nPoints x 3] tensor of source points.
:param vertices_target: [nBatch x nVerts x 3] tensor of target mesh vertices.
:param faces_target: [nFaces x 3] tensor of target mesh triangle indices.
:return: A tuple of three tensors.  The first is [nBatch x nPoints x 3] and contains the closest point on the target mesh.
         The second is [nBatch x nPoints] and contains the index of the closest triangle (-1 if none).
         The third is [nBatch x nPoints] and is a boolean tensor indicating whether a valid closest point was found.
      )",
      py::arg("points_source"),
      py::arg("vertices_target"),
      py::arg("faces_target"));
}

PYBIND11_MODULE(diff_geometry, m) {
  m.doc() =
      "Differentiable geometry and forward kinematics for momentum models using PyTorch tensors.";
  m.attr("__name__") = "pymomentum.diff_geometry";

#ifdef PYMOMENTUM_LIMITED_TORCH_API
  m.attr("AUTOGRAD_ENABLED") = false;
#else
  m.attr("AUTOGRAD_ENABLED") = true;
#endif

  pybind11::module_::import("torch"); // @dep=//caffe2:torch
  pybind11::module_::import("pymomentum.geometry"); // @dep=//pymomentum:geometry

  // Define forward kinematics functions
  defForwardKinematics(m);

  // Define character operations
  defCharacterOperations(m);

  // Define parameter transform operations
  defParameterTransforms(m);

  // Define blend shape operations
  defBlendShapeOperations(m);

  // Define utility functions
  defUtilityFunctions(m);
}
