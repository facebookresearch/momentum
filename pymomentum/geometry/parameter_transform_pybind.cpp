/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/parameter_transform_pybind.h"

#include "pymomentum/geometry/array_parameter_transform.h"
#include "pymomentum/geometry/momentum_geometry.h"

#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/character/parameter_transform.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fmt/format.h>

#include <utility>

namespace mm = momentum;

namespace pymomentum {

void registerParameterTransformBindings(
    py::class_<mm::ParameterTransform>& parameterTransformClass) {
  // =====================================================
  // momentum::ParameterTransform
  // - names
  // - size()
  // - apply(modelParameters)
  // - getScalingParameters()
  // - getRigidParameters()
  // - getParametersForJoints(jointIndices)
  // - createInverseParameterTransform()
  // =====================================================
  parameterTransformClass
      .def(
          py::init([](const std::vector<std::string>& names,
                      const mm::Skeleton& skeleton,
                      const Eigen::SparseMatrix<float, Eigen::RowMajor>& transform) {
            const auto nJoints = skeleton.joints.size();
            const auto nJointParams = nJoints * mm::kParametersPerJoint;
            const auto nModelParams = names.size();

            MT_THROW_IF_T(
                transform.rows() != nJointParams,
                py::value_error,
                "Expected parameter transform to have {} rows (7*{} joints), but got {}",
                nJointParams,
                nJoints,
                transform.rows());
            MT_THROW_IF_T(
                transform.cols() != nModelParams,
                py::value_error,
                "Expected parameter transform to have {} columns (matching parameter names), but got {}",
                nModelParams,
                transform.cols());

            mm::ParameterTransform parameterTransform;
            parameterTransform.name = names;
            parameterTransform.transform.resize(nJointParams, nModelParams);

            for (int i = 0; i < transform.outerSize(); ++i) {
              for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(transform, i); it;
                   ++it) {
                parameterTransform.transform.coeffRef(
                    static_cast<long>(it.row()), static_cast<long>(it.col())) = it.value();
              }
            }

            parameterTransform.offsets.setZero(nJointParams);
            return parameterTransform;
          }),
          R"(Create a parameter transform from a sparse matrix.

:param names: List of model parameter names.
:param skeleton: The skeleton that this parameter transform operates on.
:param transform: A sparse matrix of size (7*n_joints x n_params) that maps from model parameters to joint parameters.)",
          py::arg("names"),
          py::arg("skeleton"),
          py::arg("transform"))
      .def(
          py::init([](const std::vector<std::string>& names,
                      const mm::Skeleton& skeleton,
                      const py::array_t<float>& transform) {
            const auto nJoints = skeleton.joints.size();
            const auto nJointParams = nJoints * mm::kParametersPerJoint;
            const auto nModelParams = names.size();

            MT_THROW_IF_T(
                transform.ndim() != 2,
                py::value_error,
                "Expected parameter transform to be a 2D array, but got {}D array",
                transform.ndim());
            MT_THROW_IF_T(
                transform.shape(0) != nJointParams,
                py::value_error,
                "Expected parameter transform to have {} rows (7*{} joints), but got {}",
                nJointParams,
                nJoints,
                transform.shape(0));
            MT_THROW_IF_T(
                transform.shape(1) != nModelParams,
                py::value_error,
                "Expected parameter transform to have {} columns (matching {} parameter names), but got {}",
                nModelParams,
                nModelParams,
                transform.shape(1));

            mm::ParameterTransform parameterTransform;
            parameterTransform.name = names;

            std::vector<Eigen::Triplet<float>> triplets;
            auto accessor = transform.unchecked<2>();
            for (int i = 0; i < accessor.shape(0); ++i) {
              for (int j = 0; j < accessor.shape(1); ++j) {
                if (accessor(i, j) != 0) {
                  triplets.emplace_back(i, j, accessor(i, j));
                }
              }
            }

            parameterTransform.transform.resize(nJointParams, nModelParams);
            parameterTransform.transform.setFromTriplets(triplets.begin(), triplets.end());
            parameterTransform.offsets.setZero(nJointParams);
            return parameterTransform;
          }),
          R"(Create a parameter transform from a dense numpy array.

The array will be converted to a sparse matrix internally for efficient storage and computation.

:param names: List of model parameter names.
:param skeleton: The skeleton that this parameter transform operates on.
:param transform: A dense numpy array of size (7*n_joints x n_params) that maps from model parameters to joint parameters.)",
          py::arg("names"),
          py::arg("skeleton"),
          py::arg("transform"))
      .def_readonly("names", &mm::ParameterTransform::name, "List of model parameter names")
      .def_property_readonly(
          "size",
          &mm::ParameterTransform::numAllModelParameters,
          "Size of the model parameter vector.")
      .def(
          "apply",
          [](const mm::ParameterTransform& paramTransform,
             const py::array& modelParams,
             bool flatten) -> py::array {
            return applyParameterTransformArray(paramTransform, modelParams, flatten);
          },
          R"(Apply the parameter transform to model parameters.

Supports arbitrary leading dimensions with broadcasting and both float32/float64 dtypes.

Input shape: [..., numModelParams]
Output shape: [..., numJoints * 7] if flatten=True, [..., numJoints, 7] if flatten=False

The modelParameters store the reduced set of parameters (typically around 50) that are actually
optimized in the IK step.

The jointParameters store (tx, ty, tz, rx, ry, rz, scale) for each joint, relative to the parent
joint. Rotations are in Euler angles.

:param model_parameters: Numpy array containing model parameters
:param flatten: If True (default), return shape [..., numJoints * 7]. If False, return shape [..., numJoints, 7].
:return: Numpy array containing joint parameters)",
          py::arg("model_parameters"),
          py::arg("flatten") = true)
      .def_property_readonly(
          "scaling_parameters",
          &getScalingParametersArray,
          "Boolean numpy array indicating which parameters are used to control the character's scale.")
      .def_property_readonly(
          "rigid_parameters",
          &getRigidParametersArray,
          "Boolean numpy array indicating which parameters are used to control the character's rigid transform (translation and rotation).")
      .def_property_readonly(
          "all_parameters",
          &getAllParametersArray,
          "Boolean numpy array with all parameters enabled.")
      .def_property_readonly(
          "blend_shape_parameters",
          &getBlendShapeParametersArray,
          "Boolean numpy array with just the blend shape parameters enabled.")
      .def_property_readonly(
          "face_expression_parameters",
          &getFaceExpressionParametersArray,
          "Boolean numpy array with just the face expression parameters enabled.")
      .def_property_readonly(
          "pose_parameters",
          &getPoseParametersArray,
          "Boolean numpy array with all the parameters used to pose the body, excluding and scaling, blend shape, or physics parameters.")
      .def_property_readonly(
          "no_parameters",
          [](const momentum::ParameterTransform& parameterTransform) {
            return parameterSetToArray(parameterTransform, momentum::ParameterSet());
          },
          "Boolean numpy array with no parameters enabled.")
      .def_property_readonly(
          "parameter_sets",
          &getParameterSetsArray,
          R"(A dictionary mapping names to sets of parameters (as a boolean numpy array) that are defined in the .model file.
This is convenient for turning off certain body features; for example the 'fingers' parameters
can be used to enable/disable finger motion in the character model.  )")
      .def(
          "add_parameter_set",
          &addParameterSetArray,
          R"(Adds a parameter set.

:param parameter_set_name: The name of the parameter set.
:param parameter_set: The numpy array of parameter set values.)",
          py::arg("parameter_set_name"),
          py::arg("parameter_set"))
      .def(
          "parameters_for_joints",
          [](const mm::ParameterTransform& paramTransform,
             const std::vector<size_t>& jointIndices) -> py::array_t<bool> {
            return getParametersForJointsArray(paramTransform, jointIndices);
          },
          R"(Gets a boolean numpy array indicating which parameters affect the passed-in joints.

:param joint_indices: List of integers of skeleton joints.
:return: Boolean numpy array of shape (numModelParams,))",
          py::arg("joint_indices"))
      .def(
          "find_parameters",
          [](const mm::ParameterTransform& paramTransform,
             const std::vector<std::string>& names,
             bool allowMissing) -> py::array_t<bool> {
            return findParametersArray(paramTransform, names, allowMissing);
          },
          R"(Return a boolean numpy array with the named parameters set to true.

:param names: Names of the parameters to find.
:param allow_missing: If false, missing parameters will throw an exception.
:return: Boolean numpy array of shape (numModelParams,))",
          py::arg("names"),
          py::arg("allow_missing") = false)
      .def(
          "inverse",
          [](const mm::ParameterTransform& transform) {
            return std::make_unique<mm::InverseParameterTransform>(transform);
          },
          R"(Compute the inverse of the parameter transform (a mapping from joint parameters to model parameters).

:return: The inverse parameter transform.)")
      .def_property_readonly(
          "transform",
          &getParameterTransformMatrix,
          "Returns the parameter transform matrix which when applied maps model parameters to joint parameters.")
      .def("__repr__", [](const mm::ParameterTransform& pt) {
        return fmt::format(
            "ParameterTransform(parameters={}, joints={})",
            pt.numAllModelParameters(),
            pt.transform.rows() / mm::kParametersPerJoint);
      });
}

void registerInverseParameterTransformBindings(
    py::class_<mm::InverseParameterTransform>& inverseParameterTransformClass) {
  // =====================================================
  // momentum::InverseParameterTransform
  // - apply()
  // =====================================================
  inverseParameterTransformClass
      .def(
          "apply",
          [](const mm::InverseParameterTransform& invParamTransform,
             const py::array& jointParams) -> py::array_t<float> {
            return applyInverseParameterTransformArray(invParamTransform, jointParams);
          },
          R"(Apply the inverse parameter transform to joint parameters.

Supports arbitrary leading dimensions with broadcasting. Always returns float32.

Because the number of joint parameters is much larger than the number of model parameters, this will in general have a non-zero residual.

Input shape: [..., numJoints, 7]
Output shape: [..., numModelParams]

:param joint_parameters: Numpy array containing joint parameters with shape (..., numJoints, 7).
:return: Float32 numpy array containing the (..., nModelParameters) model parameters.)",
          py::arg("joint_parameters"))
      .def("__repr__", [](const mm::InverseParameterTransform& ipt) {
        return fmt::format(
            "InverseParameterTransform(parameters={}, joints={})",
            ipt.transform.cols(),
            ipt.transform.rows() / mm::kParametersPerJoint);
      });
}

} // namespace pymomentum
