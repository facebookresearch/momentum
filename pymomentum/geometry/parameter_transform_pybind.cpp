/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/parameter_transform_pybind.h"

#include "pymomentum/geometry/momentum_geometry.h"
#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"

#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/character/parameter_transform.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>

#include <fmt/format.h>

#include <utility>

namespace py = pybind11;
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
            mm::ParameterTransform parameterTransform;
            parameterTransform.name = names;
            parameterTransform.transform.resize(
                static_cast<int>(skeleton.joints.size()) * mm::kParametersPerJoint,
                static_cast<int>(names.size()));

            for (int i = 0; i < transform.outerSize(); ++i) {
              for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(transform, i); it;
                   ++it) {
                parameterTransform.transform.coeffRef(
                    static_cast<long>(it.row()), static_cast<long>(it.col())) = it.value();
              }
            }

            parameterTransform.offsets.setZero(skeleton.joints.size() * mm::kParametersPerJoint);
            return parameterTransform;
          }),
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
          [](const mm::ParameterTransform* paramTransform,
             torch::Tensor modelParams) -> torch::Tensor {
            return applyParamTransform(paramTransform, std::move(modelParams));
          },
          R"(Apply the parameter transform to a k-dimensional model parameter vector (returns the 7*nJoints joint parameter vector).

The modelParameters store the reduced set of parameters (typically around 50) that are actually
optimized in the IK step.

The jointParameters are stored (tx, ty, tz; rx, ry, rz; s) and each represents the transform relative to the parent joint.
Rotations are in Euler angles.)",
          py::arg("model_parameters"))
      .def_property_readonly(
          "scaling_parameters",
          &getScalingParameters,
          "Boolean torch.Tensor indicating which parameters are used to control the character's scale.")
      .def_property_readonly(
          "rigid_parameters",
          &getRigidParameters,
          "Boolean torch.Tensor indicating which parameters are used to control the character's rigid transform (translation and rotation).")
      .def_property_readonly(
          "all_parameters", &getAllParameters, "Boolean torch.Tensor with all parameters enabled.")
      .def_property_readonly(
          "blend_shape_parameters",
          &getBlendShapeParameters,
          "Boolean torch.Tensor with just the blend shape parameters enabled.")
      .def_property_readonly(
          "pose_parameters",
          &getPoseParameters,
          "Boolean torch.Tensor with all the parameters used to pose the body, excluding and scaling, blend shape, or physics parameters.")
      .def_property_readonly(
          "no_parameters",
          [](const momentum::ParameterTransform& parameterTransform) {
            return parameterSetToTensor(parameterTransform, momentum::ParameterSet());
          },
          "Boolean torch.Tensor with no parameters enabled.")
      .def_property_readonly(
          "parameter_sets",
          &getParameterSets,
          R"(A dictionary mapping names to sets of parameters (as a boolean torch.Tensor) that are defined in the .model file.
This is convenient for turning off certain body features; for example the 'fingers' parameters
can be used to enable/disable finger motion in the character model.  )")
      .def(
          "parameters_for_joints",
          &getParametersForJoints,
          R"(Gets a boolean torch.Tensor indicating which parameters affect the passed-in joints.

:param jointIndices: List of integers of skeleton joints.)",
          py::arg("joint_indices"))
      .def(
          "find_parameters",
          &findParameters,
          R"(Return a boolean tensor with the named parameters set to true.

:param parameter_names: Names of the parameters to find.
:param allow_missng: If false, missing parameters will throw an exception.
        )",
          py::arg("names"),
          py::arg("allow_missing") = false)
      .def(
          "inverse",
          &createInverseParameterTransform,
          R"(Compute the inverse of the parameter transform (a mapping from joint parameters to model parameters).

:return: The inverse parameter transform.)")
      .def_property_readonly(
          "transform",
          &getParameterTransformTensor,
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
          &applyInverseParamTransform,
          R"(Apply the inverse parameter transform to a 7*nJoints-dimensional joint parameter vector (returns the k-dimensional model parameter vector).

Because the number of joint parameters is much larger than the number of model parameters, this will in general have a non-zero residual.

:param joint_parameters: Joint parameter tensor with dimensions (nBatch x 7*nJoints).
:return: A torch.Tensor containing the (nBatch x nModelParameters) model parameters.)",
          py::arg("joint_parameters"))
      .def("__repr__", [](const mm::InverseParameterTransform& ipt) {
        return fmt::format(
            "InverseParameterTransform(parameters={}, joints={})",
            ipt.transform.cols(),
            ipt.transform.rows() / mm::kParametersPerJoint);
      });
}

} // namespace pymomentum
