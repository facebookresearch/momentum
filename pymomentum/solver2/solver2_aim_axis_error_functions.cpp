/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pymomentum/solver2/solver2_error_functions.h>

#include <momentum/character/character.h>
#include <momentum/character_solver/aim_error_function.h>
#include <momentum/character_solver/fixed_axis_error_function.h>
#include <pymomentum/solver2/solver2_utility.h>

#include <fmt/format.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

namespace {

template <typename AimErrorFunctionT>
void defAimErrorFunction(py::module_& m, const char* name, const char* description) {
  py::class_<AimErrorFunctionT, mm::SkeletonErrorFunction, std::shared_ptr<AimErrorFunctionT>>(
      m, name, description)
      .def(
          "__repr__",
          [=](const AimErrorFunctionT& self) -> std::string {
            return fmt::format(
                "{}(weight={}, num_constraints={})", name, self.getWeight(), self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character, float lossAlpha, float lossC, float weight)
                  -> std::shared_ptr<AimErrorFunctionT> {
                validateWeight(weight, "weight");
                auto result = std::make_shared<AimErrorFunctionT>(character, lossAlpha, lossC);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize error function.

            :param character: The character to use.
            :param alpha: P-norm to use; 2 is Euclidean distance, 1 is L1 norm, 0 is Cauchy norm.
            :param c: c parameter in the generalized loss function, this is roughly equivalent to normalizing by the standard deviation in a Gaussian distribution.
            :param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("alpha") = 2.0f,
          py::arg("c") = 1.0f,
          py::arg("weight") = 1.0f)
      .def_property_readonly(
          "constraints",
          [](const AimErrorFunctionT& self) { return self.getConstraints(); },
          "Returns the list of aim constraints.")
      .def(
          "add_constraint",
          [](AimErrorFunctionT& self,
             const Eigen::Vector3f& localPoint,
             const Eigen::Vector3f& localDir,
             const Eigen::Vector3f& globalTarget,
             int parent,
             float weight,
             const std::string& name) {
            validateJointIndex(parent, "parent", self.getSkeleton());
            validateWeight(weight, "weight");
            self.addConstraint(
                mm::AimDataT<float>(localPoint, localDir, globalTarget, parent, weight, name));
          },
          R"(Adds an aim constraint to the error function.

      :param local_point: The origin of the local ray.
      :param local_dir: The direction of the local ray.
      :param global_target: The global aim target.
      :param parent_index: The index of the parent joint.
      :param weight: The weight of the constraint.
      :param name: The name of the constraint (for debugging).)",
          py::arg("local_point"),
          py::arg("local_dir"),
          py::arg("global_target"),
          py::arg("parent"),
          py::arg("weight") = 1.0f,
          py::arg("name") = std::string{})
      .def(
          "add_constraints",
          [](AimErrorFunctionT& self,
             const py::array_t<float>& localPoint,
             const py::array_t<float>& localDir,
             const py::array_t<float>& globalTarget,
             const py::array_t<int>& parent,
             const std::optional<py::array_t<float>>& weight,
             const std::optional<std::vector<std::string>>& name) {
            ArrayShapeValidator arrayValidator;
            const int nConsIdx = -1;
            arrayValidator.validate(localPoint, "local_point", {nConsIdx, 3}, {"n_cons", "xyz"});
            arrayValidator.validate(localDir, "local_dir", {nConsIdx, 3}, {"n_cons", "xyz"});
            arrayValidator.validate(
                globalTarget, "global_target", {nConsIdx, 3}, {"n_cons", "xyz"});
            arrayValidator.validate(parent, "parent", {nConsIdx}, {"n_cons"});
            validateJointIndex(parent, "parent", self.getSkeleton());
            arrayValidator.validate(weight, "weight", {nConsIdx}, {"n_cons"});

            if (name.has_value() && name->size() != parent.shape(0)) {
              throw std::runtime_error(
                  fmt::format(
                      "Invalid names; expected {} names but got {}",
                      parent.shape(0),
                      name->size()));
            }

            py::gil_scoped_release release;

            auto localPointAcc = localPoint.unchecked<2>();
            auto localDirAcc = localDir.unchecked<2>();
            auto globalTargetAcc = globalTarget.unchecked<2>();
            auto parentAcc = parent.unchecked<1>();
            auto weightAcc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            for (py::ssize_t i = 0; i < localPoint.shape(0); ++i) {
              validateJointIndex(parentAcc(i), "parent", self.getSkeleton());
              self.addConstraint(
                  mm::AimDataT<float>(
                      Eigen::Vector3f(
                          localPointAcc(i, 0), localPointAcc(i, 1), localPointAcc(i, 2)),
                      Eigen::Vector3f(localDirAcc(i, 0), localDirAcc(i, 1), localDirAcc(i, 2)),
                      Eigen::Vector3f(
                          globalTargetAcc(i, 0), globalTargetAcc(i, 1), globalTargetAcc(i, 2)),
                      parentAcc(i),
                      weightAcc.has_value() ? (*weightAcc)(i) : 1.0f,
                      name.has_value() ? name->at(i) : std::string{}));
            }
          },
          R"(Adds multiple aim constraints to the error function.

      :param local_point: A numpy array of shape (n, 3) for the origins of the local rays.
      :param local_dir: A numpy array of shape (n, 3) for the directions of the local rays.
      :param global_target: A numpy array of shape (n, 3) for the global aim targets.
      :param parent_index: A numpy array of size n for the indices of the parent joints.
      :param weight: A numpy array of size n for the weights of the constraints.
      :param name: An optional list of names for the constraints (for debugging).)",
          py::arg("local_point"),
          py::arg("local_dir"),
          py::arg("global_target"),
          py::arg("parent_index"),
          py::arg("weight"),
          py::arg("name") = std::optional<std::vector<std::string>>{})
      .def(
          "clear_constraints",
          &AimErrorFunctionT::clearConstraints,
          "Clears all aim constraints from the error function.");
}

template <typename FixedAxisErrorFunctionT>
void defFixedAxisError(py::module_& m, const char* name, const char* description) {
  py::class_<
      FixedAxisErrorFunctionT,
      mm::SkeletonErrorFunctionT<float>,
      std::shared_ptr<FixedAxisErrorFunctionT>>(m, name, description)
      .def(
          "__repr__",
          [=](const FixedAxisErrorFunctionT& self) -> std::string {
            return fmt::format(
                "{}(weight={}, num_constraints={})", name, self.getWeight(), self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character, float lossAlpha, float lossC, float weight)
                  -> std::shared_ptr<FixedAxisErrorFunctionT> {
                validateWeight(weight, "weight");
                auto result =
                    std::make_shared<FixedAxisErrorFunctionT>(character, lossAlpha, lossC);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a FixedAxisDiffErrorFunction.

            :param character: The character to use.
            :param alpha: P-norm to use; 2 is Euclidean distance, 1 is L1 norm, 0 is Cauchy norm.
            :param c: c parameter in the generalized loss function, this is roughly equivalent to normalizing by the standard deviation in a Gaussian distribution.
            :param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("alpha") = 2.0f,
          py::arg("c") = 1.0f,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](FixedAxisErrorFunctionT& self,
             const Eigen::Vector3f& localAxis,
             const Eigen::Vector3f& globalAxis,
             int parent,
             float weight,
             const std::string& name) {
            validateJointIndex(parent, "parent", self.getSkeleton());
            validateWeight(weight, "weight");
            self.addConstraint(
                mm::FixedAxisDataT<float>(localAxis, globalAxis, parent, weight, name));
          },
          R"(Adds a fixed axis constraint to the error function.

        :param local_axis: The axis in the parent's coordinate frame.
        :param global_axis: The target axis in the global frame.
        :param parent: The index of the parent joint.
        :param weight: The weight of the constraint.
        :param name: The name of the constraint (for debugging).)",
          py::arg("local_axis"),
          py::arg("global_axis"),
          py::arg("parent"),
          py::arg("weight") = 1.0f,
          py::arg("name") = std::string{})
      .def_property_readonly(
          "constraints",
          [](const FixedAxisErrorFunctionT& self) { return self.getConstraints(); },
          "Returns the list of fixed axis constraints.")
      .def(
          "add_constraints",
          [](FixedAxisErrorFunctionT& self,
             const py::array_t<float>& localAxis,
             const py::array_t<float>& globalAxis,
             const py::array_t<int>& parent,
             const std::optional<py::array_t<float>>& weight,
             const std::optional<std::vector<std::string>>& name) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(localAxis, "local_axis", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(globalAxis, "global_axis", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(parent, "parent", {nConsIdx}, {"n_cons"});
            validateJointIndex(parent, "parent", self.getSkeleton());
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});

            if (name.has_value() && name->size() != parent.shape(0)) {
              throw std::runtime_error(
                  fmt::format(
                      "Invalid names; expected {} names but got {}",
                      parent.shape(0),
                      name->size()));
            }

            auto localAxisAcc = localAxis.unchecked<2>();
            auto globalAxisAcc = globalAxis.unchecked<2>();
            auto parentAcc = parent.unchecked<1>();
            auto weightAcc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            for (py::ssize_t i = 0; i < localAxis.shape(0); ++i) {
              self.addConstraint(
                  mm::FixedAxisDataT<float>(
                      Eigen::Vector3f(localAxisAcc(i, 0), localAxisAcc(i, 1), localAxisAcc(i, 2)),
                      Eigen::Vector3f(
                          globalAxisAcc(i, 0), globalAxisAcc(i, 1), globalAxisAcc(i, 2)),
                      parentAcc(i),
                      weightAcc.has_value() ? (*weightAcc)(i) : 1.0f,
                      name.has_value() ? name->at(i) : std::string{}));
            }
          },
          R"(Adds multiple fixed axis constraints to the error function.

        :param local_axis: A numpy array of shape (n, 3) for the axes in the parent's coordinate frame.
        :param global_axis: A numpy array of shape (n, 3) for the target axes in the global frame.
        :param parent_index: A numpy array of size n for the indices of the parent joints.
        :param weight: A numpy array of size n for the weights of the constraints.
        :param name: An optional list of names for the constraints (for debugging).)",
          py::arg("local_axis"),
          py::arg("global_axis"),
          py::arg("parent_index"),
          py::arg("weight"),
          py::arg("name") = std::optional<std::vector<std::string>>{})
      .def(
          "clear_constraints",
          &FixedAxisErrorFunctionT::clearConstraints,
          "Clears all fixed axis constraints from the error function.");
}

} // namespace

void addAimAxisErrorFunctions(py::module_& m) {
  // Aim error functions
  py::class_<mm::AimDataT<float>>(m, "AimData")
      .def(
          "__repr__",
          [](const mm::AimDataT<float>& self) {
            return fmt::format(
                "AimData(parent={}, weight={}, local_point=[{:.3f}, {:.3f}, {:.3f}], local_dir=[{:.3f}, {:.3f}, {:.3f}], global_target=[{:.3f}, {:.3f}, {:.3f}])",
                self.parent,
                self.weight,
                self.localPoint.x(),
                self.localPoint.y(),
                self.localPoint.z(),
                self.localDir.x(),
                self.localDir.y(),
                self.localDir.z(),
                self.globalTarget.x(),
                self.globalTarget.y(),
                self.globalTarget.z());
          })
      .def_readonly("parent", &mm::AimDataT<float>::parent, "The parent joint index")
      .def_readonly("weight", &mm::AimDataT<float>::weight, "The weight of the constraint")
      .def_readonly("local_point", &mm::AimDataT<float>::localPoint, "The origin of the local ray")
      .def_readonly("local_dir", &mm::AimDataT<float>::localDir, "The direction of the local ray")
      .def_readonly("global_target", &mm::AimDataT<float>::globalTarget, "The global aim target");

  defAimErrorFunction<mm::AimDistErrorFunctionT<float>>(
      m,
      "AimDistErrorFunction",
      R"(The AimDistErrorFunction minimizes the distance between a ray (origin, direction) defined in
joint-local space and the world-space target point.   The residual is defined as the distance
between the tartet position and its projection onto the ray.  Note that the ray is only defined
for positive t values, meaning that if the target point is _behind_ the ray origin, its
projection will be at the ray origin where t=0.)");

  defAimErrorFunction<mm::AimDirErrorFunctionT<float>>(m, "AimDirErrorFunction", R"(
The AimDirErrorFunction minimizes the element-wise difference between a ray (origin, direction)
defined in joint-local space and the normalized vector connecting the ray origin to the
world-space target point.  If the vector has near-zero length, the residual is set to zero to
avoid divide-by-zero. )");

  // Fixed axis error functions
  py::class_<mm::FixedAxisDataT<float>>(m, "FixedAxisData")
      .def(
          "__repr__",
          [](const mm::FixedAxisDataT<float>& self) {
            return fmt::format(
                "FixedAxisData(parent={}, weight={}, local_axis=[{:.3f}, {:.3f}, {:.3f}], global_axis=[{:.3f}, {:.3f}, {:.3f}])",
                self.parent,
                self.weight,
                self.localAxis.x(),
                self.localAxis.y(),
                self.localAxis.z(),
                self.globalAxis.x(),
                self.globalAxis.y(),
                self.globalAxis.z());
          })
      .def_readonly("parent", &mm::FixedAxisDataT<float>::parent, "The parent joint index")
      .def_readonly("weight", &mm::FixedAxisDataT<float>::weight, "The weight of the constraint")
      .def_readonly(
          "local_axis", &mm::FixedAxisDataT<float>::localAxis, "The local axis in parent space")
      .def_readonly("global_axis", &mm::FixedAxisDataT<float>::globalAxis, "The global axis");

  defFixedAxisError<mm::FixedAxisDiffErrorFunctionT<float>>(
      m,
      "FixedAxisDiffErrorFunction",
      R"(Error function that minimizes the difference between a local axis (in
      joint-local space) and a global axis using element-wise differences.)");
  defFixedAxisError<mm::FixedAxisCosErrorFunctionT<float>>(
      m,
      "FixedAxisCosErrorFunction",
      R"(Error function that minimizes the difference between a local axis (in
      joint-local space) and a global axis using the cosine of the angle between
      the vectors (which is the same as the dot product of the vectors).)");
  defFixedAxisError<mm::FixedAxisAngleErrorFunctionT<float>>(
      m,
      "FixedAxisAngleErrorFunction",
      R"(Error function that minimizes the difference between a local axis (in
      joint-local space) and a global axis using the angle between the vectors.)");
}

} // namespace pymomentum
