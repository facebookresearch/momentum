/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character_solver/aim_error_function.h>
#include <momentum/character_solver/collision_error_function.h>
#include <momentum/character_solver/distance_error_function.h>
#include <momentum/character_solver/fixed_axis_error_function.h>
#include <momentum/character_solver/joint_to_joint_distance_error_function.h>
#include <momentum/character_solver/limit_error_function.h>
#include <momentum/character_solver/model_parameters_error_function.h>
#include <momentum/character_solver/normal_error_function.h>
#include <momentum/character_solver/orientation_error_function.h>
#include <momentum/character_solver/plane_error_function.h>
#include <momentum/character_solver/point_triangle_vertex_error_function.h>
#include <momentum/character_solver/pose_prior_error_function.h>
#include <momentum/character_solver/position_error_function.h>
#include <momentum/character_solver/projection_error_function.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/state_error_function.h>
#include <momentum/character_solver/vertex_projection_error_function.h>
#include <momentum/character_solver/vertex_vertex_distance_error_function.h>
#include <pymomentum/python_utility/eigen_quaternion.h>
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
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(localPoint, "local_point", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(localDir, "local_dir", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(globalTarget, "global_target", {nConsIdx, 3}, {"n_cons", "xyz"});
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
          [](mm::FixedAxisDiffErrorFunctionT<float>& self,
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
        :param parent_index: The index of the parent joint.
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

void defNormalErrorFunction(py::module_& m) {
  py::class_<mm::NormalDataT<float>>(m, "NormalData")
      .def(
          "__repr__",
          [](const mm::NormalDataT<float>& self) {
            return fmt::format(
                "NormalData(parent={}, weight={}, local_point=[{:.3f}, {:.3f}, {:.3f}], local_normal=[{:.3f}, {:.3f}, {:.3f}], global_point=[{:.3f}, {:.3f}, {:.3f}])",
                self.parent,
                self.weight,
                self.localPoint.x(),
                self.localPoint.y(),
                self.localPoint.z(),
                self.localNormal.x(),
                self.localNormal.y(),
                self.localNormal.z(),
                self.globalPoint.x(),
                self.globalPoint.y(),
                self.globalPoint.z());
          })
      .def_readonly("parent", &mm::NormalDataT<float>::parent, "The parent joint index")
      .def_readonly("weight", &mm::NormalDataT<float>::weight, "The weight of the constraint")
      .def_readonly(
          "local_point", &mm::NormalDataT<float>::localPoint, "The local point in parent space")
      .def_readonly(
          "local_normal", &mm::NormalDataT<float>::localNormal, "The local normal in parent space")
      .def_readonly("global_point", &mm::NormalDataT<float>::globalPoint, "The global point");
  py::class_<
      mm::NormalErrorFunctionT<float>,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::NormalErrorFunctionT<float>>>(
      m,
      "NormalErrorFunction",
      R"(The NormalErrorFunction computes a "point-to-plane" (signed) distance from a target point to the
plane defined by a local point and a local normal vector.)")
      .def(
          "__repr__",
          [](const mm::NormalErrorFunctionT<float>& self) {
            return fmt::format(
                "NormalErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character, float lossAlpha, float lossC, float weight)
                  -> std::shared_ptr<mm::NormalErrorFunctionT<float>> {
                validateWeight(weight, "weight");
                auto result =
                    std::make_shared<mm::NormalErrorFunctionT<float>>(character, lossAlpha, lossC);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a NormalErrorFunction.

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
          [](mm::NormalErrorFunctionT<float>& self,
             const Eigen::Vector3f& localNormal,
             const Eigen::Vector3f& globalPoint,
             int parent,
             const std::optional<Eigen::Vector3f>& localPoint,
             float weight,
             const std::string& name) {
            validateJointIndex(parent, "parent", self.getSkeleton());
            validateWeight(weight, "weight");
            self.addConstraint(
                mm::NormalDataT<float>(
                    localPoint.value_or(Eigen::Vector3f::Zero()),
                    localNormal,
                    globalPoint,
                    parent,
                    weight,
                    name));
          },
          R"(Adds a normal constraint to the error function.

        :param local_normal: The normal in the parent's coordinate frame.
        :param global_point: The target point in global space.
        :param parent: The index of the parent joint.
        :param local_point: The point in the parent's coordinate frame (defaults to origin).
        :param weight: The weight of the constraint.
        :param name: The name of the constraint (for debugging).)",
          py::arg("local_normal"),
          py::arg("global_point"),
          py::arg("parent"),
          py::arg("local_point") = std::nullopt,
          py::arg("weight") = 1.0f,
          py::arg("name") = std::string{})
      .def_property_readonly(
          "constraints",
          [](const mm::NormalErrorFunctionT<float>& self) { return self.getConstraints(); },
          "Returns the list of normal constraints.")
      .def(
          "add_constraints",
          [](mm::NormalErrorFunctionT<float>& self,
             const py::array_t<float>& localNormal,
             const py::array_t<float>& globalPoint,
             const py::array_t<int>& parent,
             const std::optional<py::array_t<float>>& localPoint,
             const std::optional<py::array_t<float>>& weight,
             const std::optional<std::vector<std::string>>& name) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(localPoint, "local_point", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(localNormal, "local_normal", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(globalPoint, "global_point", {nConsIdx, 3}, {"n_cons", "xyz"});
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

            auto localPointAcc = localPoint.has_value()
                ? std::make_optional(localPoint->unchecked<2>())
                : std::nullopt;
            auto localNormalAcc = localNormal.unchecked<2>();
            auto globalPointAcc = globalPoint.unchecked<2>();
            auto parentAcc = parent.unchecked<1>();
            auto weightAcc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            for (py::ssize_t i = 0; i < parent.shape(0); ++i) {
              self.addConstraint(
                  mm::NormalDataT<float>(
                      localPointAcc.has_value() ? Eigen::Vector3f(
                                                      (*localPointAcc)(i, 0),
                                                      (*localPointAcc)(i, 1),
                                                      (*localPointAcc)(i, 2))
                                                : Eigen::Vector3f::Zero(),
                      Eigen::Vector3f(
                          localNormalAcc(i, 0), localNormalAcc(i, 1), localNormalAcc(i, 2)),
                      Eigen::Vector3f(
                          globalPointAcc(i, 0), globalPointAcc(i, 1), globalPointAcc(i, 2)),
                      parentAcc(i),
                      weightAcc.has_value() ? (*weightAcc)(i) : 1.0f,
                      name.has_value() ? name->at(i) : std::string{}));
            }
          },
          R"(Adds multiple normal constraints to the error function.

        :param local_normal: A numpy array of shape (n, 3) for the normals in the parent's coordinate frame.
        :param global_point: A numpy array of shape (n, 3) for the target points in global space.
        :param parent: A numpy array of size n for the indices of the parent joints.
        :param local_point: A numpy array of shape (n, 3) for the points in the parent's coordinate frame (defaults to origin).
        :param weight: A numpy array of size n for the weights of the constraints.
        :param name: An optional list of names for the constraints (for debugging).)",
          py::arg("local_normal"),
          py::arg("global_point"),
          py::arg("parent"),
          py::arg("local_point") = std::nullopt,
          py::arg("weight") = std::nullopt,
          py::arg("name") = std::optional<std::vector<std::string>>{})
      .def(
          "clear_constraints",
          &mm::NormalErrorFunctionT<float>::clearConstraints,
          "Clears all normal constraints from the error function.");
}

void defDistanceErrorFunction(py::module_& m) {
  py::class_<mm::DistanceConstraintDataT<float>>(m, "DistanceData")
      .def(
          "__repr__",
          [](const mm::DistanceConstraintDataT<float>& self) {
            return fmt::format(
                "DistanceData(parent={}, weight={}, offset=[{:.3f}, {:.3f}, {:.3f}], origin=[{:.3f}, {:.3f}, {:.3f}], target={:.3f})",
                self.parent,
                self.weight,
                self.offset.x(),
                self.offset.y(),
                self.offset.z(),
                self.origin.x(),
                self.origin.y(),
                self.origin.z(),
                self.target);
          })
      .def_readonly("parent", &mm::DistanceConstraintDataT<float>::parent, "The parent joint index")
      .def_readonly(
          "weight", &mm::DistanceConstraintDataT<float>::weight, "The weight of the constraint")
      .def_readonly(
          "offset", &mm::DistanceConstraintDataT<float>::offset, "The offset in parent space")
      .def_readonly(
          "origin", &mm::DistanceConstraintDataT<float>::origin, "The origin point in world space")
      .def_readonly("target", &mm::DistanceConstraintDataT<float>::target, "The target distance");

  py::class_<
      mm::DistanceErrorFunctionT<float>,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::DistanceErrorFunctionT<float>>>(
      m,
      "DistanceErrorFunction",
      R"(The DistanceErrorFunction computes the squared difference between the distance from a point to an origin
and a target distance. The constraint is defined as ||(p_joint - origin)^2 - target||^2.)")
      .def(
          "__repr__",
          [](const mm::DistanceErrorFunctionT<float>& self) {
            return fmt::format(
                "DistanceErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 float weight) -> std::shared_ptr<mm::DistanceErrorFunctionT<float>> {
                validateWeight(weight, "weight");
                auto result = std::make_shared<mm::DistanceErrorFunctionT<float>>(
                    character.skeleton, character.parameterTransform);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a DistanceErrorFunction.

            :param character: The character to use.
            :param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::DistanceErrorFunctionT<float>& self,
             const Eigen::Vector3f& origin,
             float target,
             int parent,
             const Eigen::Vector3f& offset,
             float weight) {
            validateJointIndex(parent, "parent", self.getSkeleton());
            validateWeight(weight, "weight");
            mm::DistanceConstraintDataT<float> constraint;
            constraint.origin = origin;
            constraint.target = target;
            constraint.parent = parent;
            constraint.offset = offset;
            constraint.weight = weight;
            self.addConstraint(constraint);
          },
          R"(Adds a distance constraint to the error function.

        :param origin: The origin point in world space.
        :param target: The target distance in world space.
        :param parent: The index of the parent joint.
        :param offset: The offset from the parent joint in local space.
        :param weight: The weight of the constraint.)",
          py::arg("origin"),
          py::arg("target"),
          py::arg("parent"),
          py::arg("offset"),
          py::arg("weight") = 1.0f)
      .def(
          "add_constraints",
          [](mm::DistanceErrorFunctionT<float>& self,
             const py::array_t<float>& origin,
             const py::array_t<float>& target,
             const py::array_t<int>& parent,
             const py::array_t<float>& offset,
             const std::optional<py::array_t<float>>& weight) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(origin, "origin", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(target, "target", {nConsIdx}, {"n_cons"});
            validator.validate(parent, "parent", {nConsIdx}, {"n_cons"});
            validateJointIndex(parent, "parent", self.getSkeleton());
            validator.validate(offset, "offset", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});

            auto originAcc = origin.unchecked<2>();
            auto targetAcc = target.unchecked<1>();
            auto parentAcc = parent.unchecked<1>();
            auto offsetAcc = offset.unchecked<2>();
            auto weightAcc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            py::gil_scoped_release release;

            for (py::ssize_t i = 0; i < parent.shape(0); ++i) {
              mm::DistanceConstraintDataT<float> constraint;
              constraint.origin =
                  Eigen::Vector3f(originAcc(i, 0), originAcc(i, 1), originAcc(i, 2));
              constraint.target = targetAcc(i);
              constraint.parent = parentAcc(i);
              constraint.offset =
                  Eigen::Vector3f(offsetAcc(i, 0), offsetAcc(i, 1), offsetAcc(i, 2));
              constraint.weight = weightAcc.has_value() ? (*weightAcc)(i) : 1.0f;
              self.addConstraint(constraint);
            }
          },
          R"(Adds multiple distance constraints to the error function.

        :param origin: A numpy array of shape (n, 3) for the origin points in world space.
        :param target: A numpy array of size n for the target distances in world space.
        :param parent: A numpy array of size n for the indices of the parent joints.
        :param offset: A numpy array of shape (n, 3) for the offsets from the parent joints in local space.
        :param weight: A numpy array of size n for the weights of the constraints.)",
          py::arg("origin"),
          py::arg("target"),
          py::arg("parent"),
          py::arg("offset"),
          py::arg("weight") = std::nullopt)
      .def(
          "clear_constraints",
          &mm::DistanceErrorFunctionT<float>::clearConstraints,
          "Clears all distance constraints from the error function.")
      .def(
          "empty",
          &mm::DistanceErrorFunctionT<float>::empty,
          "Returns true if there are no constraints.")
      .def(
          "num_constraints",
          &mm::DistanceErrorFunctionT<float>::numConstraints,
          "Returns the number of constraints.")
      .def_property_readonly(
          "constraints",
          &mm::DistanceErrorFunctionT<float>::getConstraints,
          "Returns the list of distance constraints.");
}

void defProjectionErrorFunction(py::module_& m) {
  py::class_<mm::ProjectionConstraintDataT<float>>(m, "ProjectionConstraint")
      .def(
          "__repr__",
          [](const mm::ProjectionConstraintDataT<float>& self) {
            return fmt::format(
                "ProjectionConstraint(parent={}, weight={}, offset=[{:.3f}, {:.3f}, {:.3f}], target=[{:.3f}, {:.3f}])",
                self.parent,
                self.weight,
                self.offset.x(),
                self.offset.y(),
                self.offset.z(),
                self.target.x(),
                self.target.y());
          })
      .def_readonly(
          "parent", &mm::ProjectionConstraintDataT<float>::parent, "The parent joint index")
      .def_readonly(
          "weight", &mm::ProjectionConstraintDataT<float>::weight, "The weight of the constraint")
      .def_readonly(
          "offset", &mm::ProjectionConstraintDataT<float>::offset, "The offset in parent space")
      .def_readonly(
          "target", &mm::ProjectionConstraintDataT<float>::target, "The target 2D position")
      .def_readonly(
          "projection",
          &mm::ProjectionConstraintDataT<float>::projection,
          "The 3x4 projection matrix");
  py::class_<
      mm::ProjectionErrorFunctionT<float>,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::ProjectionErrorFunctionT<float>>>(
      m,
      "ProjectionErrorFunction",
      R"(The ProjectionErrorFunction projects a 3D point to a 2D point using a projection matrix.
This is useful for camera-based constraints where you want to match a 3D point to a 2D observation.)")
      .def(
          "__repr__",
          [](const mm::ProjectionErrorFunctionT<float>& self) {
            return fmt::format(
                "ProjectionErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 float nearClip,
                 float weight) -> std::shared_ptr<mm::ProjectionErrorFunctionT<float>> {
                validateWeight(weight, "weight");
                auto result = std::make_shared<mm::ProjectionErrorFunctionT<float>>(
                    character.skeleton, character.parameterTransform, nearClip);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a ProjectionErrorFunction.

            :param character: The character to use.
            :param near_clip: The near clip distance. Points closer than this distance to the camera will be ignored.
            :param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("near_clip") = 1.0f,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::ProjectionErrorFunctionT<float>& self,
             const Eigen::Matrix<float, 3, 4>& projection,
             const Eigen::Vector2f& target,
             int parent,
             const std::optional<Eigen::Vector3f>& offset,
             float weight) {
            validateJointIndex(parent, "parent", self.getSkeleton());
            validateWeight(weight, "weight");
            mm::ProjectionConstraintDataT<float> constraint;
            constraint.projection = projection;
            constraint.parent = parent;
            constraint.offset = offset.value_or(Eigen::Vector3f::Zero());
            constraint.weight = weight;
            constraint.target = target;
            self.addConstraint(constraint);
          },
          R"(Adds a projection constraint to the error function.

        :param projection: The 3x4 projection matrix.
        :param target: The 2D target point.
        :param parent: The index of the parent joint.
        :param offset: The offset from the parent joint in local space.
        :param weight: The weight of the constraint.)",
          py::arg("projection"),
          py::arg("target"),
          py::arg("parent"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraints",
          [](mm::ProjectionErrorFunctionT<float>& self,
             const py::array_t<float>& projection,
             const py::array_t<float>& target,
             const py::array_t<int>& parent,
             const std::optional<py::array_t<float>>& offset,
             const std::optional<py::array_t<float>>& weight) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(
                projection, "projection", {nConsIdx, 3, 4}, {"n_cons", "rows", "cols"});
            validator.validate(target, "target", {nConsIdx, 2}, {"n_cons", "xy"});
            validator.validate(parent, "parent", {nConsIdx}, {"n_cons"});
            validateJointIndex(parent, "parent", self.getSkeleton());
            validator.validate(offset, "offset", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});

            auto projectionAcc = projection.unchecked<3>();
            auto targetAcc = target.unchecked<2>();
            auto parentAcc = parent.unchecked<1>();
            auto offsetAcc =
                offset.has_value() ? std::make_optional(offset->unchecked<2>()) : std::nullopt;
            auto weightAcc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            py::gil_scoped_release release;

            for (py::ssize_t i = 0; i < parent.shape(0); ++i) {
              mm::ProjectionConstraintDataT<float> constraint;

              // Set projection matrix
              Eigen::Matrix<float, 3, 4> proj;
              for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 4; ++col) {
                  proj(row, col) = projectionAcc(i, row, col);
                }
              }
              constraint.projection = proj;

              // Set target
              constraint.target = Eigen::Vector2f(targetAcc(i, 0), targetAcc(i, 1));

              // Set parent
              constraint.parent = parentAcc(i);

              // Set offset
              constraint.offset = offsetAcc.has_value()
                  ? Eigen::Vector3f((*offsetAcc)(i, 0), (*offsetAcc)(i, 1), (*offsetAcc)(i, 2))
                  : Eigen::Vector3f::Zero();

              // Set weight
              constraint.weight = weightAcc.has_value() ? (*weightAcc)(i) : 1.0f;

              self.addConstraint(constraint);
            }
          },
          R"(Adds multiple projection constraints to the error function.

        :param projection: A numpy array of shape (n, 3, 4) for the projection matrices.
        :param target: A numpy array of shape (n, 2) for the 2D target points.
        :param parent: A numpy array of size n for the indices of the parent joints.
        :param offset: A numpy array of shape (n, 3) for the offsets from the parent joints in local space.
        :param weight: A numpy array of size n for the weights of the constraints.)",
          py::arg("projection"),
          py::arg("target"),
          py::arg("parent"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = std::nullopt)
      .def(
          "clear_constraints",
          &mm::ProjectionErrorFunctionT<float>::clearConstraints,
          "Clears all projection constraints from the error function.")
      .def(
          "empty",
          &mm::ProjectionErrorFunctionT<float>::empty,
          "Returns true if there are no constraints.")
      .def(
          "num_constraints",
          &mm::ProjectionErrorFunctionT<float>::numConstraints,
          "Returns the number of constraints.")
      .def_property_readonly(
          "constraints",
          &mm::ProjectionErrorFunctionT<float>::getConstraints,
          "Returns the list of projection constraints.");
}

void defVertexProjectionErrorFunction(py::module_& m) {
  py::class_<mm::VertexProjectionConstraint>(
      m, "VertexProjectionConstraint", "Read-only access to a vertex projection constraint.")
      .def(
          "__repr__",
          [](const mm::VertexProjectionConstraint& self) {
            return fmt::format(
                "VertexProjectionConstraint(vertex_index={}, weight={}, target_position=[{:.3f}, {:.3f}])",
                self.vertexIndex,
                self.weight,
                self.targetPosition.x(),
                self.targetPosition.y());
          })
      .def_readonly(
          "vertex_index",
          &mm::VertexProjectionConstraint::vertexIndex,
          "Returns the index of the vertex to project.")
      .def_readonly(
          "weight",
          &mm::VertexProjectionConstraint::weight,
          "Returns the weight of the constraint.")
      .def_readonly(
          "target_position",
          &mm::VertexProjectionConstraint::targetPosition,
          "Returns the target 2D position.")
      .def_readonly(
          "projection",
          &mm::VertexProjectionConstraint::projection,
          "Returns the 3x4 projection matrix.");

  py::class_<
      mm::VertexProjectionErrorFunctionT<float>,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::VertexProjectionErrorFunctionT<float>>>(
      m,
      "VertexProjectionErrorFunction",
      R"(The VertexProjectionErrorFunction projects 3D vertices onto 2D points using a projection matrix.
This is useful for camera-based constraints where you want to match a 3D vertex to a 2D observation.)")
      .def(
          "__repr__",
          [](const mm::VertexProjectionErrorFunctionT<float>& self) {
            return fmt::format(
                "VertexProjectionErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 size_t maxThreads,
                 float weight) -> std::shared_ptr<mm::VertexProjectionErrorFunctionT<float>> {
                validateWeight(weight, "weight");
                auto result = std::make_shared<mm::VertexProjectionErrorFunctionT<float>>(
                    character, maxThreads);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a VertexProjectionErrorFunction.

            :param character: The character to use.
            :param max_threads: The maximum number of threads to use for computation.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("max_threads") = 0,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::VertexProjectionErrorFunctionT<float>& self,
             int vertexIndex,
             float weight,
             const Eigen::Vector2f& targetPosition,
             const Eigen::Matrix<float, 3, 4>& projection) {
            validateVertexIndex(vertexIndex, "vertex_index", self.getCharacter());
            validateWeight(weight, "weight");
            self.addConstraint(vertexIndex, weight, targetPosition, projection);
          },
          R"(Adds a vertex projection constraint to the error function.

        :param vertex_index: The index of the vertex to project.
        :param weight: The weight of the constraint.
        :param target_position: The target 2D position.
        :param projection: The 3x4 projection matrix.)",
          py::arg("vertex_index"),
          py::arg("weight"),
          py::arg("target_position"),
          py::arg("projection"))
      .def(
          "clear_constraints",
          &mm::VertexProjectionErrorFunctionT<float>::clearConstraints,
          "Clears all vertex projection constraints from the error function.")
      .def_property_readonly(
          "constraints",
          [](const mm::VertexProjectionErrorFunctionT<float>& self) {
            return self.getConstraints();
          },
          "Returns the list of vertex projection constraints.")
      .def(
          "add_constraints",
          [](mm::VertexProjectionErrorFunctionT<float>& self,
             const py::array_t<int>& vertexIndex,
             const py::array_t<float>& weight,
             const py::array_t<float>& targetPosition,
             const py::array_t<float>& projection) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(vertexIndex, "vertex_index", {nConsIdx}, {"n_cons"});
            validateVertexIndex(vertexIndex, "vertex_index", self.getCharacter());
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});
            validator.validate(targetPosition, "target_position", {nConsIdx, 2}, {"n_cons", "xy"});
            validator.validate(
                projection, "projection", {nConsIdx, 3, 4}, {"n_cons", "rows", "cols"});

            auto vertexIndexAcc = vertexIndex.unchecked<1>();
            auto weightAcc = weight.unchecked<1>();
            auto targetPositionAcc = targetPosition.unchecked<2>();
            auto projectionAcc = projection.unchecked<3>();

            py::gil_scoped_release release;

            for (py::ssize_t i = 0; i < vertexIndex.shape(0); ++i) {
              // Set projection matrix
              Eigen::Matrix<float, 3, 4> proj;
              for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 4; ++col) {
                  proj(row, col) = projectionAcc(i, row, col);
                }
              }

              self.addConstraint(
                  vertexIndexAcc(i),
                  weightAcc(i),
                  Eigen::Vector2f(targetPositionAcc(i, 0), targetPositionAcc(i, 1)),
                  proj);
            }
          },
          R"(Adds multiple vertex projection constraints to the error function.

        :param vertex_index: A numpy array of size n for the indices of the vertices to project.
        :param weight: A numpy array of size n for the weights of the constraints.
        :param target_position: A numpy array of shape (n, 2) for the target 2D positions.
        :param projection: A numpy array of shape (n, 3, 4) for the projection matrices.)",
          py::arg("vertex_index"),
          py::arg("weight"),
          py::arg("target_position"),
          py::arg("projection"));
}

void defPlaneErrorFunction(py::module_& m) {
  py::class_<mm::PlaneDataT<float>>(m, "PlaneData")
      .def(
          "__repr__",
          [](const mm::PlaneDataT<float>& self) {
            return fmt::format(
                "PlaneData(parent={}, weight={}, offset=[{:.3f}, {:.3f}, {:.3f}], normal=[{:.3f}, {:.3f}, {:.3f}], d={:.3f})",
                self.parent,
                self.weight,
                self.offset.x(),
                self.offset.y(),
                self.offset.z(),
                self.normal.x(),
                self.normal.y(),
                self.normal.z(),
                self.d);
          })
      .def_readonly("parent", &mm::PlaneDataT<float>::parent, "The parent joint index")
      .def_readonly("weight", &mm::PlaneDataT<float>::weight, "The weight of the constraint")
      .def_readonly("offset", &mm::PlaneDataT<float>::offset, "The offset in parent space")
      .def_readonly("normal", &mm::PlaneDataT<float>::normal, "The normal of the plane")
      .def_readonly("d", &mm::PlaneDataT<float>::d, "The d parameter in the plane equation");

  py::class_<
      mm::PlaneErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::PlaneErrorFunction>>(
      m,
      "PlaneErrorFunction",
      R"(The PlaneErrorFunction computes the point-on-plane error or point-in-half-plane error.
For point-on-plane error (above = false), it computes the signed distance of a point to a plane
using the plane equation. For half-plane error (above = true), the error is zero when the
distance is greater than zero (ie. the point being above).)")
      .def(
          "__repr__",
          [](const mm::PlaneErrorFunction& self) {
            return fmt::format(
                "PlaneErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 bool above,
                 float lossAlpha,
                 float lossC,
                 float weight) -> std::shared_ptr<mm::PlaneErrorFunction> {
                validateWeight(weight, "weight");
                auto result =
                    std::make_shared<mm::PlaneErrorFunction>(character, above, lossAlpha, lossC);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a PlaneErrorFunction.

            :param character: The character to use.
            :param above: True means "above the plane" (half plane inequality), false means "on the plane" (equality).
            :param alpha: P-norm to use; 2 is Euclidean distance, 1 is L1 norm, 0 is Cauchy norm.
            :param c: c parameter in the generalized loss function, this is roughly equivalent to normalizing by the standard deviation in a Gaussian distribution.
            :param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::arg("above") = false,
          py::kw_only(),
          py::arg("alpha") = 2.0f,
          py::arg("c") = 1.0f,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::PlaneErrorFunction& self,
             const Eigen::Vector3f& offset,
             const Eigen::Vector3f& normal,
             float d,
             int parent,
             float weight,
             const std::string& name) {
            validateJointIndex(parent, "parent", self.getSkeleton());
            validateWeight(weight, "weight");
            self.addConstraint(mm::PlaneDataT<float>(offset, normal, d, parent, weight, name));
          },
          R"(Adds a plane constraint to the error function.

        :param offset: The offset of the point in the parent's coordinate frame.
        :param normal: The normal of the plane.
        :param d: The d parameter in the plane equation: x.dot(normal) - d = 0.
        :param parent: The index of the parent joint.
        :param weight: The weight of the constraint.
        :param name: The name of the constraint (for debugging).)",
          py::arg("offset"),
          py::arg("normal"),
          py::arg("d"),
          py::arg("parent"),
          py::arg("weight") = 1.0f,
          py::arg("name") = std::string{})
      .def_property_readonly(
          "constraints",
          [](const mm::PlaneErrorFunction& self) { return self.getConstraints(); },
          "Returns the list of plane constraints.")
      .def(
          "add_constraints",
          [](mm::PlaneErrorFunction& self,
             const py::array_t<float>& normal,
             const py::array_t<float>& d,
             const py::array_t<int>& parent,
             const std::optional<py::array_t<float>>& offset,
             const std::optional<py::array_t<float>>& weight,
             const std::optional<std::vector<std::string>>& name) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(offset, "offset", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(normal, "normal", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(d, "d", {nConsIdx}, {"n_cons"});
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

            auto offsetAcc =
                offset.has_value() ? std::make_optional(offset->unchecked<2>()) : std::nullopt;
            auto normalAcc = normal.unchecked<2>();
            auto dAcc = d.unchecked<1>();
            auto parentAcc = parent.unchecked<1>();
            auto weightAcc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            py::gil_scoped_release release;

            for (py::ssize_t i = 0; i < parent.shape(0); ++i) {
              self.addConstraint(
                  mm::PlaneDataT<float>(
                      offsetAcc.has_value()
                          ? Eigen::Vector3f(
                                (*offsetAcc)(i, 0), (*offsetAcc)(i, 1), (*offsetAcc)(i, 2))
                          : Eigen::Vector3f::Zero(),
                      Eigen::Vector3f(normalAcc(i, 0), normalAcc(i, 1), normalAcc(i, 2)),
                      dAcc(i),
                      parentAcc(i),
                      weightAcc.has_value() ? (*weightAcc)(i) : 1.0f,
                      name.has_value() ? name->at(i) : std::string{}));
            }
          },
          R"(Adds multiple plane constraints to the error function.

        :param offset: A numpy array of shape (n, 3) for the offsets of the points in the parent's coordinate frame.
        :param normal: A numpy array of shape (n, 3) for the normals of the planes.
        :param d: A numpy array of size n for the d parameters in the plane equations: x.dot(normal) - d = 0.
        :param parent: A numpy array of size n for the indices of the parent joints.
        :param weight: A numpy array of size n for the weights of the constraints.
        :param name: An optional list of names for the constraints (for debugging).)",
          py::arg("normal"),
          py::arg("d"),
          py::arg("parent"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = std::nullopt,
          py::arg("name") = std::optional<std::vector<std::string>>{})
      .def(
          "clear_constraints",
          &mm::PlaneErrorFunction::clearConstraints,
          "Clears all plane constraints from the error function.");
}

void defVertexVertexDistanceErrorFunction(py::module_& m) {
  py::class_<mm::VertexVertexDistanceConstraintT<float>>(m, "VertexVertexDistanceConstraint")
      .def(
          "__repr__",
          [](const mm::VertexVertexDistanceConstraintT<float>& self) {
            return fmt::format(
                "VertexVertexDistanceConstraint(vertex_index1={}, vertex_index2={}, weight={}, target_distance={})",
                self.vertexIndex1,
                self.vertexIndex2,
                self.weight,
                self.targetDistance);
          })
      .def_readonly(
          "vertex_index1",
          &mm::VertexVertexDistanceConstraintT<float>::vertexIndex1,
          "The index of the first vertex")
      .def_readonly(
          "vertex_index2",
          &mm::VertexVertexDistanceConstraintT<float>::vertexIndex2,
          "The index of the second vertex")
      .def_readonly(
          "weight",
          &mm::VertexVertexDistanceConstraintT<float>::weight,
          "The weight of the constraint")
      .def_readonly(
          "target_distance",
          &mm::VertexVertexDistanceConstraintT<float>::targetDistance,
          "The target distance between the two vertices");

  py::class_<
      mm::VertexVertexDistanceErrorFunctionT<float>,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::VertexVertexDistanceErrorFunctionT<float>>>(
      m,
      "VertexVertexDistanceErrorFunction",
      R"(Error function that minimizes the distance between pairs of vertices on the character's mesh.

This is useful for constraints where you want to maintain specific distances between
different parts of the character mesh, such as keeping fingers at a certain distance
or maintaining the width of body parts.)")
      .def(
          "__repr__",
          [](const mm::VertexVertexDistanceErrorFunctionT<float>& self) {
            return fmt::format(
                "VertexVertexDistanceErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 float weight) -> std::shared_ptr<mm::VertexVertexDistanceErrorFunctionT<float>> {
                validateWeight(weight, "weight");
                auto result =
                    std::make_shared<mm::VertexVertexDistanceErrorFunctionT<float>>(character);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a VertexVertexDistanceErrorFunction.

            :param character: The character to use.
            :param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::VertexVertexDistanceErrorFunctionT<float>& self,
             int vertexIndex1,
             int vertexIndex2,
             float weight,
             float targetDistance) {
            validateVertexIndex(vertexIndex1, "vertex_index1", self.getCharacter());
            validateVertexIndex(vertexIndex2, "vertex_index2", self.getCharacter());
            validateWeight(weight, "weight");
            self.addConstraint(vertexIndex1, vertexIndex2, weight, targetDistance);
          },
          R"(Adds a vertex-to-vertex distance constraint to the error function.

        :param vertex_index1: The index of the first vertex.
        :param vertex_index2: The index of the second vertex.
        :param weight: The weight of the constraint.
        :param target_distance: The desired distance between the two vertices.)",
          py::arg("vertex_index1"),
          py::arg("vertex_index2"),
          py::arg("weight"),
          py::arg("target_distance"))
      .def(
          "add_constraints",
          [](mm::VertexVertexDistanceErrorFunctionT<float>& self,
             const py::array_t<int>& vertexIndex1,
             const py::array_t<int>& vertexIndex2,
             const py::array_t<float>& weight,
             const py::array_t<float>& targetDistance) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(vertexIndex1, "vertex_index1", {nConsIdx}, {"n_cons"});
            validateVertexIndex(vertexIndex1, "vertex_index1", self.getCharacter());
            validator.validate(vertexIndex2, "vertex_index2", {nConsIdx}, {"n_cons"});
            validateVertexIndex(vertexIndex2, "vertex_index2", self.getCharacter());
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});
            validateWeights(weight, "weight");
            validator.validate(targetDistance, "target_distance", {nConsIdx}, {"n_cons"});

            auto vertexIndex1Acc = vertexIndex1.unchecked<1>();
            auto vertexIndex2Acc = vertexIndex2.unchecked<1>();
            auto weightAcc = weight.unchecked<1>();
            auto targetDistanceAcc = targetDistance.unchecked<1>();

            py::gil_scoped_release release;

            for (py::ssize_t i = 0; i < vertexIndex1.shape(0); ++i) {
              self.addConstraint(
                  vertexIndex1Acc(i), vertexIndex2Acc(i), weightAcc(i), targetDistanceAcc(i));
            }
          },
          R"(Adds multiple vertex-to-vertex distance constraints to the error function.

        :param vertex_index1: A numpy array of indices for the first vertices.
        :param vertex_index2: A numpy array of indices for the second vertices.
        :param weight: A numpy array of weights for the constraints.
        :param target_distance: A numpy array of desired distances between vertex pairs.)",
          py::arg("vertex_index1"),
          py::arg("vertex_index2"),
          py::arg("weight"),
          py::arg("target_distance"))
      .def(
          "clear_constraints",
          &mm::VertexVertexDistanceErrorFunctionT<float>::clearConstraints,
          "Clears all vertex-to-vertex distance constraints from the error function.")
      .def_property_readonly(
          "constraints",
          [](const mm::VertexVertexDistanceErrorFunctionT<float>& self) {
            return self.getConstraints();
          },
          "Returns the list of vertex-to-vertex distance constraints.")
      .def(
          "num_constraints",
          &mm::VertexVertexDistanceErrorFunctionT<float>::numConstraints,
          "Returns the number of constraints.");
}

void defJointToJointDistanceErrorFunction(py::module_& m) {
  py::class_<mm::JointToJointDistanceConstraintT<float>>(m, "JointToJointDistanceConstraint")
      .def(
          "__repr__",
          [](const mm::JointToJointDistanceConstraintT<float>& self) {
            return fmt::format(
                "JointToJointDistanceConstraint(joint1={}, joint2={}, offset1=[{:.3f}, {:.3f}, {:.3f}], offset2=[{:.3f}, {:.3f}, {:.3f}], weight={}, target_distance={})",
                self.joint1,
                self.joint2,
                self.offset1.x(),
                self.offset1.y(),
                self.offset1.z(),
                self.offset2.x(),
                self.offset2.y(),
                self.offset2.z(),
                self.weight,
                self.targetDistance);
          })
      .def_readonly(
          "joint1",
          &mm::JointToJointDistanceConstraintT<float>::joint1,
          "The index of the first joint")
      .def_readonly(
          "joint2",
          &mm::JointToJointDistanceConstraintT<float>::joint2,
          "The index of the second joint")
      .def_readonly(
          "offset1",
          &mm::JointToJointDistanceConstraintT<float>::offset1,
          "The offset from joint1 in the local coordinate system of joint1")
      .def_readonly(
          "offset2",
          &mm::JointToJointDistanceConstraintT<float>::offset2,
          "The offset from joint2 in the local coordinate system of joint2")
      .def_readonly(
          "weight",
          &mm::JointToJointDistanceConstraintT<float>::weight,
          "The weight of the constraint")
      .def_readonly(
          "target_distance",
          &mm::JointToJointDistanceConstraintT<float>::targetDistance,
          "The target distance between the two points");

  py::class_<
      mm::JointToJointDistanceErrorFunctionT<float>,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::JointToJointDistanceErrorFunctionT<float>>>(
      m,
      "JointToJointDistanceErrorFunction",
      R"(Error function that penalizes deviation from a target distance between two points attached to different joints.

This is useful for enforcing distance constraints between different parts of a character,
such as maintaining a fixed distance between hands or ensuring two joints stay a certain distance apart.)")
      .def(
          "__repr__",
          [](const mm::JointToJointDistanceErrorFunctionT<float>& self) {
            return fmt::format(
                "JointToJointDistanceErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.getConstraints().size());
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 float weight) -> std::shared_ptr<mm::JointToJointDistanceErrorFunctionT<float>> {
                validateWeight(weight, "weight");
                auto result =
                    std::make_shared<mm::JointToJointDistanceErrorFunctionT<float>>(character);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a JointToJointDistanceErrorFunction.

            :param character: The character to use.
            :param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::JointToJointDistanceErrorFunctionT<float>& self,
             size_t joint1,
             const Eigen::Vector3f& offset1,
             size_t joint2,
             const Eigen::Vector3f& offset2,
             float targetDistance,
             float weight) {
            validateJointIndex(joint1, "joint1", self.getSkeleton());
            validateJointIndex(joint2, "joint2", self.getSkeleton());
            validateWeight(weight, "weight");
            self.addConstraint(joint1, offset1, joint2, offset2, targetDistance, weight);
          },
          R"(Adds a joint-to-joint distance constraint to the error function.

        :param joint1: The index of the first joint.
        :param offset1: The offset from joint1 in the local coordinate system of joint1.
        :param joint2: The index of the second joint.
        :param offset2: The offset from joint2 in the local coordinate system of joint2.
        :param target_distance: The desired distance between the two points in world space.
        :param weight: The weight of the constraint.)",
          py::arg("joint1"),
          py::arg("offset1"),
          py::arg("joint2"),
          py::arg("offset2"),
          py::arg("target_distance"),
          py::arg("weight") = 1.0f)
      .def(
          "add_constraints",
          [](mm::JointToJointDistanceErrorFunctionT<float>& self,
             const py::array_t<int>& joint1,
             const py::array_t<float>& offset1,
             const py::array_t<int>& joint2,
             const py::array_t<float>& offset2,
             const py::array_t<float>& targetDistance,
             const std::optional<py::array_t<float>>& weight) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(joint1, "joint1", {nConsIdx}, {"n_cons"});
            validateJointIndex(joint1, "joint1", self.getSkeleton());
            validator.validate(offset1, "offset1", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(joint2, "joint2", {nConsIdx}, {"n_cons"});
            validateJointIndex(joint2, "joint2", self.getSkeleton());
            validator.validate(offset2, "offset2", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(targetDistance, "target_distance", {nConsIdx}, {"n_cons"});
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});
            validateWeights(weight, "weight");

            auto joint1Acc = joint1.unchecked<1>();
            auto offset1Acc = offset1.unchecked<2>();
            auto joint2Acc = joint2.unchecked<1>();
            auto offset2Acc = offset2.unchecked<2>();
            auto targetDistanceAcc = targetDistance.unchecked<1>();
            auto weightAcc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            py::gil_scoped_release release;

            for (py::ssize_t i = 0; i < joint1.shape(0); ++i) {
              self.addConstraint(
                  joint1Acc(i),
                  Eigen::Vector3f(offset1Acc(i, 0), offset1Acc(i, 1), offset1Acc(i, 2)),
                  joint2Acc(i),
                  Eigen::Vector3f(offset2Acc(i, 0), offset2Acc(i, 1), offset2Acc(i, 2)),
                  targetDistanceAcc(i),
                  weightAcc.has_value() ? (*weightAcc)(i) : 1.0f);
            }
          },
          R"(Adds multiple joint-to-joint distance constraints to the error function.

        :param joint1: A numpy array of indices for the first joints.
        :param offset1: A numpy array of shape (n, 3) for offsets from joint1 in local coordinates.
        :param joint2: A numpy array of indices for the second joints.
        :param offset2: A numpy array of shape (n, 3) for offsets from joint2 in local coordinates.
        :param target_distance: A numpy array of desired distances between point pairs.
        :param weight: A numpy array of weights for the constraints.)",
          py::arg("joint1"),
          py::arg("offset1"),
          py::arg("joint2"),
          py::arg("offset2"),
          py::arg("target_distance"),
          py::arg("weight") = std::nullopt)
      .def(
          "clear_constraints",
          &mm::JointToJointDistanceErrorFunctionT<float>::clearConstraints,
          "Clears all joint-to-joint distance constraints from the error function.")
      .def_property_readonly(
          "constraints",
          [](const mm::JointToJointDistanceErrorFunctionT<float>& self) {
            return self.getConstraints();
          },
          "Returns the list of joint-to-joint distance constraints.");
}

} // namespace

void addErrorFunctions(py::module_& m) {
  py::class_<
      mm::LimitErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::LimitErrorFunction>>(
      m,
      "LimitErrorFunction",
      "A skeleton error function that penalizes joints that exceed their limits.")
      .def(
          "__repr__",
          [](const mm::LimitErrorFunction& self) {
            return fmt::format("LimitErrorFunction(weight={})", self.getWeight());
          })
      .def(
          py::init<>([](const mm::Character& character, float weight) {
            validateWeight(weight, "weight");
            auto result = std::make_shared<mm::LimitErrorFunction>(character);
            result->setWeight(weight);
            return result;
          }),
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f);

  py::class_<
      mm::ModelParametersErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::ModelParametersErrorFunction>>(
      m,
      "ModelParametersErrorFunction",
      "A skeleton error function that penalizes the difference between the target model parameters and the current model parameters.")
      .def(
          "__repr__",
          [](const mm::ModelParametersErrorFunction& self) {
            return fmt::format("ModelParametersErrorFunction(weight={})", self.getWeight());
          })
      .def(
          py::init<>([](const mm::Character& character,
                        const std::optional<py::array_t<float>>& targetParams,
                        const std::optional<py::array_t<float>>& weights,
                        float weight) {
            validateWeight(weight, "weight");
            validateWeights(weights, "weights");
            auto result = std::make_shared<mm::ModelParametersErrorFunction>(character);
            result->setWeight(weight);

            if (targetParams.has_value() || weights.has_value()) {
              const auto nParams = character.parameterTransform.numAllModelParameters();
              result->setTargetParameters(
                  arrayToVec(targetParams, nParams, 0.0f, "target_parameters"),
                  arrayToVec(weights, nParams, 1.0f, "weights"));
            }
            return result;
          }),
          R"(Initialize a ModelParametersErrorFunction.

:param character: The character to use.
:param target_parameters: The target model parameters to achieve.
:param weights: Optional weights for each parameter, defaulting to 1 if not provided.
:param weight: The weight applied to the error function.)",

          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::arg("target_parameters") = std::optional<py::array_t<float>>{},
          py::kw_only(),
          py::arg("weights") = std::optional<py::array_t<float>>{},
          py::arg("weight") = 1.0f)
      .def(
          "set_target_parameters",
          [](mm::ModelParametersErrorFunction& self,
             const py::array_t<float>& targetParams,
             const std::optional<py::array_t<float>>& weights) {
            const auto nParams = self.getParameterTransform().numAllModelParameters();
            self.setTargetParameters(
                arrayToVec(targetParams, nParams, "target_parameters"),
                arrayToVec(weights, nParams, 1.0f, "weights"));
          },
          R"(Sets the target model parameters and optional weights.

:param target_parameters: The target model parameters to achieve.
:param weights: Optional weights for each parameter, defaulting to 1 if not provided.)",
          py::arg("target_parameters"),
          py::arg("weights") = std::optional<py::array_t<float>>{});

  py::class_<mm::PositionDataT<float>>(m, "PositionData")
      .def(
          "__repr__",
          [](const mm::PositionDataT<float>& self) {
            return fmt::format(
                "PositionData(parent={}, weight={}, offset=[{:.3f}, {:.3f}, {:.3f}], target=[{:.3f}, {:.3f}, {:.3f}])",
                self.parent,
                self.weight,
                self.offset.x(),
                self.offset.y(),
                self.offset.z(),
                self.target.x(),
                self.target.y(),
                self.target.z());
          })
      .def_readonly("parent", &mm::PositionDataT<float>::parent, "The parent joint index")
      .def_readonly("weight", &mm::PositionDataT<float>::weight, "The weight of the constraint")
      .def_readonly("offset", &mm::PositionDataT<float>::offset, "The offset in parent space")
      .def_readonly("target", &mm::PositionDataT<float>::target, "The target position");

  py::class_<
      mm::PositionErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::PositionErrorFunction>>(
      m,
      "PositionErrorFunction",
      R"(A skeleton error function that tries to move locators to a target position.

Uses a generalized loss function that support various forms of losses such as L1, L2, and Cauchy.  Details can be found in the paper 'A General and Adaptive Robust Loss Function' by Jonathan T. Barron.
      )")
      .def(
          "__repr__",
          [](const mm::PositionErrorFunction& self) {
            return fmt::format(
                "PositionErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character, float lossAlpha, float lossC, float weight)
                  -> std::shared_ptr<mm::PositionErrorFunction> {
                validateWeight(weight, "weight");
                auto result =
                    std::make_shared<mm::PositionErrorFunction>(character, lossAlpha, lossC);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a PositionErrorFunction.

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
          [](mm::PositionErrorFunction& errf,
             int parent,
             const Eigen::Vector3f& target,
             const std::optional<Eigen::Vector3f>& offset,
             float weight,
             const std::string& name) {
            validateJointIndex(parent, "parent", errf.getSkeleton());
            validateWeight(weight, "weight");
            errf.addConstraint(
                mm::PositionData(
                    offset.value_or(Eigen::Vector3f::Zero()), target, parent, weight, name));
          },
          R"(Adds a new constraint to the error function.

:param parent: The index of the joint that is the parent of the locator.
:param offset: The offset of the locator in the parent joint local space.
:param target: The target position of the locator.
:param weight: The weight of the constraint.
:param name: The name of the constraint (for debugging).
        )",
          py::arg("parent"),
          py::kw_only(),
          py::arg("target"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = 1.0f,
          py::arg("name") = std::string{})
      .def_property_readonly(
          "constraints",
          [](const mm::PositionErrorFunction& self) { return self.getConstraints(); },
          "Returns the list of position constraints.")
      .def(
          "clear_constraints",
          [](mm::PositionErrorFunction& self) { self.clearConstraints(); },
          "Clears all constraints from the error function.")
      .def(
          "add_constraints",
          [](mm::PositionErrorFunction& errf,
             const py::array_t<int>& parent,
             const py::array_t<float, 2>& target,
             const std::optional<py::array_t<float, 2>>& offset,
             const std::optional<py::array_t<float>>& weight,
             const std::optional<std::vector<std::string>>& name) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(parent, "parent", {nConsIdx}, {"n_cons"});
            validateJointIndex(parent, "parent", errf.getSkeleton());
            validator.validate(offset, "offset", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(target, "target", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});
            validateWeights(weight, "weight");

            if (name.has_value() && name->size() != parent.shape(0)) {
              throw std::runtime_error(
                  fmt::format(
                      "Invalid names; expected {} names but got {}",
                      parent.shape(0),
                      name->size()));
            }

            auto parent_acc = parent.unchecked<1>();
            auto offset_acc =
                offset.has_value() ? std::make_optional(offset->unchecked<2>()) : std::nullopt;
            auto target_acc = target.unchecked<2>();
            auto weight_acc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            py::gil_scoped_release release;

            for (size_t iCons = 0; iCons < parent_acc.shape(0); ++iCons) {
              errf.addConstraint(
                  mm::PositionData(
                      offset_acc.has_value() ? Eigen::Vector3f(
                                                   (*offset_acc)(iCons, 0),
                                                   (*offset_acc)(iCons, 1),
                                                   (*offset_acc)(iCons, 2))
                                             : Eigen::Vector3f::Zero(),
                      Eigen::Vector3f(
                          target_acc(iCons, 0), target_acc(iCons, 1), target_acc(iCons, 2)),
                      parent_acc(iCons),
                      weight_acc.has_value() ? (*weight_acc)(iCons) : 1.0f,
                      name.has_value() ? name->at(iCons) : std::string{}));
            }
          },
          R"(Adds a new constraint to the error function.

:param parent: The index of the joints that are the parents of the locator.
:param offset: The offset of the locators from the parent joint.
:param target: The target positions of the locator.
:param weight: The weight of the constraint.
:param names: The names of the constraints (for debugging).)",
          py::arg("parent"),
          py::kw_only(),
          py::arg("target"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = std::nullopt,
          py::arg("name") = std::optional<std::vector<std::string>>{});

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

  // Fixed axis error functions.
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

  py::enum_<mm::RotationErrorType>(m, "RotationErrorType")
      .value(
          "RotationMatrixDifference",
          mm::RotationErrorType::RotationMatrixDifference,
          R"(Frobenius norm of rotation matrix difference: ||R1 - R2||_F^2.

This is the default method. It computes the squared Frobenius norm of the
difference between two rotation matrices. While not a geodesic distance,
it has smooth derivatives everywhere and is computationally efficient.)")
      .value(
          "QuaternionLogMap",
          mm::RotationErrorType::QuaternionLogMap,
          R"(Logarithmic map of relative rotation: ||log(R1^{-1} * R2)||^2.

This method computes the squared norm of the logarithmic map of the relative
rotation quaternion. This gives the squared geodesic distance on SO(3), which
has a clear geometric interpretation. It uses numerically robust logmap
computation with Taylor series for small angles.)");

  py::class_<
      mm::StateErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::StateErrorFunction>>(m, "StateErrorFunction")
      .def(
          "__repr__",
          [](const mm::StateErrorFunction& self) {
            return fmt::format("StateErrorFunction(weight={})", self.getWeight());
          })
      .def(
          py::init<>([](const mm::Character& character,
                        float weight,
                        float positionWeight,
                        float rotationWeight,
                        const std::optional<py::array_t<float>>& jointPositionWeights,
                        const std::optional<py::array_t<float>>& jointRotationWeights,
                        mm::RotationErrorType rotationErrorType) {
            validateWeight(weight, "weight");
            validateWeight(positionWeight, "position_weight");
            validateWeight(rotationWeight, "rotation_weight");
            validateWeights(jointPositionWeights, "joint_position_weights");
            validateWeights(jointRotationWeights, "joint_rotation_weights");

            auto result = std::make_shared<mm::StateErrorFunction>(character, rotationErrorType);
            result->setWeight(weight);
            result->setWeights(positionWeight, rotationWeight);

            if (jointPositionWeights.has_value() || jointRotationWeights.has_value()) {
              result->setTargetWeights(
                  getJointWeights(
                      jointPositionWeights, character.skeleton, "joint_position_weights"),
                  getJointWeights(
                      jointRotationWeights, character.skeleton, "joint_rotation_weights"));
            }

            return result;
          }),
          R"(Constructs a StateErrorFunction for a given character.

:param character: The character for which the error function is defined.
:param weight: The weight for the error function. Defaults to 1.0.
:param position_weight: The weight for position errors. Defaults to 1.0.
:param rotation_weight: The weight for rotation errors. Defaults to 1.0.
:param joint_position_weights: Optional numpy array of per-joint position weights.
                               Must match the number of joints in the skeleton.
:param joint_rotation_weights: Optional numpy array of per-joint rotation weights.
                               Must match the number of joints in the skeleton.
:param rotation_error_type: The type of rotation error to use. Defaults to RotationMatrixDifference.
                            See :class:`RotationErrorType` for options.)",
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f,
          py::arg("position_weight") = 1.0f,
          py::arg("rotation_weight") = 1.0f,
          py::arg("joint_position_weights") = std::optional<py::array_t<float>>{},
          py::arg("joint_rotation_weights") = std::optional<py::array_t<float>>{},
          py::arg("rotation_error_type") = mm::RotationErrorType::RotationMatrixDifference)
      .def(
          "set_target_state",
          [](mm::StateErrorFunction& self, const py::array_t<float>& targetStateArray) {
            if (targetStateArray.ndim() != 2 || targetStateArray.shape(1) != 8) {
              throw std::runtime_error("Expected target state array of shape (njoints, 8)");
            }

            const auto targetTransforms = toTransformList(targetStateArray);
            if (targetTransforms.size() != self.getSkeleton().joints.size()) {
              throw std::runtime_error(
                  fmt::format(
                      "Expected target state array of shape (njoints, 8) where nJoints={} but got {}.",
                      self.getSkeleton().joints.size(),
                      getDimStr(targetStateArray)));
            }

            self.setTargetState(targetTransforms);
          },
          R"(Sets the target skeleton state for the error function.

:param target_state: A numpy array of shape (njoints, 8) where each row contains
                     3D position, 4D quaternion (rotation), and 1D scale for each joint.)",
          py::arg("target_state"))
      .def_property(
          "joint_position_weights",
          [](const mm::StateErrorFunction& self) { return self.getPositionWeights(); },
          [](mm::StateErrorFunction& self, const py::array_t<float>& weights) {
            validateWeights(weights, "weights");
            self.setPositionTargetWeights(
                getJointWeights(weights, self.getSkeleton(), "joint_position_weights"));
          },
          R"(The per-joint position weights.)")
      .def_property(
          "joint_rotation_weights",
          [](const mm::StateErrorFunction& self) { return self.getRotationWeights(); },
          [](mm::StateErrorFunction& self, const py::array_t<float>& weights) {
            validateWeights(weights, "weights");
            self.setRotationTargetWeights(
                getJointWeights(weights, self.getSkeleton(), "joint_position_weights"));
          },
          R"(The per-joint rotation weights.)");

  py::enum_<mm::VertexConstraintType>(m, "VertexConstraintType")
      .value("Position", mm::VertexConstraintType::Position, "Target the vertex position")
      .value(
          "Plane",
          mm::VertexConstraintType::Plane,
          "Point-to-plane distance using the target normal")
      .value(
          "Normal",
          mm::VertexConstraintType::Normal,
          "Point-to-plane distance using the source (body) normal")
      .value(
          "SymmetricNormal",
          mm::VertexConstraintType::SymmetricNormal,
          "Point-to-plane using a 50/50 mix of source and target normal");

  py::class_<mm::VertexConstraint>(m, "VertexConstraint")
      .def(
          "__repr__",
          [](const mm::VertexConstraint& self) {
            return fmt::format(
                "VertexConstraint(vertex_index={}, weight={}, target_position=[{:.3f}, {:.3f}, {:.3f}])",
                self.vertexIndex,
                self.weight,
                self.targetPosition.x(),
                self.targetPosition.y(),
                self.targetPosition.z());
          })
      .def_readonly(
          "vertex_index",
          &mm::VertexConstraint::vertexIndex,
          "The index of the vertex to constrain.")
      .def_readonly("weight", &mm::VertexConstraint::weight, "The weight of the constraint.")
      .def_readonly(
          "target_position",
          &mm::VertexConstraint::targetPosition,
          "The target position for the vertex.")
      .def_readonly(
          "target_normal",
          &mm::VertexConstraint::targetNormal,
          "The target normal for the vertex.");

  py::class_<
      mm::VertexErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::VertexErrorFunction>>(m, "VertexErrorFunction")
      .def(
          "__repr__",
          [](const mm::VertexErrorFunction& self) {
            return fmt::format(
                "VertexErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>([](const mm::Character& character,
                        mm::VertexConstraintType constraintType,
                        float weight,
                        size_t maxThreads) {
            validateWeight(weight, "weight");
            auto result =
                std::make_shared<mm::VertexErrorFunction>(character, constraintType, maxThreads);
            result->setWeight(weight);
            return result;
          }),
          R"(A skeleton error function that penalizes vertex distance to a target point.

  :param character: The character to use.
  :param constraint_type: The type of vertex constraint to apply.
  :param max_threads: The maximum number of threads to use for computation.)",
          py::arg("character"),
          py::arg("constraint_type") = mm::VertexConstraintType::Position,
          py::arg("weight") = 1.0f,
          py::arg("max_threads") = 0)
      .def(
          "add_constraint",
          [](mm::VertexErrorFunction& self,
             int vertexIndex,
             float weight,
             const Eigen::Vector3f& targetPosition,
             const Eigen::Vector3f& targetNormal) {
            validateVertexIndex(vertexIndex, "vertex_index", self.getCharacter());
            validateWeight(weight, "weight");
            self.addConstraint(vertexIndex, weight, targetPosition, targetNormal);
          },
          R"(Adds a vertex constraint to the error function.

  :param vertex_index: The index of the vertex to constrain.
  :param weight: The weight of the constraint.
  :param target_position: The target position for the vertex.
  :param target_normal: The target normal for the vertex.)",
          py::arg("vertex_index"),
          py::arg("weight"),
          py::arg("target_position"),
          py::arg("target_normal"))
      .def(
          "add_constraints",
          [](mm::VertexErrorFunction& errf,
             const py::array_t<int>& vertexIndex,
             const py::array_t<float, 2>& targetPosition,
             const py::array_t<float, 2>& targetNormal,
             const py::array_t<float>& weight) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(vertexIndex, "vertex_index", {nConsIdx}, {"n_cons"});
            validateVertexIndex(vertexIndex, "vertex_index", errf.getCharacter());
            validator.validate(targetPosition, "target_position", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(targetNormal, "target_normal", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});

            auto vertex_index_acc = vertexIndex.unchecked<1>();
            auto target_position_acc = targetPosition.unchecked<2>();
            auto target_normal_acc = targetNormal.unchecked<2>();
            auto weight_acc = weight.unchecked<1>();

            // Validate weights
            validateWeights(weight, "weight");

            py::gil_scoped_release release;

            for (size_t iCons = 0; iCons < vertex_index_acc.shape(0); ++iCons) {
              errf.addConstraint(
                  vertex_index_acc(iCons),
                  weight_acc(iCons),
                  Eigen::Vector3f(
                      target_position_acc(iCons, 0),
                      target_position_acc(iCons, 1),
                      target_position_acc(iCons, 2)),
                  Eigen::Vector3f(
                      target_normal_acc(iCons, 0),
                      target_normal_acc(iCons, 1),
                      target_normal_acc(iCons, 2)));
            }
          },
          R"(Adds multiple vertex constraints to the error function.

  :param vertex_index: The indices of the vertices to constrain.
  :param target_position: The target positions for the vertices.
  :param target_normal: The target normals for the vertices.
  :param weight: The weights of the constraints.)",
          py::arg("vertex_index"),
          py::arg("target_position"),
          py::arg("target_normal"),
          py::arg("weight"))
      .def(
          "clear_constraints",
          &mm::VertexErrorFunction::clearConstraints,
          "Clears all vertex constraints from the error function.")
      .def_property_readonly(
          "constraints",
          [](const mm::VertexErrorFunction& self) { return self.getConstraints(); },
          "Returns the list of vertex constraints.");

  py::class_<mm::PointTriangleVertexConstraint>(m, "PointTriangleVertexConstraint")
      .def(
          "__repr__",
          [](const mm::PointTriangleVertexConstraint& self) {
            return fmt::format(
                "PointTriangleVertexConstraint(src_vertex_index={}, depth={:.3f}, tgt_triangle_indices=[{} {} {}], tgt_bary_coords=[{:.3f}, {:.3f}, {:.3f}], weight={:.3f})",
                self.srcVertexIndex,
                self.depth,
                self.tgtTriangleIndices.x(),
                self.tgtTriangleIndices.y(),
                self.tgtTriangleIndices.z(),
                self.tgtTriangleBaryCoords.x(),
                self.tgtTriangleBaryCoords.y(),
                self.tgtTriangleBaryCoords.z(),
                self.weight);
          })
      .def_readonly(
          "src_vertex_index",
          &mm::PointTriangleVertexConstraint::srcVertexIndex,
          "The index of the vertex to constrain.")
      .def_readonly(
          "depth", &mm::PointTriangleVertexConstraint::depth, "The depth of the constraint.")
      .def_readonly(
          "weight", &mm::PointTriangleVertexConstraint::weight, "The weight of the constraint.")
      .def_readonly(
          "tgt_triangle_indices",
          &mm::PointTriangleVertexConstraint::tgtTriangleIndices,
          "The target triangle indices.")
      .def_readonly(
          "tgt_bary_coords",
          &mm::PointTriangleVertexConstraint::tgtTriangleBaryCoords,
          "The target barycentric coordinates.");

  py::class_<
      mm::PointTriangleVertexErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::PointTriangleVertexErrorFunction>>(m, "PointTriangleVertexErrorFunction")
      .def(
          "__repr__",
          [](const mm::PointTriangleVertexErrorFunction& self) {
            return fmt::format("PointTriangleVertexErrorFunction(weight={})", self.getWeight());
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 mm::VertexConstraintType constraintType,
                 float weight) -> std::shared_ptr<mm::PointTriangleVertexErrorFunction> {
                if (!character.mesh || !character.skinWeights) {
                  throw std::runtime_error("No mesh or skin weights found");
                }

                auto result = std::make_shared<mm::PointTriangleVertexErrorFunction>(
                    character, constraintType);
                result->setWeight(weight);
                return result;
              }),
          R"(A skeleton error function that penalizes the distance between a vertex and the location within a triangle.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::arg("type") = mm::VertexConstraintType::Position,
          py::arg("weight") = 1.0f)
      .def(
          "add_constraints",
          [](mm::PointTriangleVertexErrorFunction& errf,
             const py::array_t<int>& srcVertexIndex,
             const py::array_t<int, 3>& tgtTriangleIndex,
             const py::array_t<float, 3>& tgtTriangleBaryCoord,
             const py::array_t<float>& depth,
             const py::array_t<float>& weight) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(srcVertexIndex, "source_vertex_index", {nConsIdx}, {"n_cons"});
            validateVertexIndex(srcVertexIndex, "source_vertex_index", errf.getCharacter());
            validator.validate(
                tgtTriangleIndex, "target_triangle_index", {nConsIdx, 3}, {"n_cons", "idx"});
            validateVertexIndex(tgtTriangleIndex, "target_triangle_index", errf.getCharacter());
            validator.validate(
                tgtTriangleBaryCoord,
                "target_triangle_bary_coord",
                {nConsIdx, 3},
                {"n_cons", "xyz"});
            validator.validate(depth, "depth", {nConsIdx}, {"n_cons"});
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});

            auto src_vertex_index_acc = srcVertexIndex.unchecked<1>();
            auto tgt_triangle_index_acc = tgtTriangleIndex.unchecked<2>();
            auto tgt_triangle_bary_coord_acc = tgtTriangleBaryCoord.unchecked<2>();
            auto depth_acc = depth.unchecked<1>();
            auto weight_acc = weight.unchecked<1>();

            py::gil_scoped_release release;

            for (size_t iCons = 0; iCons < src_vertex_index_acc.shape(0); ++iCons) {
              errf.addConstraint(
                  src_vertex_index_acc(iCons),
                  Eigen::Vector3i(
                      tgt_triangle_index_acc(iCons, 0),
                      tgt_triangle_index_acc(iCons, 1),
                      tgt_triangle_index_acc(iCons, 2)),
                  Eigen::Vector3f(
                      tgt_triangle_bary_coord_acc(iCons, 0),
                      tgt_triangle_bary_coord_acc(iCons, 1),
                      tgt_triangle_bary_coord_acc(iCons, 2)),
                  depth_acc(iCons),
                  weight_acc(iCons));
            }
          },
          R"(Adds a new constraint to the error function.

  :param parents: The indices of the joints that are the parents of the locators.
  :param offsets: The offsets of the locators from the parent joints.
  :param targets: The target positions of the locators.
  :param weights: The weights of the constraints.
  :param names: The names of the constraints (for debugging).)",
          py::arg("source_vertex_index"),
          py::arg("target_triangle_index"),
          py::arg("target_triangle_bary_coord"),
          py::arg("depth"),
          py::arg("weight"))
      .def_property_readonly(
          "constraints",
          [](const mm::PointTriangleVertexErrorFunction& self) { return self.getConstraints(); },
          "Returns the list of point triangle constraints.")
      .def(
          "clear_constraints",
          &mm::PointTriangleVertexErrorFunction::clearConstraints,
          "Clears all vertex constraints from the error function.");

  py::class_<
      mm::PosePriorErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::PosePriorErrorFunction>>(m, "PosePriorErrorFunction")
      .def(
          "__repr__",
          [](const mm::PosePriorErrorFunction& self) {
            return fmt::format("PosePriorErrorFunction(weight={})", self.getWeight());
          })
      .def(
          py::init<>([](const mm::Character& character,
                        std::shared_ptr<const mm::Mppca> posePrior,
                        float weight) {
            validateWeight(weight, "weight");
            auto result = std::make_shared<mm::PosePriorErrorFunction>(character, posePrior);
            result->setWeight(weight);
            return result;
          }),
          R"(A skeleton error function that uses a mixture of PCA models pose prior to penalize deviations from expected poses.

  :param character: The character to use.
  :param pose_prior: The pose prior model to use.
  :param weight: The weight applied to the error function.)",
          py::arg("character"),
          py::arg("pose_prior"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "set_pose_prior",
          &mm::PosePriorErrorFunction::setPosePrior,
          R"(Sets the pose prior model for the error function.

  :param pose_prior: The pose prior model to use.)",
          py::arg("pose_prior"))
      .def(
          "log_probability",
          [](const mm::PosePriorErrorFunction& self, const py::array_t<float>& modelParameters) {
            return self.logProbability(
                toModelParameters(modelParameters, self.getParameterTransform()));
          },
          R"(Computes the log probability of the given model parameters under the pose prior.

  :param model_parameters: The model parameters to evaluate.
  :return: The log probability value.)",
          py::arg("model_parameters"));

  py::class_<mm::OrientationData>(m, "OrientationData")
      .def(
          "__repr__",
          [](const mm::OrientationData& self) {
            return fmt::format(
                "OrientationData(parent={}, weight={}, offset=[{:.3f}, {:.3f}, {:.3f}, {:.3f}], target=[{:.3f}, {:.3f}, {:.3f}, {:.3f}])",
                self.parent,
                self.weight,
                self.offset.x(),
                self.offset.y(),
                self.offset.z(),
                self.offset.w(),
                self.target.x(),
                self.target.y(),
                self.target.z(),
                self.target.w());
          })
      .def_readonly("parent", &mm::OrientationData::parent, "The parent joint index")
      .def_readonly("weight", &mm::OrientationData::weight, "The weight of the constraint")
      .def_readonly("offset", &mm::OrientationData::offset, "The offset in parent space")
      .def_readonly("target", &mm::OrientationData::target, "The target orientation");

  py::class_<
      mm::OrientationErrorFunctionT<float>,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::OrientationErrorFunctionT<float>>>(
      m,
      "OrientationErrorFunction",
      R"(A skeleton error function that minimizes the F-norm of the element-wise difference of a 3x3
rotation matrix to a target rotation.)")
      .def(
          "__repr__",
          [](const mm::OrientationErrorFunctionT<float>& self) {
            return fmt::format(
                "OrientationErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character, float lossAlpha, float lossC, float weight)
                  -> std::shared_ptr<mm::OrientationErrorFunctionT<float>> {
                validateWeight(weight, "weight");
                auto result = std::make_shared<mm::OrientationErrorFunctionT<float>>(
                    character, lossAlpha, lossC);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize an OrientationErrorFunction.

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
          [](mm::OrientationErrorFunctionT<float>& self,
             const Eigen::Vector4f& target,
             int parent,
             const std::optional<Eigen::Vector4f>& offset,
             float weight,
             const std::string& name) {
            validateJointIndex(parent, "parent", self.getSkeleton());
            validateWeight(weight, "weight");
            self.addConstraint(
                mm::OrientationDataT<float>(
                    offset.has_value() ? toQuaternion(*offset) : Eigen::Quaternionf::Identity(),
                    toQuaternion(target),
                    parent,
                    weight,
                    name));
          },
          R"(Adds an orientation constraint to the error function.

      :param offset: The rotation offset in the parent space.
      :param target: The target rotation in global space.
      :param parent: The index of the parent joint.
      :param weight: The weight of the constraint.
      :param name: The name of the constraint (for debugging).)",
          py::arg("target"),
          py::arg("parent"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = 1.0f,
          py::arg("name") = std::string{})
      .def(
          "clear_constraints",
          [](mm::OrientationErrorFunctionT<float>& self) { self.clearConstraints(); },
          R"(Clears all constraints.)")
      .def_property_readonly(
          "constraints",
          [](const mm::OrientationErrorFunctionT<float>& self) { return self.getConstraints(); },
          "Returns the list of orientation constraints.")
      .def(
          "add_constraints",
          [](mm::OrientationErrorFunctionT<float>& self,
             const py::array_t<float>& target,
             const py::array_t<int>& parent,
             const std::optional<py::array_t<float>>& offset,
             const std::optional<py::array_t<float>>& weight,
             const std::optional<std::vector<std::string>>& name) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(offset, "offset", {nConsIdx, 4}, {"n_cons", "xyzw"});
            validator.validate(target, "target", {nConsIdx, 4}, {"n_cons", "xyzw"});
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

            auto offsetAcc =
                offset.has_value() ? std::make_optional(offset->unchecked<2>()) : std::nullopt;
            auto targetAcc = target.unchecked<2>();
            auto parentAcc = parent.unchecked<1>();
            auto weightAcc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            py::gil_scoped_release release;

            for (py::ssize_t i = 0; i < parent.shape(0); ++i) {
              validateJointIndex(parentAcc(i), "parent", self.getSkeleton());
              self.addConstraint(
                  mm::OrientationDataT<float>(
                      offsetAcc.has_value() ? Eigen::Quaternionf(
                                                  (*offsetAcc)(i, 3), // w
                                                  (*offsetAcc)(i, 0), // x
                                                  (*offsetAcc)(i, 1), // y
                                                  (*offsetAcc)(i, 2))
                                            : Eigen::Quaternionf::Identity(), // z
                      Eigen::Quaternionf(
                          targetAcc(i, 3), // w
                          targetAcc(i, 0), // x
                          targetAcc(i, 1), // y
                          targetAcc(i, 2)), // z
                      parentAcc(i),
                      weightAcc.has_value() ? (*weightAcc)(i) : 1.0f,
                      name.has_value() ? name->at(i) : std::string{}));
            }
          },
          R"(Adds multiple orientation constraints to the error function.

      :param target: A numpy array of shape (n, 4) for the (x, y, z, w) quaternion target rotations in global space.
      :param parent: A numpy array of size n for the indices of the parent joints.
      :param offset: A numpy array of shape (n, 4) for the (x, y, z, w) quaternion rotation offsets in the parent space.
      :param weight: A numpy array of size n for the weights of the constraints.
      :param name: An optional list of names for the constraints (for debugging).)",
          py::arg("target"),
          py::arg("parent"),
          py::arg("offset") = std::nullopt,
          py::arg("weight") = std::nullopt,
          py::arg("name") = std::nullopt);

  py::class_<
      mm::CollisionErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::CollisionErrorFunction>>(m, "CollisionErrorFunction")
      .def(
          "__repr__",
          [](const mm::CollisionErrorFunction& self) {
            return fmt::format(
                "CollisionErrorFunction(weight={}, collision_pairs={})",
                self.getWeight(),
                self.getCollisionPairs().size());
          })
      .def(
          py::init<>([](const mm::Character& character, float weight) {
            validateWeight(weight, "weight");
            auto result = std::make_shared<mm::CollisionErrorFunction>(character);
            result->setWeight(weight);
            return result;
          }),
          R"(A skeleton error function that penalizes collisions between character parts.

        :param character: The character to use.
        :param weight: The weight applied to the error function.)",
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "get_collision_pairs",
          [](const mm::CollisionErrorFunction& self) {
            const auto& collisionPairs = self.getCollisionPairs();
            py::array_t<float> result(
                std::vector<py::ssize_t>{(py::ssize_t)collisionPairs.size(), 2});
            auto resultAcc = result.mutable_unchecked<2>();
            for (size_t i = 0; i < collisionPairs.size(); ++i) {
              resultAcc(i, 0) = collisionPairs[i].x();
              resultAcc(i, 1) = collisionPairs[i].y();
            }
            return result;
          },
          R"(Returns the pairs of colliding elements.)");

  // Plane error function
  defPlaneErrorFunction(m);

  // Normal error function
  defNormalErrorFunction(m);

  // Distance error function
  defDistanceErrorFunction(m);

  // Projection error function
  defProjectionErrorFunction(m);

  // Vertex Projection error function
  defVertexProjectionErrorFunction(m);

  // Vertex-to-vertex distance error function
  defVertexVertexDistanceErrorFunction(m);

  // Joint-to-joint distance error function
  defJointToJointDistanceErrorFunction(m);
}

} // namespace pymomentum
