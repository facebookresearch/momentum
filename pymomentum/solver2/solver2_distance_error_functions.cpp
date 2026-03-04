/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pymomentum/solver2/solver2_error_functions.h>

#include <momentum/character/character.h>
#include <momentum/character_solver/distance_error_function.h>
#include <momentum/character_solver/joint_to_joint_distance_error_function.h>
#include <momentum/character_solver/joint_to_joint_position_error_function.h>
#include <momentum/character_solver/normal_error_function.h>
#include <momentum/character_solver/plane_error_function.h>
#include <momentum/character_solver/vertex_vertex_distance_error_function.h>
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

void addNormalErrorFunction(py::module_& m) {
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

void addDistanceErrorFunction(py::module_& m) {
  // Distance error function
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

void addPlaneErrorFunction(py::module_& m) {
  // Plane error function
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

void addVertexVertexDistanceErrorFunction(py::module_& m) {
  // Vertex-to-vertex distance error function
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

void addJointToJointDistanceErrorFunction(py::module_& m) {
  // Joint-to-joint distance error function
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

void addJointToJointPositionErrorFunction(py::module_& m) {
  // Joint-to-joint position error function
  py::class_<mm::JointToJointPositionDataT<float>>(m, "JointToJointPositionData")
      .def(
          "__repr__",
          [](const mm::JointToJointPositionDataT<float>& self) {
            return fmt::format(
                "JointToJointPositionData(source_joint={}, reference_joint={}, source_offset=[{:.3f}, {:.3f}, {:.3f}], reference_offset=[{:.3f}, {:.3f}, {:.3f}], target=[{:.3f}, {:.3f}, {:.3f}], weight={})",
                self.sourceJoint,
                self.referenceJoint,
                self.sourceOffset.x(),
                self.sourceOffset.y(),
                self.sourceOffset.z(),
                self.referenceOffset.x(),
                self.referenceOffset.y(),
                self.referenceOffset.z(),
                self.target.x(),
                self.target.y(),
                self.target.z(),
                self.weight);
          })
      .def_readonly(
          "source_joint",
          &mm::JointToJointPositionDataT<float>::sourceJoint,
          "The index of the source joint")
      .def_readonly(
          "reference_joint",
          &mm::JointToJointPositionDataT<float>::referenceJoint,
          "The index of the reference joint")
      .def_readonly(
          "source_offset",
          &mm::JointToJointPositionDataT<float>::sourceOffset,
          "The offset from source joint in the local coordinate system of source joint")
      .def_readonly(
          "reference_offset",
          &mm::JointToJointPositionDataT<float>::referenceOffset,
          "The offset from reference joint in the local coordinate system of reference joint")
      .def_readonly(
          "target",
          &mm::JointToJointPositionDataT<float>::target,
          "The target position in the reference joint's coordinate frame")
      .def_readonly(
          "weight", &mm::JointToJointPositionDataT<float>::weight, "The weight of the constraint")
      .def_readonly(
          "name", &mm::JointToJointPositionDataT<float>::name, "The name of the constraint");

  py::class_<
      mm::JointToJointPositionErrorFunctionT<float>,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::JointToJointPositionErrorFunctionT<float>>>(
      m,
      "JointToJointPositionErrorFunction",
      R"(Error function that penalizes deviation from a target position expressed in a reference joint's coordinate frame.

This is useful for constraints where you want to control the relative position
between two body parts, such as keeping a hand at a specific position relative
to the body, or maintaining a specific relationship between two joints that
should move together.

The error is computed as:
  error = T_ref^{-1} * (T_src * src_offset) - target

where T_ref and T_src are the global transformations of the reference and
source joints respectively.)")
      .def(
          "__repr__",
          [](const mm::JointToJointPositionErrorFunctionT<float>& self) {
            return fmt::format(
                "JointToJointPositionErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.numConstraints());
          })
      .def(
          py::init<>(
              [](const mm::Character& character,
                 float weight) -> std::shared_ptr<mm::JointToJointPositionErrorFunctionT<float>> {
                validateWeight(weight, "weight");
                auto result =
                    std::make_shared<mm::JointToJointPositionErrorFunctionT<float>>(character);
                result->setWeight(weight);
                return result;
              }),
          R"(Initialize a JointToJointPositionErrorFunction.

            :param character: The character to use.
            :param weight: The weight applied to the error function.)",
          py::keep_alive<1, 2>(),
          py::arg("character"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::JointToJointPositionErrorFunctionT<float>& self,
             int sourceJoint,
             const Eigen::Vector3f& sourceOffset,
             int referenceJoint,
             const Eigen::Vector3f& referenceOffset,
             const Eigen::Vector3f& target,
             float weight,
             const std::string& name) {
            validateJointIndex(sourceJoint, "source_joint", self.getSkeleton());
            validateJointIndex(referenceJoint, "reference_joint", self.getSkeleton());
            validateWeight(weight, "weight");
            self.addConstraint(
                sourceJoint, sourceOffset, referenceJoint, referenceOffset, target, weight, name);
          },
          R"(Adds a joint-to-joint position constraint to the error function.

        :param source_joint: The index of the source joint (the joint whose position we want to constrain).
        :param source_offset: The offset from source joint in the local coordinate system of source joint.
        :param reference_joint: The index of the reference joint (the joint whose coordinate frame we use).
        :param reference_offset: The offset from reference joint in the local coordinate system of reference joint.
        :param target: The target position in the reference joint's coordinate frame.
        :param weight: The weight of the constraint.
        :param name: The name of the constraint (for debugging).)",
          py::arg("source_joint"),
          py::arg("source_offset"),
          py::arg("reference_joint"),
          py::arg("reference_offset"),
          py::arg("target"),
          py::arg("weight") = 1.0f,
          py::arg("name") = std::string{})
      .def(
          "add_constraints",
          [](mm::JointToJointPositionErrorFunctionT<float>& self,
             const py::array_t<int>& sourceJoint,
             const py::array_t<float>& sourceOffset,
             const py::array_t<int>& referenceJoint,
             const py::array_t<float>& referenceOffset,
             const py::array_t<float>& target,
             const std::optional<py::array_t<float>>& weight,
             const std::optional<std::vector<std::string>>& name) {
            ArrayShapeValidator validator;
            const int nConsIdx = -1;
            validator.validate(sourceJoint, "source_joint", {nConsIdx}, {"n_cons"});
            validateJointIndex(sourceJoint, "source_joint", self.getSkeleton());
            validator.validate(sourceOffset, "source_offset", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(referenceJoint, "reference_joint", {nConsIdx}, {"n_cons"});
            validateJointIndex(referenceJoint, "reference_joint", self.getSkeleton());
            validator.validate(
                referenceOffset, "reference_offset", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(target, "target", {nConsIdx, 3}, {"n_cons", "xyz"});
            validator.validate(weight, "weight", {nConsIdx}, {"n_cons"});
            validateWeights(weight, "weight");

            if (name.has_value() && name->size() != sourceJoint.shape(0)) {
              throw std::runtime_error(
                  fmt::format(
                      "Invalid names; expected {} names but got {}",
                      sourceJoint.shape(0),
                      name->size()));
            }

            auto sourceJointAcc = sourceJoint.unchecked<1>();
            auto sourceOffsetAcc = sourceOffset.unchecked<2>();
            auto referenceJointAcc = referenceJoint.unchecked<1>();
            auto referenceOffsetAcc = referenceOffset.unchecked<2>();
            auto targetAcc = target.unchecked<2>();
            auto weightAcc =
                weight.has_value() ? std::make_optional(weight->unchecked<1>()) : std::nullopt;

            py::gil_scoped_release release;

            for (py::ssize_t i = 0; i < sourceJoint.shape(0); ++i) {
              self.addConstraint(
                  sourceJointAcc(i),
                  Eigen::Vector3f(
                      sourceOffsetAcc(i, 0), sourceOffsetAcc(i, 1), sourceOffsetAcc(i, 2)),
                  referenceJointAcc(i),
                  Eigen::Vector3f(
                      referenceOffsetAcc(i, 0), referenceOffsetAcc(i, 1), referenceOffsetAcc(i, 2)),
                  Eigen::Vector3f(targetAcc(i, 0), targetAcc(i, 1), targetAcc(i, 2)),
                  weightAcc.has_value() ? (*weightAcc)(i) : 1.0f,
                  name.has_value() ? name->at(i) : std::string{});
            }
          },
          R"(Adds multiple joint-to-joint position constraints to the error function.

        :param source_joint: A numpy array of indices for the source joints.
        :param source_offset: A numpy array of shape (n, 3) for offsets from source joints in local coordinates.
        :param reference_joint: A numpy array of indices for the reference joints.
        :param reference_offset: A numpy array of shape (n, 3) for offsets from reference joints in local coordinates.
        :param target: A numpy array of shape (n, 3) for target positions in reference joint coordinate frames.
        :param weight: A numpy array of weights for the constraints.
        :param name: An optional list of names for the constraints (for debugging).)",
          py::arg("source_joint"),
          py::arg("source_offset"),
          py::arg("reference_joint"),
          py::arg("reference_offset"),
          py::arg("target"),
          py::arg("weight") = std::nullopt,
          py::arg("name") = std::nullopt)
      .def(
          "clear_constraints",
          &mm::JointToJointPositionErrorFunctionT<float>::clearConstraints,
          "Clears all joint-to-joint position constraints from the error function.")
      .def_property_readonly(
          "constraints",
          [](const mm::JointToJointPositionErrorFunctionT<float>& self) {
            return self.getConstraints();
          },
          "Returns the list of joint-to-joint position constraints.")
      .def(
          "num_constraints",
          &mm::JointToJointPositionErrorFunctionT<float>::numConstraints,
          "Returns the number of constraints.");
}

} // namespace

void addDistanceErrorFunctions(py::module_& m) {
  addNormalErrorFunction(m);
  addDistanceErrorFunction(m);
  addPlaneErrorFunction(m);
  addVertexVertexDistanceErrorFunction(m);
  addJointToJointDistanceErrorFunction(m);
  addJointToJointPositionErrorFunction(m);
}

} // namespace pymomentum
