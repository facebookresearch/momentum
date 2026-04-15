/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pymomentum/solver2/solver2_error_functions.h>

#include <momentum/character/character.h>
#include <momentum/character/sdf_collision_geometry.h>
#include <momentum/character_solver/sdf_collision_error_function.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/vertex_sdf_error_function.h>
#include <pymomentum/solver2/solver2_utility.h>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

void addSDFErrorFunctions(pybind11::module_& m) {
  // SDF collision error function
  py::class_<
      mm::SDFCollisionErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::SDFCollisionErrorFunction>>(m, "SDFCollisionErrorFunction")
      .def(
          "__repr__",
          [](const mm::SDFCollisionErrorFunction& self) {
            return fmt::format(
                "SDFCollisionErrorFunction(weight={}, num_active_vertices={}, num_collision_pairs={})",
                self.getWeight(),
                self.getNumActiveVertices(),
                self.getNumCollisionPairs());
          })
      .def(
          py::init<>([](const mm::Character& character,
                        std::vector<mm::SDFColliderT<float>> sdfColliders,
                        const std::vector<int>& participatingVertices,
                        const std::vector<float>& vertexWeights,
                        bool filterRestPoseIntersections,
                        uint8_t maxCollisionsPerVertex,
                        float weight) {
            validateWeight(weight, "weight");
            auto result = std::make_shared<mm::SDFCollisionErrorFunction>(
                character,
                mm::SDFCollisionGeometry(std::move(sdfColliders)),
                participatingVertices,
                vertexWeights,
                filterRestPoseIntersections,
                maxCollisionsPerVertex);
            result->setWeight(weight);
            return result;
          }),
          R"(A skeleton error function that penalizes mesh vertex penetration into SDF colliders.

Tests source mesh vertices against a collection of SDF collision objects and applies
penalties for interpenetration. Uses broad-phase AABB culling followed by narrow-phase
SDF sampling for efficient collision detection.

Collision pairs between bones and SDF colliders that intersect in rest pose can optionally be
filtered out to avoid penalizing expected overlaps (e.g. adjacent body parts).

:param character: The character containing skeleton, mesh, and skin weights.
:param sdf_colliders: List of :class:`pymomentum.geometry.SDFCollider` objects to test against.
:param participating_vertices: Vertex indices to include in collision testing (empty = all vertices).
:param vertex_weights: Per-vertex collision weights (empty = uniform weight of 1.0).
:param filter_rest_pose_intersections: If True, bone-collider pairs that intersect in rest pose are excluded.
:param max_collisions_per_vertex: Maximum number of collisions to report per vertex (default: 1).
:param weight: The weight applied to the error function.)",
          py::arg("character"),
          py::arg("sdf_colliders"),
          py::arg("participating_vertices") = std::vector<int>{},
          py::arg("vertex_weights") = std::vector<float>{},
          py::kw_only(),
          py::arg("filter_rest_pose_intersections") = true,
          py::arg("max_collisions_per_vertex") = static_cast<uint8_t>(1),
          py::arg("weight") = 1.0f)
      .def_property_readonly(
          "num_active_vertices",
          &mm::SDFCollisionErrorFunction::getNumActiveVertices,
          R"(The number of active vertices participating in collision testing.)")
      .def_property_readonly(
          "num_collision_pairs",
          &mm::SDFCollisionErrorFunction::getNumCollisionPairs,
          R"(The number of bone-collider pairs being tested.)");

  // Vertex SDF error function
  py::class_<
      mm::VertexSDFErrorFunction,
      mm::SkeletonErrorFunction,
      std::shared_ptr<mm::VertexSDFErrorFunction>>(m, "VertexSDFErrorFunction")
      .def(
          "__repr__",
          [](const mm::VertexSDFErrorFunction& self) {
            return fmt::format(
                "VertexSDFErrorFunction(weight={}, num_constraints={})",
                self.getWeight(),
                self.getNumConstraints());
          })
      .def(
          py::init<>([](const mm::Character& character,
                        const mm::SDFColliderT<float>& sdfCollider,
                        float weight) {
            validateWeight(weight, "weight");
            auto result = std::make_shared<mm::VertexSDFErrorFunction>(character, sdfCollider);
            result->setWeight(weight);
            return result;
          }),
          R"(A skeleton error function that penalizes deviation of mesh vertex SDF distance from a target value.

Unlike CollisionErrorFunction (which penalizes only penetration), this targets an arbitrary signed
distance value against a single SDF, enabling use cases like fitting a mesh to a target isosurface.

:param character: The character to use.
:param sdf_collider: The SDF collider to test against (with optional parent joint and offset).
:param weight: The weight applied to the error function.)",
          py::arg("character"),
          py::arg("sdf_collider"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "add_constraint",
          [](mm::VertexSDFErrorFunction& self,
             int vertexIndex,
             float targetDistance,
             float weight) {
            validateWeight(weight, "weight");
            mm::VertexSDFConstraintT<float> constraint;
            constraint.vertexIndex = vertexIndex;
            constraint.targetDistance = targetDistance;
            constraint.weight = weight;
            self.addConstraint(constraint);
          },
          R"(Add a constraint specifying a target signed distance for a vertex.

:param vertex_index: The mesh vertex index.
:param target_distance: The target signed distance value (0 = surface, positive = outside, negative = inside).
:param weight: Per-constraint weight (default: 1.0).)",
          py::arg("vertex_index"),
          py::arg("target_distance"),
          py::kw_only(),
          py::arg("weight") = 1.0f)
      .def(
          "clear_constraints",
          &mm::VertexSDFErrorFunction::clearConstraints,
          R"(Clear all constraints.)")
      .def_property_readonly(
          "num_constraints",
          [](const mm::VertexSDFErrorFunctionT<float>& self) { return self.getNumConstraints(); },
          R"(The number of active constraints.)");
}

} // namespace pymomentum
