/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/skin_weights.h>
#include <momentum/common/exception.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>

#include <fmt/format.h>

namespace py = pybind11;
namespace mm = momentum;

namespace pymomentum {

void registerSkinWeightsBindings(py::class_<mm::SkinWeights>& skinWeightsClass) {
  // =====================================================
  // momentum::SkinWeights
  // - weight
  // - index
  // - from_dense
  // - to_dense
  // =====================================================
  skinWeightsClass
      .def(
          py::init(
              [](const py::array_t<int>& index_array,
                 const py::array_t<float>& weights_array) -> mm::SkinWeights {
                // Validate index array dimensions
                MT_THROW_IF_T(
                    index_array.ndim() != 2,
                    py::value_error,
                    "Index array must be 2-dimensional, got {} dimensions",
                    index_array.ndim());
                MT_THROW_IF_T(
                    weights_array.ndim() != 2,
                    py::value_error,
                    "Weights array must be 2-dimensional, got {} dimensions",
                    index_array.ndim());

                // Validate that both arrays have same number of rows
                MT_THROW_IF_T(
                    index_array.shape(0) != weights_array.shape(0) ||
                        index_array.shape(1) != weights_array.shape(1),
                    py::value_error,
                    "Index and weights arrays must have same shape: index is {}x{}, weights is {}x{}",
                    index_array.shape(0),
                    index_array.shape(1),
                    weights_array.shape(0),
                    weights_array.shape(1));

                MT_THROW_IF_T(
                    index_array.shape(1) > mm::kMaxSkinJoints,
                    py::value_error,
                    "Index array has {} influence joints per vertex, but maximum allowed is {} (kMaxSkinJoints)",
                    index_array.shape(1),
                    mm::kMaxSkinJoints);

                // Validate weights array dimensions
                const auto num_vertices = index_array.shape(0);
                const auto num_influences = index_array.shape(1);

                // Initialize matrices with zeros - these are fixed-size
                // matrices with kMaxSkinJoints columns
                mm::IndexMatrix index_matrix =
                    mm::IndexMatrix::Zero(num_vertices, mm::kMaxSkinJoints);
                mm::WeightMatrix weight_matrix =
                    mm::WeightMatrix::Zero(num_vertices, mm::kMaxSkinJoints);

                // Copy data from numpy arrays to Eigen matrices
                auto index_accessor = index_array.unchecked<2>();
                auto weights_accessor = weights_array.unchecked<2>();

                for (int i = 0; i < num_vertices; ++i) {
                  for (int j = 0; j < num_influences; ++j) {
                    auto index = index_accessor(i, j);
                    if (index < 0) {
                      throw py::value_error(fmt::format(
                          "Index array contains negative index value at row {}, column {}", i, j));
                    }
                    index_matrix(i, j) = static_cast<uint32_t>(index);
                  }
                  for (int j = 0; j < num_influences; ++j) {
                    weight_matrix(i, j) = weights_accessor(i, j);
                  }
                }

                return {std::move(index_matrix), std::move(weight_matrix)};
              }),
          py::arg("index"),
          py::arg("weights"),
          R"(Create SkinWeights from index and weight arrays.

:param index: 2D numpy array of shape (num_vertices, num_influences) containing joint indices.
              Maximum influencing joints per vertex is defined by kMaxSkinJoints constant.
              Values should be non-negative integers representing joint indices.
:param weights: 2D numpy array of shape (num_vertices, num_influences) containing joint weights.
               Maximum influencing joints per vertex is defined by kMaxSkinJoints constant.
               Weights for each vertex typically sum to 1.0.
:return: SkinWeights object with properly formatted index and weight matrices.
:raises ValueError: If arrays have incompatible dimensions or exceed maximum joint limits.)")
      .def(
          "to_dense",
          [](const mm::SkinWeights& skinWeights, int num_joints) -> py::array_t<float> {
            MT_THROW_IF_T(
                num_joints <= 0,
                py::value_error,
                "num_joints must be positive, got {}",
                num_joints);

            const auto num_vertices = skinWeights.weight.rows();

            // Create dense matrix [num_vertices x num_joints]
            py::array_t<float> dense_weights({num_vertices, static_cast<Eigen::Index>(num_joints)});
            auto dense_accessor = dense_weights.mutable_unchecked<2>();

            // Initialize to zeros
            for (Eigen::Index i = 0; i < num_vertices; ++i) {
              for (int j = 0; j < num_joints; ++j) {
                dense_accessor(i, j) = 0.0f;
              }
            }

            // Fill in the weights from sparse representation
            for (Eigen::Index vertex_idx = 0; vertex_idx < num_vertices; ++vertex_idx) {
              for (int influence_idx = 0; influence_idx < mm::kMaxSkinJoints; ++influence_idx) {
                const uint32_t joint_idx = skinWeights.index(vertex_idx, influence_idx);
                const float weight = skinWeights.weight(vertex_idx, influence_idx);

                // Skip if weight is zero or joint index is out of bounds
                if (weight == 0.0f || joint_idx >= static_cast<uint32_t>(num_joints)) {
                  continue;
                }

                dense_accessor(vertex_idx, joint_idx) = weight;
              }
            }

            return dense_weights;
          },
          py::arg("num_joints"),
          R"(Convert sparse skin weights to dense matrix representation.

:param num_joints: Total number of joints in the skeleton.
:return: Dense numpy array of shape (num_vertices, num_joints) where entry [i,j] 
         is the weight of vertex i for joint j. Entries are 0.0 for joints that 
         don't influence a given vertex.)")
      .def_static(
          "from_dense",
          [](const py::array_t<float>& dense_weights,
             float weight_threshold = 1e-6f) -> mm::SkinWeights {
            MT_THROW_IF_T(
                dense_weights.ndim() != 2,
                py::value_error,
                "Dense weights array must be 2-dimensional, got {} dimensions",
                dense_weights.ndim());

            MT_THROW_IF_T(
                weight_threshold < 0.0f,
                py::value_error,
                "Weight threshold must be non-negative, got {}",
                weight_threshold);

            const int num_vertices = static_cast<int>(dense_weights.shape(0));
            const int num_joints = static_cast<int>(dense_weights.shape(1));

            // Initialize sparse matrices
            mm::IndexMatrix index_matrix = mm::IndexMatrix::Zero(num_vertices, mm::kMaxSkinJoints);
            mm::WeightMatrix weight_matrix =
                mm::WeightMatrix::Zero(num_vertices, mm::kMaxSkinJoints);

            auto dense_accessor = dense_weights.unchecked<2>();

            // Convert each vertex's weights from dense to sparse
            for (int vertex_idx = 0; vertex_idx < num_vertices; ++vertex_idx) {
              // Collect all non-zero weights for this vertex
              std::vector<std::pair<int, float>> vertex_weights;
              for (int joint_idx = 0; joint_idx < num_joints; ++joint_idx) {
                const float weight = dense_accessor(vertex_idx, joint_idx);
                if (weight > weight_threshold) {
                  vertex_weights.emplace_back(joint_idx, weight);
                }
              }

              // Sort by weight (descending) to keep the most important influences
              std::sort(
                  vertex_weights.begin(), vertex_weights.end(), [](const auto& a, const auto& b) {
                    return a.second > b.second;
                  });

              // Take at most kMaxSkinJoints influences
              const int num_influences = std::min(
                  static_cast<int>(vertex_weights.size()), static_cast<int>(mm::kMaxSkinJoints));

              MT_THROW_IF_T(
                  num_influences == 0,
                  py::value_error,
                  "Vertex {} has no weights above threshold {}. All vertices must have at least one non-zero weight.",
                  vertex_idx,
                  weight_threshold);

              // Store the top influences in sparse format
              for (int influence_idx = 0; influence_idx < num_influences; ++influence_idx) {
                const auto& [joint_idx, weight] = vertex_weights[influence_idx];
                index_matrix(vertex_idx, influence_idx) = static_cast<uint32_t>(joint_idx);
                weight_matrix(vertex_idx, influence_idx) = weight;
              }
            }

            return {std::move(index_matrix), std::move(weight_matrix)};
          },
          py::arg("dense_weights"),
          py::arg("weight_threshold") = 1e-6f,
          R"(Create SkinWeights from dense matrix representation.

:param dense_weights: Dense numpy array of shape (num_vertices, num_joints) where entry [i,j] 
                     is the weight of vertex i for joint j.
:param weight_threshold: Minimum weight value to consider. Weights below this threshold are ignored.
                        Defaults to 1e-6.
:return: SkinWeights object with sparse representation. Only the top kMaxSkinJoints influences 
         per vertex are retained, sorted by weight in descending order.
:raises ValueError: If any vertex has no weights above the threshold.)")
      .def_property_readonly(
          "weight",
          [](const mm::SkinWeights& skinning) { return skinning.weight; },
          "Returns the skinning weights matrix. Shape: (num_vertices, kMaxSkinJoints)")
      .def_property_readonly(
          "index",
          [](const mm::SkinWeights& skinning) { return skinning.index; },
          "Returns the skinning indices matrix. Shape: (num_vertices, kMaxSkinJoints)")
      .def_property_readonly(
          "num_vertices",
          [](const mm::SkinWeights& skinning) { return skinning.weight.rows(); },
          "Returns the number of vertices.")
      .def_property_readonly(
          "max_influences_per_vertex",
          [](const mm::SkinWeights& /*skinning*/) { return mm::kMaxSkinJoints; },
          "Returns the maximum number of joint influences per vertex (kMaxSkinJoints constant).")
      .def(
          "normalize_weights",
          [](const mm::SkinWeights& skinWeights) -> mm::SkinWeights {
            mm::SkinWeights normalized = skinWeights;
            const auto num_vertices = normalized.weight.rows();

            for (Eigen::Index vertex_idx = 0; vertex_idx < num_vertices; ++vertex_idx) {
              float total_weight = 0.0f;

              // Calculate sum of weights for this vertex
              for (int influence_idx = 0; influence_idx < mm::kMaxSkinJoints; ++influence_idx) {
                total_weight += normalized.weight(vertex_idx, influence_idx);
              }

              // Normalize if total weight is non-zero
              if (total_weight > 1e-8f) {
                for (int influence_idx = 0; influence_idx < mm::kMaxSkinJoints; ++influence_idx) {
                  normalized.weight(vertex_idx, influence_idx) /= total_weight;
                }
              }
            }

            return normalized;
          },
          R"(Return a normalized copy of the skin weights where each vertex's weights sum to 1.0.

:return: New SkinWeights object with normalized weights. Vertices with zero total weight are left unchanged.)")
      .def("__repr__", [](const mm::SkinWeights& sw) {
        return fmt::format(
            "SkinWeights(vertices={}, max_influences={})", sw.weight.rows(), mm::kMaxSkinJoints);
      });
}

} // namespace pymomentum
