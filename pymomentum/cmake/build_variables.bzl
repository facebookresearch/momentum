# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

backend_sources = [
    "backend/__init__.py",
    "backend/skel_state_backend.py",
    "backend/trs_backend.py",
    "backend/utils.py",
]

python_utility_public_headers = [
    "python_utility/eigen_quaternion.h",
    "python_utility/python_utility.h",
]

python_utility_sources = [
    "python_utility/python_utility.cpp",
]

tensor_utility_public_headers = [
    "tensor_utility/autograd_utility.h",
    "tensor_utility/tensor_utility.h",
]

tensor_utility_sources = [
    "tensor_utility/tensor_utility.cpp",
]

array_utility_public_headers = [
    "array_utility/array_utility.h",
    "array_utility/batch_accessor.h",
    "array_utility/geometry_accessors.h",
]

array_utility_sources = [
    "array_utility/array_utility.cpp",
    "array_utility/geometry_accessors.cpp",
]

array_utility_test_sources = [
    "cpp_test/array_utility_test.cpp",
]

tensor_utility_test_sources = [
    "cpp_test/tensor_utility_test.cpp",
]

tensor_momentum_public_headers = [
    "tensor_momentum/tensor_blend_shape.h",
    "tensor_momentum/tensor_joint_parameters_to_positions.h",
    "tensor_momentum/tensor_kd_tree.h",
    "tensor_momentum/tensor_momentum_utility.h",
    "tensor_momentum/tensor_mppca.h",
    "tensor_momentum/tensor_parameter_transform.h",
    "tensor_momentum/tensor_quaternion.h",
    "tensor_momentum/tensor_skeleton_state.h",
    "tensor_momentum/tensor_skinning.h",
    "tensor_momentum/tensor_transforms.h",
]

tensor_momentum_sources = [
    "tensor_momentum/tensor_blend_shape.cpp",
    "tensor_momentum/tensor_joint_parameters_to_positions.cpp",
    "tensor_momentum/tensor_kd_tree.cpp",
    "tensor_momentum/tensor_momentum_utility.cpp",
    "tensor_momentum/tensor_mppca.cpp",
    "tensor_momentum/tensor_parameter_transform.cpp",
    "tensor_momentum/tensor_quaternion.cpp",
    "tensor_momentum/tensor_skeleton_state.cpp",
    "tensor_momentum/tensor_skinning.cpp",
    "tensor_momentum/tensor_transforms.cpp",
]

tensor_ik_public_headers = [
    "tensor_ik/solver_options.h",
    "tensor_ik/tensor_collision_error_function.h",
    "tensor_ik/tensor_diff_pose_prior_error_function.h",
    "tensor_ik/tensor_distance_error_function.h",
    "tensor_ik/tensor_error_function_utility.h",
    "tensor_ik/tensor_error_function.h",
    "tensor_ik/tensor_gradient.h",
    "tensor_ik/tensor_ik_utility.h",
    "tensor_ik/tensor_ik.h",
    "tensor_ik/tensor_limit_error_function.h",
    "tensor_ik/tensor_marker_error_function.h",
    "tensor_ik/tensor_motion_error_function.h",
    "tensor_ik/tensor_pose_prior_error_function.h",
    "tensor_ik/tensor_projection_error_function.h",
    "tensor_ik/tensor_residual.h",
    "tensor_ik/tensor_vertex_error_function.h",
    "tensor_ik/tensor_vertex_projection_error_function.h",
]

tensor_ik_sources = [
    "tensor_ik/tensor_collision_error_function.cpp",
    "tensor_ik/tensor_diff_pose_prior_error_function.cpp",
    "tensor_ik/tensor_distance_error_function.cpp",
    "tensor_ik/tensor_error_function.cpp",
    "tensor_ik/tensor_gradient.cpp",
    "tensor_ik/tensor_ik_utility.cpp",
    "tensor_ik/tensor_ik.cpp",
    "tensor_ik/tensor_limit_error_function.cpp",
    "tensor_ik/tensor_marker_error_function.cpp",
    "tensor_ik/tensor_motion_error_function.cpp",
    "tensor_ik/tensor_pose_prior_error_function.cpp",
    "tensor_ik/tensor_projection_error_function.cpp",
    "tensor_ik/tensor_residual.cpp",
    "tensor_ik/tensor_vertex_error_function.cpp",
    "tensor_ik/tensor_vertex_projection_error_function.cpp",
]

tensor_ik_test_sources = [
    "cpp_test/tensor_ik_test.cpp",
]

geometry_public_headers = [
    "geometry/array_blend_shape.h",
    "geometry/array_joint_parameters_to_positions.h",
    "geometry/array_parameter_transform.h",
    "geometry/array_skeleton_state.h",
    "geometry/array_skinning.h",
    "geometry/array_vertex_normals.h",
    "geometry/character_pybind.h",
    "geometry/gltf_builder_pybind.h",
    "geometry/limit_pybind.h",
    "geometry/locators_pybind.h",
    "geometry/mesh_pybind.h",
    "geometry/momentum_geometry.h",
    "geometry/momentum_io.h",
    "geometry/parameter_transform_pybind.h",
    "geometry/skeleton_pybind.h",
    "geometry/skin_weights_pybind.h",
]

geometry_sources = [
    "geometry/array_blend_shape.cpp",
    "geometry/array_joint_parameters_to_positions.cpp",
    "geometry/array_parameter_transform.cpp",
    "geometry/array_skeleton_state.cpp",
    "geometry/array_skinning.cpp",
    "geometry/array_vertex_normals.cpp",
    "geometry/character_pybind.cpp",
    "geometry/geometry_pybind.cpp",
    "geometry/gltf_builder_pybind.cpp",
    "geometry/limit_pybind.cpp",
    "geometry/locators_pybind.cpp",
    "geometry/mesh_pybind.cpp",
    "geometry/momentum_geometry.cpp",
    "geometry/momentum_io.cpp",
    "geometry/parameter_transform_pybind.cpp",
    "geometry/skeleton_pybind.cpp",
    "geometry/skin_weights_pybind.cpp",
]

diff_geometry_public_headers = [
    "diff_geometry/diff_geometry_pybind.h",
    "diff_geometry/diff_character_pybind.h",
    "diff_geometry/diff_transform_pybind.h",
    "diff_geometry/diff_blendshape_pybind.h",
]

diff_geometry_sources = [
    "diff_geometry/diff_geometry_pybind.cpp",
    "diff_geometry/diff_character_pybind.cpp",
    "diff_geometry/diff_transform_pybind.cpp",
    "diff_geometry/diff_blendshape_pybind.cpp",
]

solver_public_headers = [
    "solver/momentum_ik.h",
]

solver_sources = [
    "solver/momentum_ik.cpp",
    "solver/solver_pybind.cpp",
]

solver2_public_headers = [
    "solver2/solver2_error_functions.h",
    "solver2/solver2_sequence_error_functions.h",
    "solver2/solver2_utility.h",
]

solver2_sources = [
    "solver2/solver2_error_functions.cpp",
    "solver2/solver2_pybind.cpp",
    "solver2/solver2_sequence_error_functions.cpp",
    "solver2/solver2_utility.cpp",
]

quaternion_sources = [
    "quaternion.py",
]

skel_state_sources = [
    "skel_state.py",
]

trs_sources = [
    "trs.py",
]

marker_tracking_public_headers = [
]

marker_tracking_sources = [
    "marker_tracking/marker_tracking_pybind.cpp",
]

marker_tracking_extensions_public_headers = [
]

marker_tracking_extensions_sources = [
    "marker_tracking_extensions/marker_tracking_extensions_pybind.cpp",
]

gpu_character_sources = [
    "torch/character.py",
    "torch/parameter_limits.py",
    "torch/utility.py",
]

character_manager_sources = [
    "character_manager/character_manager_pybind.cpp",
]

renderer_public_headers = [
    "renderer/mesh_processing.h",
    "renderer/momentum_render.h",
    "renderer/software_rasterizer.h",
]

renderer_sources = [
    "renderer/mesh_processing.cpp",
    "renderer/momentum_render.cpp",
    "renderer/renderer_pybind.cpp",
    "renderer/software_rasterizer.cpp",
]

axel_public_headers = [
    "axel/axel_utility.h",
    "axel/tri_bvh_pybind.h",
]

axel_sources = [
    "axel/axel_pybind.cpp",
    "axel/axel_utility.cpp",
    "axel/tri_bvh_pybind.cpp",
]
