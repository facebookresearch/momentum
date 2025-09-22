# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

tensor_utility_public_headers = [
    "tensor_utility/autograd_utility.h",
    "tensor_utility/tensor_utility.h",
]

tensor_utility_sources = [
    "tensor_utility/tensor_utility.cpp",
]

tensor_utility_test_sources = [
    "cpp_test/tensor_utility_test.cpp",
]

python_utility_public_headers = [
    "python_utility/python_utility.h",
]

python_utility_sources = [
    "python_utility/python_utility.cpp",
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
    "geometry/momentum_geometry.h",
    "geometry/momentum_io.h",
    "geometry/gltf_builder_pybind.h",
]

geometry_sources = [
    "geometry/geometry_pybind.cpp",
    "geometry/momentum_geometry.cpp",
    "geometry/momentum_io.cpp",
    "geometry/gltf_builder_pybind.cpp",
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
