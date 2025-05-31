# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

def fix_path(prefix, list_with_prefix):
    return [elem.replace(prefix, "") for elem in list_with_prefix]

common_public_headers = [
    "common/aligned.h",
    "common/checks.h",
    "common/exception.h",
    "common/filesystem.h",
    "common/fwd.h",
    "common/log_channel.h",
    "common/log.h",
    "common/memory.h",
    "common/profile.h",
    "common/progress_bar.h",
    "common/string.h",
]

common_sources = [
    "common/log.cpp",
    "common/progress_bar.cpp",
    "common/string.cpp",
]

common_test_sources = [
    "test/common/aligned_allocator_test.cpp",
    "test/common/exception_test.cpp",
    "test/common/log_test.cpp",
    "test/common/progress_bar_test.cpp",
    "test/common/string_test.cpp",
]

simd_public_headers = [
    "simd/simd.h",
]

simd_test_sources = [
    "test/simd/simd_test.cpp",
]

fmt_eigen_public_headers = [
    "math/fmt_eigen.h",
]

online_qr_public_headers = [
    "math/online_householder_qr.h",
]

online_qr_sources = [
    "math/online_householder_qr.cpp",
]

online_qr_test_sources = [
    "test/math/online_qr_test.cpp",
]

math_public_headers = [
    "math/constants.h",
    "math/covariance_matrix.h",
    "math/fwd.h",
    "math/generalized_loss.h",
    "math/intersection.h",
    "math/mesh.h",
    "math/mppca.h",
    "math/random-inl.h",
    "math/random.h",
    "math/transform.h",
    "math/types.h",
    "math/utility.h",
]

math_sources = [
    "math/covariance_matrix.cpp",
    "math/generalized_loss.cpp",
    "math/intersection.cpp",
    "math/mesh.cpp",
    "math/mppca.cpp",
    "math/transform.cpp",
    "math/utility.cpp",
]

math_test_sources = [
    "test/math/covariance_matrix_test.cpp",
    "test/math/generalized_loss_test.cpp",
    "test/math/intersection_test.cpp",
    "test/math/mesh_test.cpp",
    "test/math/mppca_test.cpp",
    "test/math/random_test.cpp",
    "test/math/transform_test.cpp",
    "test/math/utility_test.cpp",
]

simd_generalized_loss_public_headers = [
    "math/simd_generalized_loss.h",
]

simd_generalized_loss_sources = [
    "math/simd_generalized_loss.cpp",
]

simd_generalized_loss_test_sources = [
    "test/math/simd_generalized_loss_test.cpp",
]

solver_public_headers = [
    "solver/fwd.h",
    "solver/gauss_newton_solver.h",
    "solver/solver_function.h",
    "solver/solver.h",
    "solver/subset_gauss_newton_solver.h",
    "solver/gradient_descent_solver.h",
]

solver_sources = [
    "solver/gauss_newton_solver.cpp",
    "solver/solver_function.cpp",
    "solver/solver.cpp",
    "solver/subset_gauss_newton_solver.cpp",
    "solver/gradient_descent_solver.cpp",
]

skeleton_public_headers = [
    "character/fwd.h",
    "character/joint_state.h",
    "character/joint.h",
    "character/parameter_limits.h",
    "character/parameter_transform.h",
    "character/skeleton_state.h",
    "character/skeleton.h",
    "character/skeleton_utility.h",
    "character/types.h",
]

skeleton_sources = [
    "character/joint_state.cpp",
    "character/parameter_limits.cpp",
    "character/parameter_transform.cpp",
    "character/skeleton_state.cpp",
    "character/skeleton.cpp",
    "character/skeleton_utility.cpp",
]

character_public_headers = [
    "character/blend_shape_base.h",
    "character/blend_shape_skinning.h",
    "character/blend_shape.h",
    "character/character_state.h",
    "character/character_utility.h",
    "character/character.h",
    "character/collision_geometry_state.h",
    "character/collision_geometry.h",
    "character/inverse_parameter_transform.h",
    "character/linear_skinning.h",
    "character/locator_state.h",
    "character/locator.h",
    "character/marker.h",
    "character/pose_shape.h",
    "character/skin_weights.h",
]

character_sources = [
    "character/blend_shape_base.cpp",
    "character/blend_shape_skinning.cpp",
    "character/blend_shape.cpp",
    "character/character_state.cpp",
    "character/character_utility.cpp",
    "character/character.cpp",
    "character/collision_geometry_state.cpp",
    "character/inverse_parameter_transform.cpp",
    "character/linear_skinning.cpp",
    "character/locator_state.cpp",
    "character/pose_shape.cpp",
    "character/skin_weights.cpp",
]

character_test_sources = [
    "test/character/blend_shape_test.cpp",
    "test/character/character_test.cpp",
    "test/character/forward_kinematics_test.cpp",
    "test/character/joint_test.cpp",
    "test/character/parameter_limits_test.cpp",
    "test/character/parameter_transform_test.cpp",
    "test/character/simplify_test.cpp",
    "test/character/skeleton_bake_test.cpp",
    "test/character/skeleton_state_test.cpp",
    "test/character/skeleton_test.cpp",
]

character_solver_public_headers = [
    "character_solver/aim_error_function.h",
    "character_solver/collision_error_function.h",
    "character_solver/collision_error_function_stateless.h",
    "character_solver/constraint_error_function-inl.h",
    "character_solver/constraint_error_function.h",
    "character_solver/distance_error_function.h",
    "character_solver/fixed_axis_error_function.h",
    "character_solver/fwd.h",
    "character_solver/gauss_newton_solver_qr.h",
    "character_solver/limit_error_function.h",
    "character_solver/model_parameters_error_function.h",
    "character_solver/normal_error_function.h",
    "character_solver/orientation_error_function.h",
    "character_solver/plane_error_function.h",
    "character_solver/pose_prior_error_function.h",
    "character_solver/position_error_function.h",
    "character_solver/projection_error_function.h",
    "character_solver/skeleton_error_function.h",
    "character_solver/skeleton_solver_function.h",
    "character_solver/skinning_weight_iterator.h",
    "character_solver/state_error_function.h",
    "character_solver/transform_pose.h",
    "character_solver/trust_region_qr.h",
    "character_solver/vertex_error_function.h",
    "character_solver/vertex_projection_error_function.h",
]

character_solver_sources = [
    "character_solver/aim_error_function.cpp",
    "character_solver/collision_error_function.cpp",
    "character_solver/collision_error_function_stateless.cpp",
    "character_solver/distance_error_function.cpp",
    "character_solver/fixed_axis_error_function.cpp",
    "character_solver/gauss_newton_solver_qr.cpp",
    "character_solver/limit_error_function.cpp",
    "character_solver/model_parameters_error_function.cpp",
    "character_solver/normal_error_function.cpp",
    "character_solver/orientation_error_function.cpp",
    "character_solver/plane_error_function.cpp",
    "character_solver/pose_prior_error_function.cpp",
    "character_solver/position_error_function.cpp",
    "character_solver/projection_error_function.cpp",
    "character_solver/skeleton_solver_function.cpp",
    "character_solver/skinning_weight_iterator.cpp",
    "character_solver/state_error_function.cpp",
    "character_solver/transform_pose.cpp",
    "character_solver/trust_region_qr.cpp",
    "character_solver/vertex_error_function.cpp",
    "character_solver/vertex_projection_error_function.cpp",
]

character_solver_test_sources = [
    "test/character_solver/blend_shape_test.cpp",
    "test/character_solver/error_functions_test.cpp",
    "test/character_solver/inverse_kinematics_test.cpp",
    "test/character_solver/solver_test.cpp",
]

simd_constraints_public_headers = [
    "character_solver/simd_collision_error_function.h",
    "character_solver/simd_normal_error_function.h",
    "character_solver/simd_plane_error_function.h",
    "character_solver/simd_position_error_function.h",
]

simd_constraints_sources = [
    "character_solver/simd_collision_error_function.cpp",
    "character_solver/simd_normal_error_function.cpp",
    "character_solver/simd_plane_error_function.cpp",
    "character_solver/simd_position_error_function.cpp",
]

simd_constraints_test_sources = [
    "test/character_solver/simd_functions_test.cpp",
]

character_sequence_solver_public_headers = [
    "character_sequence_solver/fwd.h",
    "character_sequence_solver/model_parameters_sequence_error_function.h",
    "character_sequence_solver/multipose_solver_function.h",
    "character_sequence_solver/multipose_solver.h",
    "character_sequence_solver/sequence_error_function.h",
    "character_sequence_solver/sequence_solver_function.h",
    "character_sequence_solver/sequence_solver.h",
    "character_sequence_solver/state_sequence_error_function.h",
]

character_sequence_solver_sources = [
    "character_sequence_solver/model_parameters_sequence_error_function.cpp",
    "character_sequence_solver/multipose_solver_function.cpp",
    "character_sequence_solver/multipose_solver.cpp",
    "character_sequence_solver/sequence_solver_function.cpp",
    "character_sequence_solver/sequence_solver.cpp",
    "character_sequence_solver/state_sequence_error_function.cpp",
]

character_sequence_solver_test_sources = [
    "test/character_sequence_solver/sequence_test.cpp",
    "test/character_sequence_solver/solver_test.cpp",
]

diff_ik_public_headers = [
    "diff_ik/ceres_utility.h",
    "diff_ik/fully_differentiable_body_ik.h",
    "diff_ik/fully_differentiable_distance_error_function.h",
    "diff_ik/fully_differentiable_motion_error_function.h",
    "diff_ik/fully_differentiable_orientation_error_function.h",
    "diff_ik/fully_differentiable_pose_prior_error_function.h",
    "diff_ik/fully_differentiable_position_error_function.h",
    "diff_ik/fully_differentiable_projection_error_function.h",
    "diff_ik/fully_differentiable_skeleton_error_function.h",
    "diff_ik/fully_differentiable_state_error_function.h",
    "diff_ik/fwd.h",
    "diff_ik/union_error_function.h",
]

diff_ik_sources = [
    "diff_ik/fully_differentiable_body_ik.cpp",
    "diff_ik/fully_differentiable_distance_error_function.cpp",
    "diff_ik/fully_differentiable_motion_error_function.cpp",
    "diff_ik/fully_differentiable_orientation_error_function.cpp",
    "diff_ik/fully_differentiable_pose_prior_error_function.cpp",
    "diff_ik/fully_differentiable_position_error_function.cpp",
    "diff_ik/fully_differentiable_projection_error_function.cpp",
    "diff_ik/fully_differentiable_skeleton_error_function.cpp",
    "diff_ik/fully_differentiable_state_error_function.cpp",
    "diff_ik/union_error_function.cpp",
]

diff_ik_test_headers = [
    "test/diff_ik/test_util.h",
]

diff_ik_test_sources = [
    "test/diff_ik/test_differentiable_ik.cpp",
    "test/diff_ik/test_error_functions.cpp",
    "test/diff_ik/test_util.cpp",
]

io_common_public_headers = [
    "io/common/gsl_utils.h",
    "io/common/stream_utils.h",
]

io_common_sources = [
    "io/common/stream_utils.cpp",
]

io_common_test_sources = [
    "test/io/common/stream_utils_test.cpp",
]

io_skeleton_public_headers = [
    "io/skeleton/locator_io.h",
    "io/skeleton/mppca_io.h",
    "io/skeleton/parameter_limits_io.h",
    "io/skeleton/parameter_transform_io.h",
    "io/skeleton/parameters_io.h",
]

io_skeleton_sources = [
    "io/skeleton/locator_io.cpp",
    "io/skeleton/mppca_io.cpp",
    "io/skeleton/parameter_limits_io.cpp",
    "io/skeleton/parameter_transform_io.cpp",
    "io/skeleton/parameters_io.cpp",
]

io_skeleton_test_sources = [
    "test/io/io_parameter_limits_test.cpp",
]

io_shape_public_headers = [
    "io/shape/blend_shape_io.h",
    "io/shape/pose_shape_io.h",
]

io_shape_sources = [
    "io/shape/blend_shape_io.cpp",
    "io/shape/pose_shape_io.cpp",
]

io_fbx_public_headers = [
    "io/fbx/fbx_io.h",
]

io_fbx_private_headers = [
    "io/fbx/fbx_memory_stream.h",
    "io/fbx/openfbx_loader.h",
    "io/fbx/polygon_data.h",
]

io_fbx_sources = [
    "io/fbx/fbx_io.cpp",
    "io/fbx/fbx_memory_stream.cpp",
    "io/fbx/openfbx_loader.cpp",
    "io/fbx/polygon_data.cpp",
]

io_fbx_sources_unsupported = [
    "io/fbx/fbx_io_unsupported.cpp",
    "io/fbx/openfbx_loader.cpp",
    "io/fbx/polygon_data.cpp",
]

io_fbx_test_sources = [
    "test/io/io_fbx_test.cpp",
]

io_gltf_public_headers = [
    "io/gltf/gltf_builder.h",
    "io/gltf/gltf_file_format.h",
    "io/gltf/gltf_io.h",
]

io_gltf_private_headers = [
    "io/gltf/utils/json_utils.h",
    "io/gltf/utils/accessor_utils.h",
    "io/gltf/utils/coordinate_utils.h",
]

io_gltf_sources = [
    "io/gltf/utils/json_utils.cpp",
    "io/gltf/gltf_builder.cpp",
    "io/gltf/gltf_io.cpp",
]

io_gltf_test_sources = [
    "test/io/io_gltf_test.cpp",
]

io_urdf_public_headers = [
    "io/urdf/urdf_io.h",
]

io_urdf_sources = [
    "io/urdf/urdf_io.cpp",
]

io_urdf_test_sources = [
    "test/io/io_urdf_test.cpp",
]

io_motion_public_headers = [
    "io/motion/joint_params_binary_io.h",
    "io/motion/mmo_io.h",
]

io_motion_sources = [
    "io/motion/joint_params_binary_io.cpp",
    "io/motion/mmo_io.cpp",
]

io_marker_public_headers = [
    "io/marker/c3d_io.h",
    "io/marker/conversions.h",
    "io/marker/coordinate_system.h",
    "io/marker/marker_io.h",
    "io/marker/trc_io.h",
]

io_marker_sources = [
    "io/marker/c3d_io.cpp",
    "io/marker/conversions.cpp",
    "io/marker/marker_io.cpp",
    "io/marker/trc_io.cpp",
]

io_marker_test_sources = [
    "test/io/io_marker_test.cpp",
]

io_public_headers = [
    "io/character_io.h",
]

io_sources = [
    "io/character_io.cpp",
]

test_helpers_public_headers = [
    "test/helpers/expect_throw.h",
    "test/helpers/unique_temporary_directory.h",
    "test/helpers/unique_temporary_file.h",
]

test_helpers_sources = [
    "test/helpers/unique_temporary_directory.cpp",
    "test/helpers/unique_temporary_file.cpp",
]

character_test_helpers_public_headers = [
    "test/character/character_helpers.h",
]

character_test_helpers_sources = [
    "test/character/character_helpers.cpp",
]

character_test_helpers_gtest_public_headers = [
    "test/character/character_helpers_gtest.h",
]

character_test_helpers_gtest_sources = [
    "test/character/character_helpers_gtest.cpp",
]

solver_test_helper_public_headers = [
    "test/solver/solver_test_helpers.h",
]

error_function_helper_public_headers = [
    "test/character_solver/error_function_helpers.h",
]

error_function_helper_sources = [
    "test/character_solver/error_function_helpers.cpp",
]

io_test_helper_public_headers = [
    "test/io/io_helpers.h",
]

io_test_helper_sources_android = [
    "test/io/io_helpers_linux.cpp",
    "test/io/io_helpers.cpp",
]

io_test_helper_sources_iphoneos = [
    "test/io/io_helpers_iphoneos.cpp",
    "test/io/io_helpers.cpp",
]

io_test_helper_sources_linux = io_test_helper_sources_android

io_test_helper_sources_macos = [
    "test/io/io_helpers_macos.mm",
    "test/io/io_helpers.cpp",
]

io_test_helper_sources_windows = [
    "test/io/io_helpers_win32.cpp",
    "test/io/io_helpers.cpp",
]

marker_tracker_public_headers = [
    "marker_tracking/tracker_utils.h",
    "marker_tracking/marker_tracker.h",
]

marker_tracker_sources = [
    "marker_tracking/marker_tracker.cpp",
    "marker_tracking/tracker_utils.cpp",
]

app_utils_public_headers = [
    "marker_tracking/app_utils.h",
]

app_utils_sources = [
    "marker_tracking/app_utils.cpp",
]

process_markers_public_headers = [
    "marker_tracking/process_markers.h",
]

process_markers_sources = [
    "marker_tracking/process_markers.cpp",
]

rerun_eigen_adapters_public_headers = [
    "gui/rerun/eigen_adapters.h",
]

rerun_public_headers = [
    "gui/rerun/logger.h",
    "gui/rerun/logging_redirect.h",
]

rerun_sources = [
    "gui/rerun/logger.cpp",
    "gui/rerun/logging_redirect.cpp",
]

#==========
# Examples
#==========

hello_world_sources = [
    "examples/hello_world/main.cpp",
]

convert_model_sources = [
    "examples/convert_model/convert_model.cpp",
]

glb_viewer_sources = [
    "examples/glb_viewer/glb_viewer.cpp",
]

fbx_viewer_sources = [
    "examples/fbx_viewer/fbx_viewer.cpp",
]

urdf_viewer_sources = [
    "examples/urdf_viewer/urdf_viewer.cpp",
]

c3d_viewer_sources = [
    "examples/c3d_viewer/c3d_viewer.cpp",
]

animate_shapes_sources = [
    "examples/animate_shapes/animate_shapes.cpp",
]

process_markers_app_sources = [
    "examples/process_markers_app/process_markers_app.cpp",
]

refine_motion_sources = [
    "examples/refine_motion/refine_motion.cpp",
]

#===========
# Tutorials
#===========

forward_kinematics_sources = [
    "tutorials/forward_kinematics/forward_kinematics.cpp",
]
