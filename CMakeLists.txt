# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.16.3)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(momentum)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Needed by mt_defs.cmake
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include(cmake/mt_defs.cmake)

# Print intro
message(STATUS "")
message(STATUS "============================================")
message(STATUS "                Momentum")
message(STATUS "============================================")
message(STATUS "")

# Print OS and architecture information
message(STATUS "[ System Info ]")
message(STATUS "- Operating System    : ${CMAKE_SYSTEM_NAME}")
message(STATUS "- Processor Arch      : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "")

# Print build tool information
message(STATUS "[ Build Tools ]")
message(STATUS "- CMake               : ${CMAKE_VERSION}")
message(STATUS "- CMake Generator     : ${CMAKE_GENERATOR}")
message(STATUS "- C Compiler          : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "- C++ Compiler        : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(MSVC)
  message(STATUS "- CMake Toolchain File: ${CMAKE_TOOLCHAIN_FILE}")
endif()
message(STATUS "")

# Print CMake variables
message(STATUS "[ CMake Variables ]")
mt_print_flags(CMAKE_C_FLAGS)
mt_print_flags(CMAKE_CXX_FLAGS)
mt_print_flags(CMAKE_CUDA_FLAGS)
message(STATUS "")

#===============================================================================
# Build dependencies
#===============================================================================

include(GNUInstallDirs)

#===============================================================================
# Build Options
#===============================================================================

mt_option(BUILD_SHARED_LIBS "Build as shared libraries" ON)
mt_option(MOMENTUM_BUILD_WITH_FBXSDK "Build with IO FBX" OFF)
mt_option(MOMENTUM_BUILD_PYMOMENTUM "Build Python binding" OFF)
mt_option(MOMENTUM_BUILD_TESTING "Enable building tests" OFF)
mt_option(MOMENTUM_BUILD_EXAMPLES "Enable building examples" OFF)
mt_option(MOMENTUM_ENABLE_PROFILING "Enable building with profiling annotations" OFF)
mt_option(MOMENTUM_ENABLE_SIMD "Enable building with SIMD instructions" ON)
mt_option(MOMENTUM_INSTALL_EXAMPLES "Install examples" OFF)
mt_option(MOMENTUM_USE_SYSTEM_GOOGLETEST "Use GoogleTest installed in system" OFF)
mt_option(MOMENTUM_USE_SYSTEM_PYBIND11 "Use pybind11 installed in system" OFF)
mt_option(MOMENTUM_USE_SYSTEM_RERUN_CPP_SDK "Use Rerun C++ SDK installed in system" OFF)
mt_option(MOMENTUM_USE_SYSTEM_TRACY "Use Tracy installed in system" OFF)

if(MOMENTUM_ENABLE_PROFILING AND CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  # Force disable profiling on Mac x86 to avoid initialization failure:
  # "Tracy Profiler initialization failure: CPU doesn't support invariant TSC."
  # Defining TRACY_NO_INVARIANT_CHECK=1 or TRACY_TIMER_FALLBACK is not recommended,
  # as it may lead to inaccurate profiling results.
  message(WARNING "MOMENTUM_ENABLE_PROFILING was set to ON, but Tracy profiling is not supported on Mac x86 due to CPU incompatibility. Disabling profiling.")
  set(MOMENTUM_ENABLE_PROFILING OFF CACHE BOOL "Enable building with profiling annotations" FORCE)
endif()

if(MOMENTUM_INSTALL_EXAMPLES AND NOT MOMENTUM_BUILD_EXAMPLES)
  message(WARNING "MOMENTUM_INSTALL_EXAMPLES is ON, but MOMENTUM_BUILD_EXAMPLES is OFF. No examples will be built or installed.")
endif()

mt_print_options()

#===============================================================================
# Find dependencies
#===============================================================================

# 3rd party
find_package(Ceres CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(Dispenso CONFIG REQUIRED)
find_package(drjit CONFIG REQUIRED)
find_package(Eigen3 3.4.0 CONFIG REQUIRED)
find_package(ezc3d CONFIG REQUIRED)
find_package(Microsoft.GSL CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(fx-gltf CONFIG REQUIRED)
find_package(indicators 2.3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(openfbx CONFIG REQUIRED)
find_package(re2 MODULE REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(urdfdom CONFIG REQUIRED)

if(MOMENTUM_USE_SYSTEM_RERUN_CPP_SDK)
  find_package(rerun_sdk CONFIG REQUIRED)
else()
  include(FetchContent)
  FetchContent_Declare(rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/download/0.22.1/rerun_cpp_sdk.zip
  )
  FetchContent_MakeAvailable(rerun_sdk)
endif()

if(MOMENTUM_ENABLE_PROFILING)
  if(MOMENTUM_USE_SYSTEM_TRACY)
    find_package(Tracy CONFIG REQUIRED)
  else()
    include(FetchContent)
    FetchContent_Declare(tracy
      GIT_REPOSITORY https://github.com/wolfpld/tracy.git
      GIT_TAG v0.11.1
      GIT_SHALLOW TRUE
      GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(tracy)
    if(MSVC)
      target_compile_options(TracyClient PRIVATE /W0)
    else()
      target_compile_options(TracyClient PRIVATE -w)
    endif()
  endif()
endif()

# 1st party
# TODO: Make Axel as a separate project and change to find_package(axel CONFIG REQUIRED)
add_subdirectory(axel)

#===============================================================================
# Compiler Settings
#===============================================================================

if(MOMENTUM_BUILD_TESTING)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(MSVC)
  add_compile_options(/bigobj)
  add_compile_options(/utf-8)
endif()

#===============================================================================
# Build momentum
#===============================================================================

mt_library(
  NAME common
  HEADERS_VARS common_public_headers
  SOURCES_VARS common_sources
  PUBLIC_LINK_LIBRARIES
    fmt::fmt
    indicators::indicators
    spdlog::spdlog
  PUBLIC_COMPILE_DEFINITIONS
    MOMENTUM_WITH_SPDLOG=1
)

mt_library(
  NAME simd
  HEADERS_VARS simd_public_headers
  PUBLIC_LINK_LIBRARIES
    Eigen3::Eigen
    drjit
)

mt_library(
  NAME fmt_eigen
  HEADERS_VARS fmt_eigen_public_headers
  PUBLIC_LINK_LIBRARIES
    fmt::fmt
    Eigen3::Eigen
)

mt_library(
  NAME online_qr
  HEADERS_VARS online_qr_public_headers
  SOURCES_VARS online_qr_sources
  PUBLIC_LINK_LIBRARIES
    common
    fmt_eigen
    Eigen3::Eigen
    Microsoft.GSL::GSL
)

mt_library(
  NAME math
  HEADERS_VARS math_public_headers
  SOURCES_VARS math_sources
  PUBLIC_LINK_LIBRARIES
    common
    fmt_eigen
    online_qr
    Eigen3::Eigen
    Microsoft.GSL::GSL
  PRIVATE_LINK_LIBRARIES
    axel
)

mt_library(
  NAME simd_generalized_loss
  HEADERS_VARS simd_generalized_loss_public_headers
  SOURCES_VARS simd_generalized_loss_sources
  PUBLIC_LINK_LIBRARIES
    math
    simd
)

mt_library(
  NAME skeleton
  HEADERS_VARS skeleton_public_headers
  SOURCES_VARS skeleton_sources
  PUBLIC_LINK_LIBRARIES
    common
    math
    Eigen3::Eigen
    Microsoft.GSL::GSL
)

mt_library(
  NAME character
  HEADERS_VARS character_public_headers
  SOURCES_VARS character_sources
  PUBLIC_LINK_LIBRARIES
    common
    math
    skeleton
    axel
    Dispenso::dispenso
    Eigen3::Eigen
    Microsoft.GSL::GSL
)

mt_library(
  NAME character_test_helpers
  HEADERS_VARS character_test_helpers_public_headers
  SOURCES_VARS character_test_helpers_sources
  PUBLIC_LINK_LIBRARIES
    character
  NO_INSTALL
)

mt_library(
  NAME solver
  HEADERS_VARS solver_public_headers
  SOURCES_VARS solver_sources
  PUBLIC_LINK_LIBRARIES
    common
    math
    Eigen3::Eigen
    Microsoft.GSL::GSL
)

mt_library(
  NAME character_solver
  HEADERS_VARS character_solver_public_headers
  SOURCES_VARS character_solver_sources
  PUBLIC_LINK_LIBRARIES
    character
    solver
    axel
    Eigen3::Eigen
    Microsoft.GSL::GSL
  PRIVATE_LINK_LIBRARIES
    Dispenso::dispenso
)

mt_library(
  NAME simd_constraints
  HEADERS_VARS simd_constraints_public_headers
  SOURCES_VARS simd_constraints_sources
  PUBLIC_LINK_LIBRARIES
    character_solver
    simd
)

mt_library(
  NAME character_sequence_solver
  HEADERS_VARS character_sequence_solver_public_headers
  SOURCES_VARS character_sequence_solver_sources
  PUBLIC_LINK_LIBRARIES
    character_solver
    Eigen3::Eigen
    Microsoft.GSL::GSL
  PRIVATE_LINK_LIBRARIES
    Dispenso::dispenso
)

mt_library(
  NAME momentum
  PUBLIC_LINK_LIBRARIES
    common
    simd
    online_qr
    math
    solver
    skeleton
    character
    character_solver
    simd_constraints
    character_sequence_solver
)

mt_library(
  NAME diff_ik
  HEADERS_VARS diff_ik_public_headers
  SOURCES_VARS diff_ik_sources
  PUBLIC_LINK_LIBRARIES
    character_solver
    Ceres::ceres
    Dispenso::dispenso
)

mt_library(
  NAME io_common
  HEADERS_VARS io_common_public_headers
  SOURCES_VARS io_common_sources
  PUBLIC_LINK_LIBRARIES
    common
    Microsoft.GSL::GSL
)

mt_library(
  NAME io_skeleton
  HEADERS_VARS io_skeleton_public_headers
  SOURCES_VARS io_skeleton_sources
  PUBLIC_LINK_LIBRARIES
    character
  PRIVATE_LINK_LIBRARIES
    io_common
    re2::re2
)

mt_library(
  NAME io_shape
  HEADERS_VARS io_shape_public_headers
  SOURCES_VARS io_shape_sources
  PUBLIC_LINK_LIBRARIES
    character
)

if(MOMENTUM_BUILD_WITH_FBXSDK)
  set(io_fbx_sources_var io_fbx_sources)
  find_package(FbxSdk MODULE REQUIRED)

  # Fix FBX SDK header typos for Clang 19 compatibility
  set(_fbx_header "${FBXSDK_INCLUDE_DIR}/fbxsdk/core/base/fbxredblacktree.h")
  if(EXISTS "${_fbx_header}")
    file(READ "${_fbx_header}" _content)
    string(REPLACE "mLefttChild" "mLeftChild" _content "${_content}")
    string(REPLACE "mRighttChild" "mRightChild" _content "${_content}")
    set(_patched_header "${CMAKE_BINARY_DIR}/fbxsdk_patched/fbxsdk/core/base/fbxredblacktree.h")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/fbxsdk_patched/fbxsdk/core/base")
    file(WRITE "${_patched_header}" "${_content}")
    target_include_directories(fbxsdk::fbxsdk BEFORE INTERFACE "${CMAKE_BINARY_DIR}/fbxsdk_patched")
  endif()

  set(io_fbx_private_link_libraries fbxsdk::fbxsdk)
else()
  set(io_fbx_sources_var io_fbx_sources_unsupported)
  set(io_fbx_private_link_libraries )
endif()

mt_library(
  NAME io_fbx
  HEADERS_VARS
    io_fbx_public_headers
  PRIVATE_HEADERS_VARS
    io_fbx_private_headers
  SOURCES_VARS
    ${io_fbx_sources_var}
  PRIVATE_LINK_LIBRARIES
    io_common
    io_skeleton
    ${io_fbx_private_link_libraries}
    OpenFBX
)

mt_library(
  NAME io_gltf
  HEADERS_VARS
    io_gltf_public_headers
  PRIVATE_HEADERS_VARS
    io_gltf_private_headers
  SOURCES_VARS io_gltf_sources
  PUBLIC_LINK_LIBRARIES
    character
    io_common
    io_skeleton
    fx-gltf::fx-gltf
  PRIVATE_LINK_LIBRARIES
    nlohmann_json::nlohmann_json
)

mt_library(
  NAME io_urdf
  HEADERS_VARS
    io_urdf_public_headers
  SOURCES_VARS io_urdf_sources
  PUBLIC_LINK_LIBRARIES
    character
    io_common
    io_skeleton
    urdfdom::urdf_parser
)

mt_library(
  NAME io_legacy_json
  HEADERS_VARS io_legacy_json_public_headers
  SOURCES_VARS io_legacy_json_sources
  PUBLIC_LINK_LIBRARIES
    character
    io_common
    io_skeleton
    nlohmann_json::nlohmann_json
)

mt_library(
  NAME io_motion
  HEADERS_VARS io_motion_public_headers
  SOURCES_VARS io_motion_sources
  PUBLIC_LINK_LIBRARIES
    character
)

mt_library(
  NAME io_marker
  HEADERS_VARS io_marker_public_headers
  SOURCES_VARS io_marker_sources
  PUBLIC_LINK_LIBRARIES
    character
  PRIVATE_LINK_LIBRARIES
    io_common
    io_gltf
    ezc3d
)

mt_library(
  NAME io
  HEADERS_VARS io_public_headers
  SOURCES_VARS io_sources
  PUBLIC_LINK_LIBRARIES
    io_common
    io_fbx
    io_gltf
    io_legacy_json
    io_motion
    io_marker
    io_shape
    io_skeleton
)

mt_library(
  NAME marker_tracker
  HEADERS_VARS
    marker_tracker_public_headers
  SOURCES_VARS
    marker_tracker_sources
  PUBLIC_LINK_LIBRARIES
    character_solver
  PRIVATE_LINK_LIBRARIES
    character_sequence_solver
)

mt_library(
  NAME app_utils
  HEADERS_VARS
    app_utils_public_headers
  SOURCES_VARS
    app_utils_sources
  PUBLIC_LINK_LIBRARIES
    character
    marker_tracker
    CLI11::CLI11
  PRIVATE_LINK_LIBRARIES
    io
)

mt_library(
  NAME process_markers
  HEADERS_VARS
    process_markers_public_headers
  SOURCES_VARS
    process_markers_sources
  PUBLIC_LINK_LIBRARIES
    app_utils
    marker_tracker
    skeleton
  PRIVATE_LINK_LIBRARIES
    io_marker
)

mt_library(
  NAME rerun_eigen_adapters
  HEADERS_VARS
    rerun_eigen_adapters_public_headers
  PUBLIC_LINK_LIBRARIES
    Eigen3::Eigen
    rerun_sdk
)

mt_library(
  NAME rerun
  HEADERS_VARS
    rerun_public_headers
  SOURCES_VARS
    rerun_sources
  PUBLIC_LINK_LIBRARIES
    character
    rerun_sdk
  PRIVATE_LINK_LIBRARIES
    rerun_eigen_adapters
    axel
)

#===============================================================================
# Tests
#===============================================================================

if(MOMENTUM_BUILD_TESTING)
  find_package(Boost CONFIG REQUIRED)

  enable_testing()
  mt_setup_gtest()

  mt_library(
    NAME test_helpers
    HEADERS_VARS test_helpers_public_headers
    SOURCES_VARS test_helpers_sources
    PUBLIC_LINK_LIBRARIES
      common
      GTest::gtest
    PRIVATE_LINK_LIBRARIES
      Boost::boost
    NO_INSTALL
  )

  mt_library(
    NAME character_test_helpers_gtest
    HEADERS_VARS character_test_helpers_gtest_public_headers
    SOURCES_VARS character_test_helpers_gtest_sources
    PUBLIC_LINK_LIBRARIES
      character
    PRIVATE_LINK_LIBRARIES
      GTest::gmock
      GTest::gtest
    NO_INSTALL
  )

  mt_library(
    NAME solver_test_helper
    HEADERS_VARS solver_test_helper_public_headers
    PUBLIC_LINK_LIBRARIES
      solver
    NO_INSTALL
  )

  mt_library(
    NAME error_function_helper
    HEADERS_VARS error_function_helper_public_headers
    SOURCES_VARS error_function_helper_sources
    PUBLIC_LINK_LIBRARIES
      character
      character_solver
    PRIVATE_LINK_LIBRARIES
      GTest::gmock
      GTest::gtest
    NO_INSTALL
  )

  set(io_test_helper_libs )
  if(APPLE)
    set(io_test_helper_sources io_test_helper_sources_macos)
    list(APPEND io_test_helper_libs "-framework Foundation")
  elseif(UNIX)
    set(io_test_helper_sources io_test_helper_sources_linux)
  elseif(WIN32)
    set(io_test_helper_sources io_test_helper_sources_windows)
  else()
    message(FATAL_ERROR "Unsupported platform")
  endif()

  mt_library(
    NAME io_test_helper
    HEADERS_VARS io_test_helper_public_headers
    SOURCES_VARS ${io_test_helper_sources}
    PUBLIC_LINK_LIBRARIES
      test_helpers
      Microsoft.GSL::GSL
    PRIVATE_LINK_LIBRARIES
      ${io_test_helper_libs}
    NO_INSTALL
  )

  mt_test(
    NAME common_test
    SOURCES_VARS common_test_sources
    LINK_LIBRARIES common
  )

  mt_test(
    NAME simd_test
    SOURCES_VARS simd_test_sources
    LINK_LIBRARIES simd
  )

  mt_test(
    NAME online_qr_test
    SOURCES_VARS online_qr_test_sources
    LINK_LIBRARIES online_qr
  )

  mt_test(
    NAME math_test
    SOURCES_VARS math_test_sources
    LINK_LIBRARIES math
  )

  mt_test(
    NAME simd_generalized_loss_test
    SOURCES_VARS simd_generalized_loss_test_sources
    LINK_LIBRARIES simd_generalized_loss
  )

  # TODO: Add solver_test

  mt_test(
    NAME character_test
    SOURCES_VARS character_test_sources
    LINK_LIBRARIES
      character
      character_test_helpers
      test_helpers
  )

  mt_test(
    NAME character_solver_test
    SOURCES_VARS character_solver_test_sources
    LINK_LIBRARIES
      character_solver
      character_test_helpers
      error_function_helper
      solver_test_helper
  )

  if(NOT APPLE) # TODO: Fix
    mt_test(
      NAME simd_constraints_test
      SOURCES_VARS simd_constraints_test_sources
      LINK_LIBRARIES
        character_solver
        character_test_helpers
        error_function_helper
        simd_constraints
    )
  endif()

  mt_test(
    NAME character_sequence_solver_test
    SOURCES_VARS character_sequence_solver_test_sources
    LINK_LIBRARIES
      character_sequence_solver
      character_test_helpers
      solver_test_helper
  )

  # TODO: Fix test failures on macOS
  if(NOT APPLE)
    mt_test(
      NAME diff_ik_test
      HEADERS_VARS diff_ik_test_headers
      SOURCES_VARS diff_ik_test_sources
      LINK_LIBRARIES
        character_test_helpers
        diff_ik
        io_gltf
        io_skeleton
    )
  endif()

  mt_test(
    NAME io_common_test
    LINK_LIBRARIES io_common
    SOURCES_VARS io_common_test_sources
  )

  # TODO: Add io_gltf_test

  mt_test(
    NAME io_marker_test
    SOURCES_VARS io_marker_test_sources
    LINK_LIBRARIES
      io_marker
      io_test_helper
    ENV
      "TEST_RESOURCES_PATH=${CMAKE_SOURCE_DIR}/momentum/test/resources"
  )

  mt_test(
    NAME io_shape_test
    SOURCES_VARS io_shape_test_sources
    LINK_LIBRARIES
      character
      character_test_helpers
      io_shape
      io_test_helper
  )

  mt_test(
    NAME io_skeleton_test
    SOURCES_VARS io_skeleton_test_sources
    LINK_LIBRARIES
      character_test_helpers
      io_skeleton
      io_test_helper
  )

  mt_test(
    NAME io_motion_test
    SOURCES_VARS io_motion_test_sources
    LINK_LIBRARIES
      character
      character_test_helpers
      io_motion
      io_test_helper
  )

  mt_test(
    NAME io_legacy_json_test
    SOURCES_VARS io_legacy_json_test_sources
    LINK_LIBRARIES
      character_test_helpers
      io_legacy_json
      io_test_helper
    ENV
      "TEST_RESOURCES_PATH=${CMAKE_SOURCE_DIR}/momentum/test/resources"
  )
endif()

#===============================================================================
# Examples
#===============================================================================

if(MOMENTUM_BUILD_EXAMPLES)
  mt_executable(
    NAME hello_world
    SOURCES_VARS hello_world_sources
    LINK_LIBRARIES
      momentum
  )

  mt_executable(
    NAME convert_model
    SOURCES_VARS convert_model_sources
    LINK_LIBRARIES
      momentum io CLI11::CLI11
  )

  mt_executable(
    NAME glb_viewer
    SOURCES_VARS glb_viewer_sources
    LINK_LIBRARIES
      io_gltf
      rerun
      CLI11::CLI11
      rerun_sdk
  )

  mt_executable(
    NAME fbx_viewer
    SOURCES_VARS fbx_viewer_sources
    LINK_LIBRARIES
      io_fbx
      rerun
      CLI11::CLI11
      rerun_sdk
  )

  mt_executable(
    NAME c3d_viewer
    SOURCES_VARS c3d_viewer_sources
    LINK_LIBRARIES
      io_marker
      rerun
      CLI11::CLI11
      rerun_sdk
  )

  mt_executable(
    NAME urdf_viewer
    SOURCES_VARS urdf_viewer_sources
    LINK_LIBRARIES
      io_urdf
      rerun
      CLI11::CLI11
      rerun_sdk
  )

  mt_executable(
    NAME animate_shapes
    SOURCES_VARS animate_shapes_sources
    LINK_LIBRARIES
      momentum io CLI11::CLI11
  )

  mt_executable(
    NAME process_markers_app
    SOURCES_VARS process_markers_app_sources
    LINK_LIBRARIES
      app_utils
      common
      process_markers
  )

  mt_executable(
    NAME refine_motion
    SOURCES_VARS refine_motion_sources
    LINK_LIBRARIES
      app_utils
      io
      marker_tracker
  )
endif()

#===============================================================================
# Install momentum
#===============================================================================

get_property(targets GLOBAL PROPERTY MOMENTUM_TARGETS)
get_property(executables GLOBAL PROPERTY MOMENTUM_EXECUTABLES)

if(targets)
  message(STATUS "[Installing libraries]")
  foreach(target ${targets})
    message(STATUS "  - ${target}")
  endforeach()
else()
  message(STATUS "No libraries to install.")
endif()

if(executables)
  message(STATUS "[Installing executables]")
  foreach(executable ${executables})
    message(STATUS "  - ${executable}")
  endforeach()
else()
  message(STATUS "No executables to install.")
endif()

# Generate required install artifacts for CMake buildsystem
include(CMakePackageConfigHelpers)
set(MOMENTUM_CONFIG_INPUT cmake/momentum-config.cmake.in)
set(MOMENTUM_CONFIG_OUTPUT ${CMAKE_BINARY_DIR}/cmake/momentum-config.cmake)
set(MOMENTUM_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
configure_package_config_file(
  ${MOMENTUM_CONFIG_INPUT}
  ${MOMENTUM_CONFIG_OUTPUT}
  INSTALL_DESTINATION ${MOMENTUM_CONFIG_INSTALL_DIR}
)
install(
  FILES ${MOMENTUM_CONFIG_OUTPUT}
  DESTINATION ${MOMENTUM_CONFIG_INSTALL_DIR}
)

# Install CMake modules
install(
  FILES cmake/Findre2.cmake
  DESTINATION ${MOMENTUM_CONFIG_INSTALL_DIR}
)

# Install headers
install(
  DIRECTORY momentum
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT headers
  FILES_MATCHING PATTERN "*.h"
  PATTERN "*/examples/*" EXCLUDE
  PATTERN "*/test/*" EXCLUDE
  PATTERN "*/website/*" EXCLUDE
)

# Install targets (lib)
install(
  TARGETS ${targets}
  EXPORT momentumTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(
  EXPORT momentumTargets
  NAMESPACE momentum::
  DESTINATION ${MOMENTUM_CONFIG_INSTALL_DIR}
)

# Install executables
install(
  TARGETS ${executables}
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

#===============================================================================
# PyMomentum
#===============================================================================

if(MOMENTUM_BUILD_PYMOMENTUM)
  add_subdirectory(pymomentum)
endif()
