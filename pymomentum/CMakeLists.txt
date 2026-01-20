# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#===============================================================================
# Find dependencies
#===============================================================================

if(NOT DEFINED ENV{CONDA_PREFIX})
  message(STATUS
    "CONDA_PREFIX is not set. Building without Conda/Pixi environment."
  )
endif()

# Determine torch base path based on build environment and platform
# Use CONDA_PREFIX for Unix conda-build, use site-packages for PyPI or Windows
if(DEFINED ENV{CONDA_BUILD} AND NOT WIN32)
  # Unix conda-build - use CONDA_PREFIX
  set(libtorch_base_path $ENV{CONDA_PREFIX})
else()
  # Windows or PyPI - use site-packages/torch
  if(NOT DEFINED Python3_SITELIB)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('purelib'))"
      OUTPUT_VARIABLE Python3_SITELIB
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()
  set(libtorch_base_path ${Python3_SITELIB}/torch)
endif()

if(NOT EXISTS "${libtorch_base_path}")
  message(FATAL_ERROR
    "PyTorch not found in the expected location: ${libtorch_base_path}.\n"
    "Please ensure PyTorch is installed in your Conda/Pixi environment."
  )
endif()

# Try both locations for CMake files (conda and PyPI have different structures)
find_package(ATen CONFIG REQUIRED HINTS
  ${libtorch_base_path}/share/cmake/ATen
  ${libtorch_base_path}
)
find_package(Torch CONFIG REQUIRED HINTS
  ${libtorch_base_path}/share/cmake/Torch
  ${libtorch_base_path}
)
find_library(torch_python
  NAMES torch_python
  HINTS ${libtorch_base_path}/lib/
  REQUIRED
)

#===============================================================================
# Find build dependencies
#===============================================================================

# pybind11
if(MOMENTUM_USE_SYSTEM_PYBIND11)
  find_package(pybind11 CONFIG REQUIRED)
else()
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(pybind11)
endif()

#===============================================================================
# Build PyMomentum
#===============================================================================

# TorchBridge INTERFACE library for OSS builds
set(TORCH_BRIDGE_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/torch_bridge_include)
file(MAKE_DIRECTORY ${TORCH_BRIDGE_INCLUDE_DIR}/pymomentum)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/oss/torch_bridge.h.in
  ${TORCH_BRIDGE_INCLUDE_DIR}/pymomentum/torch_bridge.h
  COPYONLY
)

add_library(torch_bridge INTERFACE)
target_include_directories(torch_bridge INTERFACE ${TORCH_BRIDGE_INCLUDE_DIR})

mt_library(
  NAME tensor_utility
  PYMOMENTUM_HEADERS_VARS tensor_utility_public_headers
  PYMOMENTUM_SOURCES_VARS tensor_utility_sources
  PUBLIC_INCLUDE_DIRECTORIES
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
  PUBLIC_LINK_LIBRARIES
    Eigen3::Eigen
    pybind11::pybind11_headers
    Python3::Module
    ${TORCH_LIBRARIES}
  PRIVATE_LINK_LIBRARIES
    momentum
  PUBLIC_COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
  NO_INSTALL
)

mt_library(
  NAME python_utility
  PYMOMENTUM_HEADERS_VARS python_utility_public_headers
  PYMOMENTUM_SOURCES_VARS python_utility_sources
  PUBLIC_LINK_LIBRARIES
    momentum
    nlohmann_json::nlohmann_json
    pybind11::pybind11
    Python3::Module
  NO_INSTALL
)

mt_library(
  NAME tensor_momentum
  PYMOMENTUM_HEADERS_VARS tensor_momentum_public_headers
  PYMOMENTUM_SOURCES_VARS tensor_momentum_sources
  PUBLIC_INCLUDE_DIRECTORIES
    ${ATEN_INCLUDE_DIR}
  PRIVATE_INCLUDE_DIRECTORIES
    ${TORCH_INCLUDE_DIRS}
  PUBLIC_LINK_LIBRARIES
    momentum
    diff_ik
    torch_bridge
    ${ATEN_LIBRARIES}
    pybind11::pybind11
  PRIVATE_LINK_LIBRARIES
    axel
    python_utility
    tensor_utility
    Ceres::ceres
    Dispenso::dispenso
    Eigen3::Eigen
    ${TORCH_LIBRARIES}
    ${torch_python}
  PUBLIC_COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
  NO_INSTALL
)

mt_library(
  NAME tensor_ik
  PYMOMENTUM_HEADERS_VARS tensor_ik_public_headers
  PYMOMENTUM_SOURCES_VARS tensor_ik_sources
  PUBLIC_INCLUDE_DIRECTORIES
    ${ATEN_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
  PUBLIC_LINK_LIBRARIES
    momentum
    diff_ik
    ${ATEN_LIBRARIES}
  PRIVATE_LINK_LIBRARIES
    tensor_utility
    Dispenso::dispenso
    Eigen3::Eigen
    ${TORCH_LIBRARIES}
    ${torch_python}
  PUBLIC_COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
  NO_INSTALL
)

mt_python_binding(
  NAME geometry
  PYMOMENTUM_HEADERS_VARS geometry_public_headers
  PYMOMENTUM_SOURCES_VARS geometry_sources
  INCLUDE_DIRECTORIES
    ${ATEN_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
  LINK_LIBRARIES
    character
    character_test_helpers
    io
    io_fbx
    io_file_save_options
    io_gltf
    io_legacy_json
    io_marker
    io_shape
    io_skeleton
    io_urdf
    $<$<BOOL:${MOMENTUM_BUILD_IO_USD}>:io_usd>
    python_utility
    tensor_momentum
    tensor_utility
    torch_bridge
    ${ATEN_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${torch_python}
  COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    $<$<BOOL:${MOMENTUM_BUILD_IO_USD}>:-DMOMENTUM_WITH_USD>
)

mt_python_binding(
  NAME diff_geometry
  PYMOMENTUM_HEADERS_VARS diff_geometry_public_headers
  PYMOMENTUM_SOURCES_VARS diff_geometry_sources
  INCLUDE_DIRECTORIES
    ${ATEN_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
  LINK_LIBRARIES
    character
    tensor_momentum
    tensor_utility
    torch_bridge
    ${ATEN_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${torch_python}
  COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
)

# Use 'pymomentum_solver' to avoid conflicts with the solver module
mt_python_binding(
  NAME pymomentum_solver
  MODULE_NAME solver
  PYMOMENTUM_HEADERS_VARS solver_public_headers
  PYMOMENTUM_SOURCES_VARS solver_sources
  INCLUDE_DIRECTORIES
    ${ATEN_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
  LINK_LIBRARIES
    character_solver
    math
    skeleton
    python_utility
    tensor_ik
    tensor_momentum
    tensor_utility
    torch_bridge
    ${ATEN_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${torch_python}
  COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
)

mt_python_binding(
  NAME solver2
  PYMOMENTUM_HEADERS_VARS solver2_public_headers
  PYMOMENTUM_SOURCES_VARS solver2_sources
  INCLUDE_DIRECTORIES
    ${ATEN_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
  LINK_LIBRARIES
    character
    character_solver
    character_sequence_solver
    math
    skeleton
    solver
    torch_bridge
    ${ATEN_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${torch_python}
  COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
)

mt_python_binding(
  NAME marker_tracking
  PYMOMENTUM_HEADERS_VARS marker_tracking_public_headers
  PYMOMENTUM_SOURCES_VARS marker_tracking_sources
  LINK_LIBRARIES
    character
    marker_tracker
    math
    process_markers
    torch_bridge
)

mt_python_binding(
  NAME pymomentum_axel
  MODULE_NAME axel
  PYMOMENTUM_HEADERS_VARS axel_public_headers
  PYMOMENTUM_SOURCES_VARS axel_sources
  INCLUDE_DIRECTORIES
    ${ATEN_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
  LINK_LIBRARIES
    axel
    python_utility
    tensor_utility
    ${ATEN_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${torch_python}
  COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
)

if(MOMENTUM_BUILD_RENDERER)
  mt_python_binding(
    NAME renderer
    PYMOMENTUM_HEADERS_VARS renderer_public_headers
    PYMOMENTUM_SOURCES_VARS renderer_sources
    INCLUDE_DIRECTORIES
      ${ATEN_INCLUDE_DIR}
      ${TORCH_INCLUDE_DIRS}
    LINK_LIBRARIES
      character
      math
      rasterizer
      skeleton
      tensor_momentum
      tensor_utility
      torch_bridge
      ${ATEN_LIBRARIES}
      ${TORCH_LIBRARIES}
      ${torch_python}
    COMPILE_OPTIONS
      ${TORCH_CXX_FLAGS}
  )
endif()

mt_python_library(
  NAME quaternion
  PYMOMENTUM_SOURCES_VARS quaternion_sources
)

mt_python_library(
  NAME skel_state
  PYMOMENTUM_SOURCES_VARS skel_state_sources
)

mt_python_library(
  NAME trs
  PYMOMENTUM_SOURCES_VARS trs_sources
)

mt_python_library(
  NAME backend
  PYMOMENTUM_SOURCES_VARS backend_sources
  PRESERVE_DIRECTORY_STRUCTURE
)

mt_python_library(
  NAME torch
  PYMOMENTUM_SOURCES_VARS gpu_character_sources
  PRESERVE_DIRECTORY_STRUCTURE
)

#===============================================================================
# Tests
#===============================================================================

if(MOMENTUM_BUILD_TESTING)
  enable_testing()
  mt_setup_gtest()

  # For SKBUILD (scikit-build-core), the main CMakeLists.txt only requests
  # Development.Module which doesn't provide Python3::Python. We need to
  # find a way to link against the Python library for test executables.
  if(NOT TARGET Python3::Python)
    # Try to find Development.Embed component which provides Python3::Python
    find_package(Python3 QUIET COMPONENTS Development.Embed)

    if(NOT TARGET Python3::Python)
      # If Development.Embed is not available (e.g., in manylinux), try to
      # create the Python3::Python target manually using the library path
      if(Python3_LIBRARY)
        add_library(Python3::Python IMPORTED INTERFACE)
        set_target_properties(Python3::Python PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${Python3_INCLUDE_DIRS}"
          INTERFACE_LINK_LIBRARIES "${Python3_LIBRARY}"
        )
        message(STATUS "Created Python3::Python target manually using ${Python3_LIBRARY}")
      else()
        # Try to find the library manually
        find_library(Python3_LIBRARY
          NAMES python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}
                python${Python3_VERSION_MAJOR}
                python
          HINTS ${Python3_LIBRARY_DIRS}
                ${Python3_RUNTIME_LIBRARY_DIRS}
          PATH_SUFFIXES lib libs
        )
        if(Python3_LIBRARY)
          add_library(Python3::Python IMPORTED INTERFACE)
          set_target_properties(Python3::Python PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Python3_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${Python3_LIBRARY}"
          )
          message(STATUS "Created Python3::Python target manually using ${Python3_LIBRARY}")
        else()
          message(FATAL_ERROR
            "Could not find Python library for linking test executables. "
            "Please ensure Python development files are installed."
          )
        endif()
      endif()
    endif()
  endif()

  mt_test(
    NAME tensor_utility_test
    PYMOMENTUM_SOURCES_VARS tensor_utility_test_sources
    LINK_LIBRARIES
      tensor_utility
      Python3::Python
  )

  mt_test(
    NAME tensor_ik_test
    PYMOMENTUM_SOURCES_VARS tensor_ik_test_sources
    LINK_LIBRARIES
      character_test_helpers
      tensor_ik
      tensor_utility
      Python3::Python
  )
endif()

#===============================================================================
# Install
#===============================================================================

mt_install_pymomentum()
