# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.16.3)

project(axel)

include(GNUInstallDirs)

#===============================================================================
# Find dependencies
#===============================================================================

find_package(drjit CONFIG REQUIRED)
find_package(Eigen3 3.4.0 CONFIG REQUIRED)
find_package(Microsoft.GSL CONFIG REQUIRED)
find_package(Dispenso CONFIG REQUIRED)

#===============================================================================
# Build axel
#===============================================================================

set(target_name axel)

add_library(${target_name})

set(headers
  axel/common/Constants.h
  axel/common/Types.h
  axel/common/VectorizationTypes.h
  axel/math/BoundingBoxUtils.h
  axel/math/PointTriangleProjection.h
  axel/math/PointTriangleProjectionDefinitions.h
  axel/math/RayBoxIntersection.h
  axel/math/RayTriangleIntersection.h
  axel/BoundingBox.h
  axel/Bvh.h
  axel/BvhBase.h
  axel/BvhCommon.h
  axel/Ray.h
  axel/SimdKdTree.h
  axel/TriBvh.h
)

set(sources
  axel/math/PointTriangleProjection.cpp
  axel/math/RayBoxIntersection.cpp
  axel/math/RayTriangleIntersection.cpp
  axel/BoundingBox.cpp
  axel/Bvh.cpp
  axel/SimdKdTree.cpp
  axel/TriBvh.cpp
)

target_sources(${target_name}
  PRIVATE
    ${headers}
    ${sources}
)

target_include_directories(${target_name}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/axel>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/axel>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(${target_name} PUBLIC cxx_std_17)

target_link_libraries(${target_name}
  PUBLIC
    Eigen3::Eigen
    Microsoft.GSL::GSL
    Dispenso::dispenso
    drjit
)

if(MSVC)
  set_target_properties(${target_name} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(MOMENTUM_ENABLE_PROFILING)
  target_link_libraries(${target_name} PUBLIC Tracy::TracyClient)
  target_compile_definitions(${target_name} PUBLIC -DAXEL_WITH_TRACY_PROFILER=1)
endif()

#===============================================================================
# Install axel
#===============================================================================

# Generate required install artifacts for CMake buildsystem
include(CMakePackageConfigHelpers)
set(AXEL_CONFIG_INPUT axel-config.cmake.in)
set(AXEL_CONFIG_OUTPUT ${CMAKE_BINARY_DIR}/cmake/axel-config.cmake)
set(AXEL_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
configure_package_config_file(
  ${AXEL_CONFIG_INPUT}
  ${AXEL_CONFIG_OUTPUT}
  INSTALL_DESTINATION ${AXEL_CONFIG_INSTALL_DIR}
)
install(
  FILES ${AXEL_CONFIG_OUTPUT}
  DESTINATION ${AXEL_CONFIG_INSTALL_DIR}
)

# Install headers
install(
  DIRECTORY axel
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT headers
  FILES_MATCHING PATTERN "*.h"
  PATTERN "*/examples/*" EXCLUDE
  PATTERN "*/test/*" EXCLUDE
  PATTERN "*/website/*" EXCLUDE
)

# Install targets (lib)
install(
  TARGETS axel
  EXPORT axelTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(
  EXPORT axelTargets
  NAMESPACE axel::
  DESTINATION ${AXEL_CONFIG_INSTALL_DIR}
)
