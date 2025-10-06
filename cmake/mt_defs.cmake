# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if(NOT Python3_EXECUTABLE)
  message(FATAL_ERROR "Python3_EXECUTABLE is needed")
endif()

function(mt_append_filelist name rootdir vars_path_in vars_path_out outputvar)
  # configure_file adds its input to the list of CMAKE_RERUN dependencies
  configure_file(
    ${vars_path_in}
    ${vars_path_out}
  )
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "exec(open('${vars_path_in}').read());print(';'.join(['${rootdir}' + x for x in ${name}]))"
    WORKING_DIRECTORY "${rootdir}"
    RESULT_VARIABLE _retval
    OUTPUT_VARIABLE _tempvar
  )
  if(NOT _retval EQUAL 0)
    message(FATAL_ERROR "Failed to fetch filelist ${name} from build_variables.bzl")
  endif()
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  list(APPEND ${outputvar} ${_tempvar})
  set(${outputvar} "${${outputvar}}" PARENT_SCOPE)
endfunction()

function(mt_append_momentum_filelist name outputvar)
  mt_append_filelist(
    ${name}
    "${PROJECT_SOURCE_DIR}/momentum/"
    ${PROJECT_SOURCE_DIR}/cmake/build_variables.bzl
    ${PROJECT_BINARY_DIR}/cmake/build_variables.bzl
    ${outputvar}
  )
  set(${outputvar} "${${outputvar}}" PARENT_SCOPE)
endfunction()

function(mt_append_pymomentum_filelist name outputvar)
  mt_append_filelist(
    ${name}
    "${PROJECT_SOURCE_DIR}/pymomentum/"
    ${PROJECT_SOURCE_DIR}/pymomentum/cmake/build_variables.bzl
    ${PROJECT_BINARY_DIR}/pymomentum/cmake/build_variables.bzl
    ${outputvar}
  )
  set(${outputvar} "${${outputvar}}" PARENT_SCOPE)
endfunction()

# mt_get_max(var [value1 value2...])
function(mt_get_max var)
  set(first YES)
  set(choice NO)
  foreach(item ${ARGN})
    if(first)
      set(choice ${item})
      set(first NO)
    elseif(choice LESS ${item})
      set(choice ${item})
    endif()
  endforeach(item)
  set(${var} ${choice} PARENT_SCOPE)
endfunction()

# mt_get_max_string_length(var [value1 value2...])
function(mt_get_max_string_length var)
  foreach(item ${ARGN})
    string(LENGTH ${item} length)
    list(APPEND list ${length})
  endforeach()
  mt_get_max(choice ${list})
  set(${var} ${choice} PARENT_SCOPE)
endfunction()

# Function to print each flag on a new line with indentation
# mt_print_flags(<variable>)
function(mt_print_flags var_name)
  if(NOT "${${var_name}}" STREQUAL "")
    string(REPLACE " " ";" FLAGS_LIST "${${var_name}}")
    message(STATUS "${var_name}:")
    foreach(flag IN LISTS FLAGS_LIST)
      message(STATUS "  - ${flag}")
    endforeach()
  else()
    message(STATUS "${var_name}: (not set)")
  endif()
endfunction()

# mt_option(<variable> "<help_text>" <value>)
function(mt_option variable help_text default_value)
  set_property(
    GLOBAL PROPERTY MOMENTUM_DETAIL_PROPERTY_OPTION_VARIABLE "${variable}" APPEND
  )
  set_property(
    GLOBAL PROPERTY MOMENTUM_DETAIL_property_option_help_text "${help_text}" APPEND
  )
  set_property(
    GLOBAL PROPERTY MOMENTUM_DETAIL_property_option_default_value "${default_value}"
    APPEND
  )

  # Add option
  option(${variable} ${help_text} ${default_value})

  # Normalize boolean value variants (e.g. 1/0, On/Off, TRUE/FALSE) to ON/OFF
  if(${variable})
    set(${variable} ON PARENT_SCOPE)
  else()
    set(${variable} OFF PARENT_SCOPE)
  endif()

endfunction()

function(mt_print_options)
  # Print the header
  message(STATUS "[ Options ]")

  get_property(
    option_variables GLOBAL PROPERTY MOMENTUM_DETAIL_PROPERTY_OPTION_VARIABLE
  )
  get_property(
    option_help_texts GLOBAL PROPERTY MOMENTUM_DETAIL_property_option_help_text
  )
  get_property(
    option_default_values GLOBAL
    PROPERTY MOMENTUM_DETAIL_property_option_default_value
  )

  mt_get_max_string_length(option_variable_max_len ${option_variables})
  list(LENGTH option_variables option_count)
  math(EXPR option_count "${option_count} - 1")
  foreach(val RANGE ${option_count})
    list(GET option_variables ${val} option_variable)
    list(GET option_default_values ${val} option_default_value)

    set(option_str "- ${option_variable}")
    set(spaces "")
    string(LENGTH ${option_variable} option_variable_len)
    math(EXPR space_count "${option_variable_max_len} - ${option_variable_len}")
    foreach(loop_var RANGE ${space_count})
      set(option_str "${option_str} ")
    endforeach()

    set(option_str "${option_str}: ${${option_variable}}")

    if(${option_variable} STREQUAL option_default_value)
      set(option_str "${option_str} [default]")
    endif()

    message(STATUS "${option_str}")
  endforeach()

  message(STATUS "")
endfunction()

function(mt_library)
  set(prefix _ARG)
  set(options
    NO_INSTALL
  )
  set(oneValueArgs
    NAME
  )
  set(multiValueArgs
    HEADERS
    HEADERS_VARS
    PYMOMENTUM_HEADERS_VARS
    PRIVATE_HEADERS
    PRIVATE_HEADERS_VARS
    PYMOMENTUM_PRIVATE_HEADERS_VARS
    SOURCES
    SOURCES_VARS
    PYMOMENTUM_SOURCES_VARS
    PUBLIC_INCLUDE_DIRECTORIES
    PRIVATE_INCLUDE_DIRECTORIES
    PUBLIC_LINK_LIBRARIES
    PRIVATE_LINK_LIBRARIES
    PUBLIC_COMPILE_DEFINITIONS
    PRIVATE_COMPILE_DEFINITIONS
    PUBLIC_COMPILE_OPTIONS
    PRIVATE_COMPILE_OPTIONS
  )
  cmake_parse_arguments(
    "${prefix}"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  foreach(var ${_ARG_HEADERS_VARS})
    mt_append_momentum_filelist("${var}" _ARG_HEADERS)
  endforeach()

  foreach(var ${_ARG_PYMOMENTUM_HEADERS_VARS})
    mt_append_pymomentum_filelist("${var}" _ARG_HEADERS)
  endforeach()

  foreach(var ${_ARG_PRIVATE_HEADERS_VARS})
    mt_append_momentum_filelist("${var}" _ARG_PRIVATE_HEADERS)
  endforeach()

  foreach(var ${_ARG_PYMOMENTUM_PRIVATE_HEADERS_VARS})
    mt_append_pymomentum_filelist("${var}" _ARG_PRIVATE_HEADERS)
  endforeach()

  foreach(var ${_ARG_SOURCES_VARS})
    mt_append_momentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  foreach(var ${_ARG_PYMOMENTUM_SOURCES_VARS})
    mt_append_pymomentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  if("${_ARG_SOURCES}" STREQUAL "")
    set(library_type INTERFACE)
    set(public_or_interface INTERFACE)
    set(private_or_interface INTERFACE)
  else()
    set(library_type )
    set(public_or_interface PUBLIC)
    set(private_or_interface PRIVATE)
  endif()

  add_library(${_ARG_NAME}
    ${library_type}
    ${_ARG_HEADERS}
    ${_ARG_SOURCES}
  )
  target_sources(${_ARG_NAME} PRIVATE
    ${_ARG_PRIVATE_HEADERS}
  )
  target_include_directories(${_ARG_NAME}
    ${public_or_interface}
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> # TODO: Remove once momentum/pymomentum/pymomentum is moved to momentum/pymomentum
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      ${_ARG_PUBLIC_INCLUDE_DIRECTORIES}
  )
  target_include_directories(${_ARG_NAME} ${private_or_interface} ${_ARG_PRIVATE_INCLUDE_DIRECTORIES})
  target_compile_features(${_ARG_NAME} ${public_or_interface} cxx_std_20)

  target_link_libraries(${_ARG_NAME} ${public_or_interface} ${_ARG_PUBLIC_LINK_LIBRARIES})
  target_link_libraries(${_ARG_NAME} ${private_or_interface} ${_ARG_PRIVATE_LINK_LIBRARIES})

  if(MOMENTUM_ENABLE_PROFILING)
    target_link_libraries(${_ARG_NAME} ${public_or_interface} Tracy::TracyClient)
    target_compile_definitions(${_ARG_NAME} ${public_or_interface} -DMOMENTUM_WITH_TRACY_PROFILER=1)
  endif()

  set_target_properties(${_ARG_NAME} PROPERTIES
    OUTPUT_NAME momentum_${_ARG_NAME}
    POSITION_INDEPENDENT_CODE ON
  )
  target_compile_definitions(${_ARG_NAME}
    ${public_or_interface} ${_ARG_PUBLIC_COMPILE_DEFINITIONS}
    ${private_or_interface} ${_ARG_PRIVATE_COMPILE_DEFINITIONS}
  )
  if(MOMENTUM_ENABLE_SIMD)
    if(MSVC)
      target_compile_options(${_ARG_NAME} ${public_or_interface} "/arch:SSE2")
      target_compile_options(${_ARG_NAME} ${public_or_interface} "/arch:AVX")
      target_compile_options(${_ARG_NAME} ${public_or_interface} "/arch:AVX2")
    elseif(APPLE)
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        target_compile_options(${_ARG_NAME} ${public_or_interface} -march=armv8-a)
      else()
        target_compile_options(${_ARG_NAME} ${public_or_interface} -march=native)
      endif()
    else()
      target_compile_options(${_ARG_NAME} ${public_or_interface} -march=native)
    endif()
    target_compile_definitions(${_ARG_NAME}
      ${public_or_interface} -DMOMENTUM_ENABLE_SIMD=1
    )
  endif()
  target_compile_options(${_ARG_NAME}
    ${public_or_interface} ${_ARG_PUBLIC_COMPILE_OPTIONS}
    ${private_or_interface} ${_ARG_PRIVATE_COMPILE_OPTIONS}
  )

  if(MSVC)
    if(NOT library_type STREQUAL "INTERFACE")
      set_target_properties(${_ARG_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
  endif()

  if(NOT ${_ARG_NO_INSTALL})
    set_property(GLOBAL APPEND PROPERTY MOMENTUM_TARGETS ${_ARG_NAME})
  endif()
endfunction()

function(mt_executable)
  set(prefix _ARG)
  set(options
    NO_INSTALL
  )
  set(oneValueArgs
    NAME
  )
  set(multiValueArgs
    HEADERS
    SOURCES
    SOURCES_VARS
    INCLUDE_DIRECTORIES
    LINK_LIBRARIES
  )
  cmake_parse_arguments(
    "${prefix}"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  foreach(var ${_ARG_SOURCES_VARS})
    mt_append_momentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  add_executable(${_ARG_NAME}
    ${_ARG_HEADERS}
    ${_ARG_SOURCES}
  )
  target_include_directories(${_ARG_NAME}
    PRIVATE
      ${_ARG_INCLUDE_DIRECTORIES}
  )
  target_link_libraries(${_ARG_NAME}
    PRIVATE
      ${_ARG_LINK_LIBRARIES}
  )

  if(MOMENTUM_INSTALL_EXAMPLES AND NOT ${_ARG_NO_INSTALL})
    set_property(GLOBAL APPEND PROPERTY MOMENTUM_EXECUTABLES ${_ARG_NAME})
  endif()
endfunction()

function(mt_setup_gtest)
  set(prefix _ARG)
  set(options
  )
  set(oneValueArgs
    GIT_TAG
  )
  set(multiValueArgs
  )
  cmake_parse_arguments(
    "${prefix}"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  if(NOT _ARG_GIT_TAG)
    set(_ARG_GIT_TAG v1.16.0)
  endif()

  if(MOMENTUM_USE_SYSTEM_GOOGLETEST)
    find_package(GTest CONFIG REQUIRED)
  else()
    include(FetchContent)
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG ${_ARG_GIT_TAG}
    )

    # Set options
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)
  endif()
endfunction()

function(mt_test)
  set(prefix _ARG)
  set(options
  )
  set(oneValueArgs
    NAME
  endif()
  )
  set(multiValueArgs
    HEADERS
    SOURCES
    SOURCES_VARS
    PYMOMENTUM_SOURCES_VARS
    INCLUDE_DIRECTORIES
    LINK_LIBRARIES
    ENV
  )
  cmake_parse_arguments(
    "${prefix}"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  foreach(var ${_ARG_SOURCES_VARS})
    mt_append_momentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  foreach(var ${_ARG_PYMOMENTUM_SOURCES_VARS})
    mt_append_pymomentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  add_executable(${_ARG_NAME}
    ${_ARG_HEADERS}
    ${_ARG_SOURCES}
  )
  add_test(NAME ${_ARG_NAME} COMMAND $<TARGET_FILE:${_ARG_NAME}>)
  target_include_directories(${_ARG_NAME}
    PRIVATE
      ${_ARG_INCLUDE_DIRECTORIES}
  )
  target_link_libraries(${_ARG_NAME}
    PRIVATE
      ${_ARG_LINK_LIBRARIES}
      GTest::gmock
      GTest::gtest
      GTest::gtest_main
  )

  set(aggregated_env "")
  foreach(env ${_ARG_ENV})
    list(APPEND aggregated_env ${env})
  endforeach()
  set_tests_properties(${_ARG_NAME} PROPERTIES ENVIRONMENT "${aggregated_env}")

endfunction()

function(mt_python_binding)
  set(prefix _ARG)
  set(options
    NO_INSTALL
  )
  set(oneValueArgs
    NAME
    MODULE_NAME
  )
  set(multiValueArgs
    HEADERS
    HEADERS_VARS
    PYMOMENTUM_HEADERS_VARS
    PYMOMENTUM_SOURCES_VARS
    SOURCES
    SOURCES_VARS
    INCLUDE_DIRECTORIES
    LINK_LIBRARIES
    COMPILE_OPTIONS
  )
  cmake_parse_arguments(
    "${prefix}"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  foreach(var ${_ARG_HEADERS_VARS})
    mt_append_momentum_filelist("${var}" _ARG_HEADERS)
  endforeach()

  foreach(var ${_ARG_SOURCES_VARS})
    mt_append_momentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  foreach(var ${_ARG_PYMOMENTUM_HEADERS_VARS})
    mt_append_pymomentum_filelist("${var}" _ARG_HEADERS)
  endforeach()

  foreach(var ${_ARG_PYMOMENTUM_SOURCES_VARS})
    mt_append_pymomentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  pybind11_add_module(${_ARG_NAME}
    MODULE
    ${_ARG_HEADERS}
    ${_ARG_SOURCES}
  )
  target_include_directories(${_ARG_NAME}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> # TODO: Remove once momentum/pymomentum/pymomentum is moved to momentum/pymomentum
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      ${_ARG_INCLUDE_DIRECTORIES}
  )
  target_compile_features(${_ARG_NAME} PRIVATE cxx_std_20)
  target_link_libraries(${_ARG_NAME} PRIVATE ${_ARG_LINK_LIBRARIES})
  target_compile_options(${_ARG_NAME} PRIVATE ${_ARG_COMPILE_OPTIONS})

  if(NOT _ARG_MODULE_NAME)
    set(_ARG_MODULE_NAME ${_ARG_NAME})
  endif()
  set_target_properties(${_ARG_NAME} PROPERTIES
    OUTPUT_NAME "${_ARG_MODULE_NAME}"
  )

  if(NOT ${_ARG_NO_INSTALL})
    set_property(GLOBAL APPEND PROPERTY PYMOMENTUM_TARGETS_TO_INSTALL ${_ARG_NAME})
  endif()
endfunction()

function(mt_python_library)
  set(prefix _ARG)
  set(options
    NO_INSTALL
    PRESERVE_DIRECTORY_STRUCTURE
  )
  set(oneValueArgs
    NAME
    DESTINATION_SUBDIR
  )
  set(multiValueArgs
    PYMOMENTUM_SOURCES_VARS
  )
  cmake_parse_arguments(
    "${prefix}"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  foreach(var ${_ARG_PYMOMENTUM_SOURCES_VARS})
    mt_append_pymomentum_filelist("${var}" libs)
  endforeach()

  if(NOT ${_ARG_NO_INSTALL})
    if(${_ARG_PRESERVE_DIRECTORY_STRUCTURE})
      # Store files with their directory structure preserved
      foreach(lib ${libs})
        # Get relative path from pymomentum source directory
        file(RELATIVE_PATH rel_path "${PROJECT_SOURCE_DIR}/pymomentum" "${lib}")
        get_filename_component(file_dir "${rel_path}" DIRECTORY)

        if(file_dir STREQUAL "")
          # File is in root, install to pymomentum directly
          set(install_dest "pymomentum")
        else()
          # File is in subdirectory, preserve structure
          set(install_dest "pymomentum/${file_dir}")
        endif()

        # Store file and its destination
        set_property(GLOBAL APPEND PROPERTY PYMOMENTUM_STRUCTURED_LIBRARIES_TO_INSTALL "${lib}:${install_dest}")
      endforeach()
    elseif(_ARG_DESTINATION_SUBDIR)
      # Install to specific subdirectory
      set(install_dest "pymomentum/${_ARG_DESTINATION_SUBDIR}")
      foreach(lib ${libs})
        set_property(GLOBAL APPEND PROPERTY PYMOMENTUM_STRUCTURED_LIBRARIES_TO_INSTALL "${lib}:${install_dest}")
      endforeach()
    else()
      # Default behavior: install flat to pymomentum directory
      set_property(GLOBAL APPEND PROPERTY PYMOMENTUM_PYTHON_LIBRARIES_TO_INSTALL ${libs})
    endif()
  endif()
endfunction()

function(mt_install_pymomentum)
  set(prefix _ARG)
  set(options
  )
  set(oneValueArgs
    GIT_TAG
  )
  set(multiValueArgs
  )
  cmake_parse_arguments(
    "${prefix}"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  # Install C++ binding modules
  get_property(pymomentum_targets_to_install GLOBAL PROPERTY PYMOMENTUM_TARGETS_TO_INSTALL)
  install(
    TARGETS ${pymomentum_targets_to_install}
    DESTINATION pymomentum
  )

  # Install Python modules (flat structure, legacy behavior)
  get_property(pymomentum_python_libraries_to_install GLOBAL PROPERTY PYMOMENTUM_PYTHON_LIBRARIES_TO_INSTALL)
  if(pymomentum_python_libraries_to_install)
    install(
      FILES ${pymomentum_python_libraries_to_install}
      DESTINATION pymomentum
    )
  endif()

  # Install Python modules with preserved directory structure
  get_property(pymomentum_structured_libraries_to_install GLOBAL PROPERTY PYMOMENTUM_STRUCTURED_LIBRARIES_TO_INSTALL)
  if(pymomentum_structured_libraries_to_install)
    # Group files by destination directory
    set(destinations_processed "")
    foreach(file_dest_pair ${pymomentum_structured_libraries_to_install})
      # Split "file:destination" pair
      string(FIND "${file_dest_pair}" ":" colon_pos)
      string(SUBSTRING "${file_dest_pair}" 0 ${colon_pos} file_path)
      math(EXPR dest_start "${colon_pos} + 1")
      string(SUBSTRING "${file_dest_pair}" ${dest_start} -1 dest_path)

      # Check if we've already processed this destination
      list(FIND destinations_processed "${dest_path}" dest_index)
      if(dest_index EQUAL -1)
        # New destination, collect all files for this destination
        set(files_for_dest "")
        foreach(other_pair ${pymomentum_structured_libraries_to_install})
          string(FIND "${other_pair}" ":" other_colon_pos)
          math(EXPR other_dest_start "${other_colon_pos} + 1")
          string(SUBSTRING "${other_pair}" ${other_dest_start} -1 other_dest_path)

          if("${other_dest_path}" STREQUAL "${dest_path}")
            string(SUBSTRING "${other_pair}" 0 ${other_colon_pos} other_file_path)
            list(APPEND files_for_dest "${other_file_path}")
          endif()
        endforeach()

        # Install all files for this destination
        install(
          FILES ${files_for_dest}
          DESTINATION ${dest_path}
        )

        # Mark this destination as processed
        list(APPEND destinations_processed "${dest_path}")
      endif()
    endforeach()
  endif()
endfunction()
