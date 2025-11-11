# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set(_fbxsdk_version "2020.3.7")

message(DEBUG "Looking for FBX SDK version: ${_fbxsdk_version}")

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(_fbxsdk_approot "/Applications/Autodesk/FBX SDK")
  set(_fbxsdk_libdir_release "lib/clang/release")
  set(_fbxsdk_libdir_debug "lib/clang/debug")
  list(APPEND _fbxsdk_libnames_release "libalembic.a" "libfbxsdk.a")
  list(APPEND _fbxsdk_libnames_debug "libalembic.a" "libfbxsdk.a")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(_fbxsdk_approot "C:/Program Files/Autodesk/FBX/FBX SDK")
  set(_fbxsdk_libdir_release "lib/x64/release")
  set(_fbxsdk_libdir_debug "lib/x64/debug")
  list(APPEND _fbxsdk_libnames_release "libfbxsdk-md.lib" "alembic-md.lib" "libxml2-md.lib" "zlib-md.lib")
  list(APPEND _fbxsdk_libnames_debug "libfbxsdk-md.lib" "alembic-md.lib" "libxml2-md.lib" "zlib-md.lib")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_fbxsdk_approot "/usr/fbxsdk")
  set(_fbxsdk_libdir_release "lib/release")
  set(_fbxsdk_libdir_debug "lib/debug")
  list(APPEND _fbxsdk_libnames_release "libalembic.a" "libfbxsdk.a")
  list(APPEND _fbxsdk_libnames_debug "libalembic.a" "libfbxsdk.a")
else()
  message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
endif()

# overwrite if FBXSDK_PATH is defined
if(DEFINED ENV{FBXSDK_PATH})
  set(_fbxsdk_approot "$ENV{FBXSDK_PATH}")
endif()

# should point the the FBX SDK installation dir
set(_fbxsdk_root "${_fbxsdk_approot}/${_fbxsdk_version}")
message(DEBUG "_fbxsdk_root: ${_fbxsdk_root}")

# find header dir
find_path(
  FBXSDK_INCLUDE_DIR "fbxsdk.h"
  PATHS ${_fbxsdk_root}
  PATH_SUFFIXES "include"
)
message(DEBUG "FBXSDK_INCLUDE_DIR: ${FBXSDK_INCLUDE_DIR}")

# find release libs
foreach(libname ${_fbxsdk_libnames_release})
  find_library(
    lib_${libname} ${libname}
    PATHS ${_fbxsdk_root}
    PATH_SUFFIXES ${_fbxsdk_libdir_release}
  )
  list(APPEND FBXSDK_LIBRARIES ${lib_${libname}})
endforeach()
message(DEBUG "FBXSDK_LIBRARIES: ${FBXSDK_LIBRARIES}")

# find debug libs
foreach(libname ${_fbxsdk_libnames_debug})
  find_library(
    lib_${libname}_debug ${libname}
    PATHS ${_fbxsdk_root}
    PATH_SUFFIXES ${_fbxsdk_libdir_debug}
  )
  list(APPEND FBXSDK_LIBRARIES_DEBUG ${lib_${libname}_debug})
endforeach()
message(DEBUG "FBXSDK_LIBRARIES_DEBUG: ${FBXSDK_LIBRARIES_DEBUG}")

set(
  required_vars
    FBXSDK_INCLUDE_DIR
    FBXSDK_LIBRARIES
    FBXSDK_LIBRARIES_DEBUG
)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  list(APPEND FBXSDK_LIBRARIES "-framework CoreFoundation")
  list(APPEND FBXSDK_LIBRARIES_DEBUG "-framework CoreFoundation")
  find_package(Iconv MODULE QUIET)
  find_package(LibXml2 MODULE QUIET)
  find_package(ZLIB MODULE QUIET)
  list(APPEND FBXSDK_LIBRARIES LibXml2::LibXml2 Iconv::Iconv ${ZLIB_LIBRARIES})
  list(APPEND FBXSDK_LIBRARIES_DEBUG LibXml2::LibXml2 Iconv::Iconv ${ZLIB_LIBRARIES})
  list(APPEND required_vars LibXml2_FOUND Iconv_FOUND ZLIB_LIBRARIES)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(LibXml2 MODULE QUIET)
  find_package(ZLIB MODULE QUIET)
  list(APPEND FBXSDK_LIBRARIES LibXml2::LibXml2 ${ZLIB_LIBRARIES} ${CMAKE_DL_LIBS})
  list(APPEND FBXSDK_LIBRARIES_DEBUG LibXml2::LibXml2 ${ZLIB_LIBRARIES} ${CMAKE_DL_LIBS})
  list(APPEND required_vars LibXml2_FOUND ZLIB_LIBRARIES)
endif()

# Set (NAME)_FOUND if all the variables and the version are satisfied.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FbxSdk
  FAIL_MESSAGE "Failed to find FBX SDK. Please download the required FBX SDK 2020.3.7 from https://aps.autodesk.com/developer/overview/fbx-sdk. After installation, set FBXSDK_PATH to the installation directory if it's not installed to the default path."
  REQUIRED_VARS ${required_vars}
  VERSION_VAR _fbxsdk_version
)

if(FbxSdk_FOUND)
  add_library(fbxsdk::fbxsdk INTERFACE IMPORTED)
  set_target_properties(fbxsdk::fbxsdk PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${FBXSDK_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${FBXSDK_LIBRARIES}"
  )

  add_library(fbxsdk::fbxsdk_debug INTERFACE IMPORTED)
  set_target_properties(fbxsdk::fbxsdk_debug PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${FBXSDK_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${FBXSDK_LIBRARIES_DEBUG}"
  )
endif()
