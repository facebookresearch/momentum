/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>
#include <momentum/character/types.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/marker/coordinate_system.h>
#include <pybind11/numpy.h>

#include <optional>
#include <string>

// Forward declarations
namespace momentum {
struct FbxCoordSystemInfo;
} // namespace momentum

namespace pymomentum {

// We need to wrap around momentum io functions because the motion matrix in
// pymomentum is a transpose. Momentum represents motion as numParameters X
// numFrames, and pymomentum represents motion as numFrames x numParameters.

// Using row-major matrices for input/output avoids some extra copies in numpy:
using RowMatrixf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

momentum::Character loadGLTFCharacterFromFile(const std::string& path);
momentum::Character loadGLTFCharacterFromBytes(const pybind11::bytes& bytes);

/// Utility function to transpose motion parameters for C++ internal format
momentum::MotionParameters transpose(const momentum::MotionParameters& motionParameters);

void saveGLTFCharacterToFile(
    const std::string& path,
    const momentum::Character& character,
    float fps,
    const std::optional<const momentum::MotionParameters>& motion,
    const std::optional<const std::tuple<std::vector<std::string>, Eigen::VectorXf>>& offsets,
    const std::optional<const std::vector<std::vector<momentum::Marker>>>& markers,
    const std::optional<const momentum::FileSaveOptions>& options);

void saveGLTFCharacterToFileFromSkelStates(
    const std::string& path,
    const momentum::Character& character,
    float fps,
    const pybind11::array_t<float>& skelStates,
    const std::optional<const std::vector<std::vector<momentum::Marker>>>& markers,
    const std::optional<const momentum::FileSaveOptions>& options);

void saveFBXCharacterToFile(
    const std::string& path,
    const momentum::Character& character,
    float fps,
    const std::optional<const Eigen::MatrixXf>& motion,
    const std::optional<const Eigen::VectorXf>& offsets,
    const std::optional<const std::vector<std::vector<momentum::Marker>>>& markers,
    const momentum::FileSaveOptions& options = momentum::FileSaveOptions());

void saveFBXCharacterToFileWithJointParams(
    const std::string& path,
    const momentum::Character& character,
    float fps,
    const std::optional<const Eigen::MatrixXf>& jointParams,
    const std::optional<const std::vector<std::vector<momentum::Marker>>>& markers,
    const momentum::FileSaveOptions& options = momentum::FileSaveOptions());

void saveFBXCharacterToFileWithSkelStates(
    const std::string& path,
    const momentum::Character& character,
    float fps,
    const pybind11::array_t<float>& skelStates,
    const std::optional<const std::vector<std::vector<momentum::Marker>>>& markers,
    const momentum::FileSaveOptions& options = momentum::FileSaveOptions());

void saveCharacterToFile(
    const std::string& path,
    const momentum::Character& character,
    float fps,
    const std::optional<const Eigen::MatrixXf>& motion,
    const std::optional<const std::vector<std::vector<momentum::Marker>>>& markers,
    const momentum::FileSaveOptions& options = momentum::FileSaveOptions());

void saveCharacterToFileWithSkelStates(
    const std::string& path,
    const momentum::Character& character,
    float fps,
    const pybind11::array_t<float>& skelStates,
    const std::optional<const std::vector<std::vector<momentum::Marker>>>& markers);

std::tuple<momentum::Character, RowMatrixf, Eigen::VectorXf, float> loadGLTFCharacterWithMotion(
    const std::string& gltfFilename);

std::tuple<momentum::Character, RowMatrixf, Eigen::VectorXf, float>
loadGLTFCharacterWithMotionFromBytes(const pybind11::bytes& gltfBytes);

std::tuple<momentum::Character, RowMatrixf, Eigen::VectorXf, float>
loadGLTFCharacterWithMotionModelParameterScales(const std::string& gltfFilename);

std::tuple<momentum::Character, RowMatrixf, Eigen::VectorXf, float>
loadGLTFCharacterWithMotionModelParameterScalesFromBytes(const pybind11::bytes& gltfBytes);

std::tuple<momentum::Character, pybind11::array_t<float>, std::vector<float>>
loadGLTFCharacterWithSkelStates(const std::string& gltfFilename);

std::tuple<momentum::Character, pybind11::array_t<float>, std::vector<float>>
loadGLTFCharacterWithSkelStatesFromBytes(const pybind11::bytes& gltfBytes);

std::string toGLTF(const momentum::Character& character);

std::tuple<RowMatrixf, std::vector<std::string>, Eigen::VectorXf, std::vector<std::string>>
loadMotion(const std::string& gltfFilename);

std::vector<momentum::MarkerSequence> loadMarkersFromFile(
    const std::string& path,
    bool mainSubjectOnly = true,
    momentum::UpVector up = momentum::UpVector::Y);

/// Utility function to convert pybind11::array_t<float> to SkeletonState vector
/// This is shared between saveGLTFCharacterToFileFromSkelStates and
/// GltfBuilder::addSkeletonStates
std::vector<momentum::SkeletonState> arrayToSkeletonStates(
    const pybind11::array_t<float>& skelStates,
    const momentum::Character& character);

bool isFbxsdkAvailable();

} // namespace pymomentum
