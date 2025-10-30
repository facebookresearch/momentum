/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/math/types.h>

namespace momentum {

/// Save motion data using character parameter mapping
///
/// @param filename Output file path
/// @param poses Motion poses as span of vectors, each vector represents one frame
/// @param scale Joint scale parameters
/// @param character Character definition for parameter mapping
void saveMmo(
    const std::string& filename,
    std::span<const VectorXf> poses,
    const VectorXf& scale,
    const Character& character);

/// Save motion data with optional additional parameters
///
/// @param filename Output file path
/// @param poses Motion poses matrix (parameters x frames)
/// @param scale Joint scale parameters
/// @param character Character definition for parameter mapping
/// @param additionalParameters Extra parameters to store alongside poses
/// @param additionalParameterNames Names for additional parameters
void saveMmo(
    const std::string& filename,
    const MatrixXf& poses,
    const VectorXf& scale,
    const Character& character,
    const MatrixXf& additionalParameters = MatrixXf(),
    std::span<const std::string> additionalParameterNames = std::vector<std::string>());

/// Save motion data with explicit parameter and joint names
///
/// @param filename Output file path
/// @param poses Motion poses matrix (parameters x frames)
/// @param scale Joint scale parameters
/// @param parameterNames Names of motion parameters
/// @param jointNames Names of skeleton joints
void saveMmo(
    const std::string& filename,
    const MatrixXf& poses,
    const VectorXf& scale,
    std::span<const std::string> parameterNames,
    std::span<const std::string> jointNames);

/// Load motion data from file
///
/// @param filename Input file path
/// @return Tuple of (poses, scale, parameter names, joint names)
std::tuple<MatrixXf, VectorXf, std::vector<std::string>, std::vector<std::string>> loadMmo(
    const std::string& filename);

/// Load motion data and map to character parameter space
///
/// @param filename Input file path
/// @param character Target character for parameter mapping
/// @return Tuple of (mapped poses, mapped scale)
std::tuple<MatrixXf, VectorXf> loadMmo(const std::string& filename, const Character& character);

/// Extract auxiliary data from motion parameters
///
/// Auxiliary parameters are identified by names wrapped with double underscores (e.g., "__param__")
///
/// @param poses Motion poses matrix
/// @param parameterNames Names of all parameters
/// @return Tuple of (auxiliary data matrix, auxiliary parameter names without underscores)
std::tuple<MatrixXf, std::vector<std::string>> getAuxiliaryDataFromMotion(
    const MatrixXf& poses,
    std::span<const std::string> parameterNames);

/// Map motion data from one parameter space to character parameter space
///
/// @param poses Source motion poses
/// @param offsets Source joint offsets
/// @param parameterNames Source parameter names
/// @param jointNames Source joint names
/// @param character Target character definition
/// @return Tuple of (mapped poses, mapped offsets)
std::tuple<MatrixXf, VectorXf> mapMotionToCharacter(
    const MatrixXf& poses,
    const VectorXf& offsets,
    std::span<const std::string> parameterNames,
    std::span<const std::string> jointNames,
    const Character& character);

} // namespace momentum
