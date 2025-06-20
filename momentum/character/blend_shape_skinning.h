/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/types.h>
#include <momentum/math/fwd.h>

namespace momentum {

/// @file blend_shape_skinning.h
///
/// This file provides functions for applying blend shape deformations and skeletal transformations
/// to character meshes. It contains two main sets of functionality:
///
/// 1. Skinning functions:
///    - skinWithBlendShapes(character, state, blendWeights, outputMesh): Use when you already have
///      blend weights calculated and want to apply them directly.
///    - skinWithBlendShapes(character, state, modelParams, outputMesh): Use when you have model
///      parameters and need to extract blend weights before skinning. This is more convenient for
///      high-level character animation where you work with model parameters.
///
/// 2. Weight extraction functions:
///    - extractBlendWeights(): Use to get blend shape weights from model parameters for
///    identity/shape
///      blend shapes (e.g., body shape variations).
///    - extractFaceExpressionBlendWeights(): Use to get blend shape weights specifically for facial
///      expressions, which are stored separately from regular blend shapes.

/// Applies blend shape deformations and skeletal transformations to a mesh.
///
/// Computes the final vertex positions by applying blend shape offsets to the base mesh,
/// then transforming the resulting vertices using linear blend skinning based on the
/// skeleton state. Use this version when you already have pre-calculated blend weights.
///
/// @param character Character containing mesh, blend shapes, and skinning data
/// @param state Current pose of the skeleton
/// @param blendWeights Weights for each blend shape
/// @param outputMesh Mesh to store the resulting deformed vertices
template <typename T>
void skinWithBlendShapes(
    const Character& character,
    const SkeletonStateT<T>& state,
    const BlendWeightsT<T>& blendWeights,
    MeshT<T>& outputMesh);

/// Applies blend shape deformations and skeletal transformations to a mesh.
///
/// Overload that extracts blend shape weights from model parameters before skinning.
/// Use this version when working with high-level character animation where you have
/// model parameters rather than direct blend weights.
///
/// @param character Character containing mesh, blend shapes, and skinning data
/// @param state Current pose of the skeleton
/// @param modelParams Model parameters containing blend shape weights
/// @param outputMesh Mesh to store the resulting deformed vertices
template <typename T>
void skinWithBlendShapes(
    const Character& character,
    const SkeletonStateT<T>& state,
    const ModelParametersT<T>& modelParams,
    MeshT<T>& outputMesh);

/// Extracts blend shape weights from model parameters.
///
/// Maps from the model parameter space to blend shape weights using the parameter transform.
/// Use this function to get weights for identity/shape blend shapes (e.g., body shape variations).
///
/// @param paramTransform Mapping between model parameters and blend shape parameters
/// @param modelParams Current model parameters
/// @return Vector of blend shape weights
template <typename T>
BlendWeightsT<T> extractBlendWeights(
    const ParameterTransform& paramTransform,
    const ModelParametersT<T>& modelParams);

/// Extracts facial expression blend shape weights from model parameters.
///
/// Maps from the model parameter space to facial expression blend shape weights
/// using the parameter transform. Use this function specifically for facial animation
/// parameters, which are stored separately from regular blend shapes.
///
/// @param paramTransform Mapping between model parameters and face expression parameters
/// @param modelParams Current model parameters
/// @return Vector of facial expression blend shape weights
template <typename T>
BlendWeightsT<T> extractFaceExpressionBlendWeights(
    const ParameterTransform& paramTransform,
    const ModelParametersT<T>& modelParams);

} // namespace momentum
