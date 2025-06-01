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

// calculate skinning
template <typename T>
void skinWithBlendShapes(
    const Character& character,
    const SkeletonStateT<T>& state,
    const BlendWeightsT<T>& blendWeights,
    MeshT<T>& outputMesh);

template <typename T>
void skinWithBlendShapes(
    const Character& character,
    const SkeletonStateT<T>& state,
    const ModelParametersT<T>& modelParameters,
    MeshT<T>& outputMesh);

template <typename T>
BlendWeightsT<T> extractBlendWeights(
    const ParameterTransform& paramTransform,
    const ModelParametersT<T>& modelParams);

template <typename T>
BlendWeightsT<T> extractFaceExpressionBlendWeights(
    const ParameterTransform& paramTransform,
    const ModelParametersT<T>& modelParams);

} // namespace momentum
