/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/blend_shape_skinning.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/character.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/checks.h"
#include "momentum/math/mesh.h"

#include <dispenso/parallel_for.h>

namespace momentum {

template <typename T>
BlendWeightsT<T> extractBlendWeights(
    const ParameterTransform& paramTransform,
    const ModelParametersT<T>& modelParams) {
  BlendWeightsT<T> result = Eigen::VectorX<T>::Zero(paramTransform.blendShapeParameters.size());
  for (Eigen::Index iBasis = 0; iBasis < result.size(); ++iBasis) {
    const auto paramIdx = paramTransform.blendShapeParameters[iBasis];
    if (paramIdx >= 0 && paramIdx < modelParams.size()) {
      result(iBasis) = modelParams(paramIdx);
    }
  }
  return result;
}

template <typename T>
BlendWeightsT<T> extractFaceExpressionBlendWeights(
    const ParameterTransform& paramTransform,
    const ModelParametersT<T>& modelParams) {
  BlendWeightsT<T> result = Eigen::VectorX<T>::Zero(paramTransform.faceExpressionParameters.size());
  for (Eigen::Index iBasis = 0; iBasis < result.size(); ++iBasis) {
    const auto paramIdx = paramTransform.faceExpressionParameters[iBasis];
    if (paramIdx >= 0 && paramIdx < modelParams.size()) {
      result(iBasis) = modelParams(paramIdx);
    }
  }
  return result;
}

template <typename T>
void skinWithBlendShapes(
    const Character& character,
    const SkeletonStateT<T>& state,
    const ModelParametersT<T>& modelParams,
    MeshT<T>& outputMesh) {
  skinWithBlendShapes(
      character, state, extractBlendWeights(character.parameterTransform, modelParams), outputMesh);
}

template <typename T>
void skinWithBlendShapes(
    const Character& character,
    const SkeletonStateT<T>& state,
    const BlendWeightsT<T>& blendWeights,
    MeshT<T>& outputMesh) {
  MT_CHECK((bool)character.skinWeights);
  MT_CHECK((bool)character.mesh);

  const auto& inverseBindPose = character.inverseBindPose;
  const auto& skin = *character.skinWeights;

  // some sanity checks
  MT_CHECK(
      state.jointState.size() == inverseBindPose.size(),
      "{} is not {}",
      state.jointState.size(),
      inverseBindPose.size());
  MT_CHECK(
      skin.index.rows() == gsl::narrow<int>(character.mesh->vertices.size()),
      "{} is not {}",
      skin.index.rows(),
      character.mesh->vertices.size());
  MT_CHECK(
      skin.weight.rows() == gsl::narrow<int>(character.mesh->vertices.size()),
      "{} is not {}",
      skin.weight.rows(),
      character.mesh->vertices.size());

  // first create a list of transformations from bindpose to final output pose
  std::vector<Eigen::Matrix4<T>> transformations(state.jointState.size());
  for (size_t i = 0; i < state.jointState.size(); i++) {
    transformations[i] =
        (state.jointState[i].transformation * inverseBindPose[i].cast<T>()).matrix();
  }

  MT_CHECK(
      !character.blendShape ||
          blendWeights.size() <= character.blendShape->getShapeVectors().cols(),
      "blendWeights.size(): {}; ShapeVectors.cols(): {}",
      blendWeights.size(),
      character.blendShape->getShapeVectors().cols());

  outputMesh.vertices.resize(character.mesh->vertices.size(), Eigen::Vector3<T>::Zero());

  // go over all vertices and perform transformation
  // SSE version for linux/windows
  dispenso::parallel_for(
      dispenso::makeChunkedRange(0, skin.index.rows(), dispenso::ParForChunking::kAuto),
      [&blendWeights, &character, &transformations, &outputMesh](
          const size_t rangeBegin, const size_t rangeEnd) {
        const auto& skin = *character.skinWeights;
        for (size_t iVert = rangeBegin; iVert != rangeEnd; iVert++) {
          // Compute rest position from blend shape:
          Eigen::Vector3<T> p_rest;
          if (character.blendShape) {
            const auto& blendShape = *character.blendShape;
            p_rest = (blendShape.getBaseShape()[iVert].cast<T>() +
                      blendShape.getShapeVectors()
                              .block(3 * iVert, 0, 3, blendWeights.size())
                              .template cast<T>() *
                          blendWeights.v)
                         .eval();
          } else {
            p_rest = character.mesh->vertices[iVert].cast<T>().eval();
          }

          Eigen::Vector3<T> p_skinned = Eigen::Vector3<T>::Zero();
          for (uint32_t k = 0; k < kMaxSkinJoints; ++k) {
            const auto bIndex = skin.index(iVert, k);
            const T bWeight = skin.weight(iVert, k);
            if (bWeight == 0) {
              break;
            }
            p_skinned +=
                (bWeight * (transformations[bIndex] * p_rest.homogeneous())).template head<3>();
          }
          outputMesh.vertices[iVert] = p_skinned;
        }
      });

  // Because of the blend shapes, we can't "skin" the normals so we just have to compute them from
  // the mesh.
  outputMesh.updateNormals();
}

template void skinWithBlendShapes(
    const Character& character,
    const SkeletonStateT<float>& state,
    const BlendWeightsT<float>& modelParams,
    MeshT<float>& outputMesh);

template void skinWithBlendShapes(
    const Character& character,
    const SkeletonStateT<double>& state,
    const BlendWeightsT<double>& modelParams,
    MeshT<double>& outputMesh);

template void skinWithBlendShapes(
    const Character& character,
    const SkeletonStateT<float>& state,
    const ModelParametersT<float>& modelParams,
    MeshT<float>& outputMesh);

template void skinWithBlendShapes(
    const Character& character,
    const SkeletonStateT<double>& state,
    const ModelParametersT<double>& modelParams,
    MeshT<double>& outputMesh);

template BlendWeightsT<float> extractBlendWeights<float>(
    const ParameterTransform& paramTransform,
    const ModelParametersT<float>& modelParams);

template BlendWeightsT<double> extractBlendWeights<double>(
    const ParameterTransform& paramTransform,
    const ModelParametersT<double>& modelParams);

template BlendWeightsT<float> extractFaceExpressionBlendWeights<float>(
    const ParameterTransform& paramTransform,
    const ModelParametersT<float>& modelParams);

template BlendWeightsT<double> extractFaceExpressionBlendWeights<double>(
    const ParameterTransform& paramTransform,
    const ModelParametersT<double>& modelParams);

} // namespace momentum
