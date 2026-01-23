/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/linear_skinning.h"

#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/mesh.h"

#include <dispenso/parallel_for.h>
#include <gsl/narrow>

namespace momentum {

template <typename T>
std::vector<Eigen::Matrix4<T>> computeSkinningTransforms(
    typename DeduceSpanType<const JointStateT<T>>::type jointState,
    const TransformationListT<T>& inverseBindPose) {
  MT_CHECK(
      jointState.size() == inverseBindPose.size(),
      "{} vs {}",
      jointState.size(),
      inverseBindPose.size());

  std::vector<Eigen::Matrix4<T>> transforms(inverseBindPose.size());
  for (size_t i = 0; i < inverseBindPose.size(); i++) {
    transforms[i].noalias() = (jointState[i].transform * inverseBindPose[i]).matrix();
  }
  return transforms;
}

template <typename T>
std::vector<Vector3<T>> applySSD(
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<T>>::type points,
    typename DeduceSpanType<const Eigen::Matrix4<T>>::type skinningTransforms) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      skin.index.rows() == gsl::narrow<int>(points.size()),
      "{} vs {}",
      skin.index.rows(),
      points.size());
  MT_CHECK(
      skin.weight.rows() == gsl::narrow<int>(points.size()),
      "{} vs {}",
      skin.weight.rows(),
      points.size());

  std::vector<Vector3<T>> result(points.size(), Vector3<T>::Zero());

  // go over all vertices and perform transformation
  dispenso::ParForOptions options;
  options.minItemsPerChunk = 1024u;

  dispenso::parallel_for(
      dispenso::makeChunkedRange(0, skin.index.rows(), dispenso::ParForChunking::kAuto),
      [&](const size_t rangeBegin, const size_t rangeEnd) {
        for (size_t i = rangeBegin; i != rangeEnd; i++) {
          // grab vertex
          const Vector3<T>& pos = points[i];
          auto& output = result[i];
          output.setZero();

          // loop over the weights
          for (size_t j = 0; j < kMaxSkinJoints; j++) {
            // get pointer to transformation and weight float
            const auto& weight = skin.weight(i, j);
            if (weight == 0.0f) {
              break;
            }

            MT_CHECK(
                skin.index(i, j) < skinningTransforms.size(),
                "skin.index({}, {}): {} vs {}",
                i,
                j,
                skin.index(i, j),
                skinningTransforms.size());
            const auto& transformation = skinningTransforms[skin.index(i, j)];

            // add up transforms: outputp += (transformation * (pos, 1)) * weight
            Eigen::Vector3<T> temp = transformation.template topRightCorner<3, 1>();
            temp.noalias() += transformation.template topLeftCorner<3, 3>() * pos;
            output.noalias() += temp * weight;
          }
        }
      },
      options);

  return result;
}

template <typename T>
std::vector<Vector3<T>> applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<T>>::type points,
    typename DeduceSpanType<const JointStateT<T>>::type jointState) {
  const auto skinningTransforms = computeSkinningTransforms<T>(jointState, inverseBindPose);
  return applySSD<T>(skin, points, skinningTransforms);
}

template <typename T>
std::vector<Vector3<T>> applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<T>>::type points,
    const SkeletonStateT<T>& state) {
  return applySSD<T>(inverseBindPose, skin, points, state.jointState);
}

template <typename T>
std::vector<Vector3<T>> applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<T>>::type points,
    const TransformationListT<T>& worldTransforms) {
  MT_CHECK(
      worldTransforms.size() == inverseBindPose.size(),
      "{} vs {}",
      worldTransforms.size(),
      inverseBindPose.size());

  std::vector<Eigen::Matrix4<T>> skinningTransforms(inverseBindPose.size());
  for (size_t i = 0; i < inverseBindPose.size(); i++) {
    skinningTransforms[i].noalias() = (worldTransforms[i] * inverseBindPose[i]).matrix();
  }
  return applySSD<T>(skin, points, skinningTransforms);
}

template <typename T>
std::vector<Vector3<T>> applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<T>>::type points,
    typename DeduceSpanType<const TransformT<T>>::type worldTransforms) {
  MT_CHECK(
      worldTransforms.size() == inverseBindPose.size(),
      "{} vs {}",
      worldTransforms.size(),
      inverseBindPose.size());

  std::vector<Eigen::Matrix4<T>> skinningTransforms(inverseBindPose.size());
  for (size_t i = 0; i < inverseBindPose.size(); i++) {
    skinningTransforms[i].noalias() = (worldTransforms[i] * inverseBindPose[i]).matrix();
  }
  return applySSD<T>(skin, points, skinningTransforms);
}

template <typename T>
void applySSD(
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    typename DeduceSpanType<const Eigen::Matrix4<T>>::type skinningTransforms,
    MeshT<T>& outputMesh) {
  MT_PROFILE_FUNCTION();

  MT_CHECK(
      skin.index.rows() == gsl::narrow<int>(mesh.vertices.size()),
      "{} vs {}",
      skin.index.rows(),
      mesh.vertices.size());
  MT_CHECK(
      skin.weight.rows() == gsl::narrow<int>(mesh.vertices.size()),
      "{} vs {}",
      skin.weight.rows(),
      mesh.vertices.size());
  MT_CHECK(
      outputMesh.vertices.size() == mesh.vertices.size(),
      "{} vs {}",
      outputMesh.vertices.size(),
      mesh.vertices.size());
  MT_CHECK(
      outputMesh.normals.size() == mesh.normals.size(),
      "{} vs {}",
      outputMesh.normals.size(),
      mesh.normals.size());
  MT_CHECK(
      outputMesh.faces.size() == mesh.faces.size(),
      "{} vs {}",
      outputMesh.faces.size(),
      mesh.faces.size());

  // go over all vertices and perform transformation
  dispenso::ParForOptions options;
  options.minItemsPerChunk = 1024u;

  dispenso::parallel_for(
      dispenso::makeChunkedRange(0, skin.index.rows(), dispenso::ParForChunking::kAuto),
      [&](const size_t rangeBegin, const size_t rangeEnd) {
        for (size_t i = rangeBegin; i != rangeEnd; i++) {
          // grab vertex
          const Vector3<T>& pos = mesh.vertices[i];
          const Vector3<T>& nml = mesh.normals[i];
          auto& outputp = outputMesh.vertices[i];
          outputp.setZero();
          auto& outputn = outputMesh.normals[i];
          outputn.setZero();

          // loop over the weights
          for (size_t j = 0; j < kMaxSkinJoints; j++) {
            // get pointer to transformation and weight float
            const auto& weight = skin.weight(i, j);
            if (weight == 0.0f) {
              break;
            }

            MT_CHECK(
                skin.index(i, j) < skinningTransforms.size(),
                "skin.index({}, {}): {} vs {}",
                i,
                j,
                skin.index(i, j),
                skinningTransforms.size());
            const auto& transformation = skinningTransforms[skin.index(i, j)];

            // add up transforms: outputp += (transformation * (pos, 1)) * weight
            const auto& topLeft = transformation.template topLeftCorner<3, 3>();
            Eigen::Vector3<T> temp = transformation.template topRightCorner<3, 1>();
            temp.noalias() += topLeft * pos;
            outputp.noalias() += temp * weight;

            // add up normals
            outputn.noalias() += topLeft * nml * weight;
          }
          outputn.normalize();
        }
      },
      options);
}

template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    typename DeduceSpanType<const JointStateT<T>>::type jointState,
    MeshT<T>& outputMesh) {
  const auto skinningTransforms = computeSkinningTransforms(jointState, inverseBindPose);
  applySSD(skin, mesh, skinningTransforms, outputMesh);
}

template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    const SkeletonStateT<T>& state,
    MeshT<T>& outputMesh) {
  applySSD(inverseBindPose, skin, mesh, state.jointState, outputMesh);
}

template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    const TransformationListT<T>& worldTransforms,
    MeshT<T>& outputMesh) {
  MT_CHECK(
      worldTransforms.size() == inverseBindPose.size(),
      "{} vs {}",
      worldTransforms.size(),
      inverseBindPose.size());

  std::vector<Eigen::Matrix4<T>> skinningTransforms(inverseBindPose.size());
  for (size_t i = 0; i < inverseBindPose.size(); i++) {
    skinningTransforms[i].noalias() = (worldTransforms[i] * inverseBindPose[i]).matrix();
  }
  applySSD(skin, mesh, skinningTransforms, outputMesh);
}

template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    typename DeduceSpanType<const TransformT<T>>::type worldTransforms,
    MeshT<T>& outputMesh) {
  MT_CHECK(
      worldTransforms.size() == inverseBindPose.size(),
      "{} vs {}",
      worldTransforms.size(),
      inverseBindPose.size());

  std::vector<Eigen::Matrix4<T>> skinningTransforms(inverseBindPose.size());
  for (size_t i = 0; i < inverseBindPose.size(); i++) {
    skinningTransforms[i].noalias() = (worldTransforms[i] * inverseBindPose[i]).matrix();
  }
  applySSD(skin, mesh, skinningTransforms, outputMesh);
}

Affine3f getInverseSSDTransformation(
    const TransformationList& inverseBindPose,
    const SkinWeights& skin,
    const SkeletonState& state,
    const size_t index) {
  MT_CHECK(
      state.jointState.size() == inverseBindPose.size(),
      "{} vs {}",
      state.jointState.size(),
      inverseBindPose.size());
  MT_CHECK(
      gsl::narrow_cast<Eigen::Index>(index) < skin.index.rows(),
      "{} vs {}",
      index,
      skin.index.rows());

  Affine3f transform;
  transform.matrix().setZero();

  for (size_t j = 0; j < kMaxSkinJoints; j++) {
    const auto& weight = skin.weight(index, j);
    if (weight == 0.0f) {
      break;
    }

    auto jointIndex = skin.index(index, j);
    MT_CHECK(
        jointIndex < inverseBindPose.size(),
        "skin.index({}, {}): {} vs {}",
        index,
        j,
        jointIndex,
        inverseBindPose.size());
    const auto transformation =
        state.jointState[jointIndex].transform * inverseBindPose[jointIndex];

    transform.matrix().noalias() += transformation.matrix() * weight;
  }

  return transform.inverse();
}

void applyInverseSSD(
    const TransformationList& inverseBindPose,
    const SkinWeights& skin,
    std::span<const Vector3f> points,
    const SkeletonState& state,
    Mesh& mesh) {
  MT_CHECK(points.size() == mesh.vertices.size(), "{} vs {}", points.size(), mesh.vertices.size());

  mesh.vertices = applyInverseSSD(inverseBindPose, skin, points, state);
}

std::vector<Vector3f> applyInverseSSD(
    const TransformationList& inverseBindPose,
    const SkinWeights& skin,
    std::span<const Vector3f> points,
    const SkeletonState& state) {
  MT_CHECK(
      state.jointState.size() == inverseBindPose.size(),
      "{} vs {}",
      state.jointState.size(),
      inverseBindPose.size());
  MT_CHECK(
      skin.index.rows() == gsl::narrow<int>(points.size()),
      "{} vs {}",
      skin.index.rows(),
      points.size());
  MT_CHECK(
      skin.weight.rows() == gsl::narrow<int>(points.size()),
      "{} vs {}",
      skin.weight.rows(),
      points.size());

  std::vector<Vector3f> res(points.size());

  TransformationList transformations(state.jointState.size());
  for (size_t i = 0; i < state.jointState.size(); i++) {
    transformations[i] = state.jointState[i].transform * inverseBindPose[i];
  }

  for (int i = 0; i != (int)skin.index.rows(); i++) {
    const Vector3f& pos = points[i];

    Affine3f transform;
    transform.matrix().setZero();

    for (size_t j = 0; j < kMaxSkinJoints; j++) {
      const auto& weight = skin.weight(i, j);
      if (weight == 0.0f) {
        break;
      }

      MT_CHECK(
          skin.index(i, j) < transformations.size(),
          "skin.index({}, {}): {} vs {}",
          i,
          j,
          skin.index(i, j),
          transformations.size());
      const auto& transformation = transformations[skin.index(i, j)];

      transform.matrix().noalias() += transformation.matrix() * weight;
    }

    res[i].noalias() = transform.inverse() * pos;
  }

  return res;
}

// Explicit template instantiations for computeSkinningTransforms
template std::vector<Eigen::Matrix4f> computeSkinningTransforms<float>(
    std::span<const JointStateT<float>> jointState,
    const TransformationListT<float>& inverseBindPose);
template std::vector<Eigen::Matrix4d> computeSkinningTransforms<double>(
    std::span<const JointStateT<double>> jointState,
    const TransformationListT<double>& inverseBindPose);

// Explicit template instantiations for applySSD with skinning transforms (points)
template std::vector<Vector3f> applySSD<float>(
    const SkinWeights& skin,
    std::span<const Vector3f> points,
    std::span<const Eigen::Matrix4f> skinningTransforms);
template std::vector<Vector3d> applySSD<double>(
    const SkinWeights& skin,
    std::span<const Vector3d> points,
    std::span<const Eigen::Matrix4d> skinningTransforms);

// Explicit template instantiations for applySSD with joint state span (points)
template std::vector<Vector3f> applySSD<float>(
    const TransformationListT<float>& inverseBindPose,
    const SkinWeights& skin,
    std::span<const Vector3f> points,
    std::span<const JointStateT<float>> jointState);
template std::vector<Vector3d> applySSD<double>(
    const TransformationListT<double>& inverseBindPose,
    const SkinWeights& skin,
    std::span<const Vector3d> points,
    std::span<const JointStateT<double>> jointState);

// Explicit template instantiations for applySSD with skeleton state (points)
template std::vector<Vector3f> applySSD<float>(
    const TransformationListT<float>& inverseBindPose,
    const SkinWeights& skin,
    std::span<const Vector3f> points,
    const SkeletonStateT<float>& state);
template std::vector<Vector3d> applySSD<double>(
    const TransformationListT<double>& inverseBindPose,
    const SkinWeights& skin,
    std::span<const Vector3d> points,
    const SkeletonStateT<double>& state);

// Explicit template instantiations for applySSD with world transforms vector (points)
template std::vector<Vector3f> applySSD<float>(
    const TransformationListT<float>& inverseBindPose,
    const SkinWeights& skin,
    std::span<const Vector3f> points,
    const TransformationListT<float>& worldTransforms);
template std::vector<Vector3d> applySSD<double>(
    const TransformationListT<double>& inverseBindPose,
    const SkinWeights& skin,
    std::span<const Vector3d> points,
    const TransformationListT<double>& worldTransforms);

// Explicit template instantiations for applySSD with world transforms span (points)
template std::vector<Vector3f> applySSD<float>(
    const TransformationListT<float>& inverseBindPose,
    const SkinWeights& skin,
    std::span<const Vector3f> points,
    std::span<const TransformT<float>> worldTransforms);
template std::vector<Vector3d> applySSD<double>(
    const TransformationListT<double>& inverseBindPose,
    const SkinWeights& skin,
    std::span<const Vector3d> points,
    std::span<const TransformT<double>> worldTransforms);

// Explicit template instantiations for applySSD with skinning transforms (mesh)
template void applySSD<float>(
    const SkinWeights& skin,
    const MeshT<float>& mesh,
    std::span<const Eigen::Matrix4f> skinningTransforms,
    MeshT<float>& outputMesh);
template void applySSD<double>(
    const SkinWeights& skin,
    const MeshT<double>& mesh,
    std::span<const Eigen::Matrix4d> skinningTransforms,
    MeshT<double>& outputMesh);

// Explicit template instantiations for applySSD with joint state span (mesh)
template void applySSD<float>(
    const TransformationListT<float>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<float>& mesh,
    std::span<const JointStateT<float>> jointState,
    MeshT<float>& outputMesh);
template void applySSD<double>(
    const TransformationListT<double>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<double>& mesh,
    std::span<const JointStateT<double>> jointState,
    MeshT<double>& outputMesh);

// Explicit template instantiations for applySSD with skeleton state (mesh)
template void applySSD<float>(
    const TransformationListT<float>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<float>& mesh,
    const SkeletonStateT<float>& state,
    MeshT<float>& outputMesh);
template void applySSD<double>(
    const TransformationListT<double>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<double>& mesh,
    const SkeletonStateT<double>& state,
    MeshT<double>& outputMesh);

// Explicit template instantiations for applySSD with world transforms vector (mesh)
template void applySSD<float>(
    const TransformationListT<float>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<float>& mesh,
    const TransformationListT<float>& worldTransforms,
    MeshT<float>& outputMesh);
template void applySSD<double>(
    const TransformationListT<double>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<double>& mesh,
    const TransformationListT<double>& worldTransforms,
    MeshT<double>& outputMesh);

// Explicit template instantiations for applySSD with world transforms span (mesh)
template void applySSD<float>(
    const TransformationListT<float>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<float>& mesh,
    std::span<const TransformT<float>> worldTransforms,
    MeshT<float>& outputMesh);
template void applySSD<double>(
    const TransformationListT<double>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<double>& mesh,
    std::span<const TransformT<double>> worldTransforms,
    MeshT<double>& outputMesh);

template <typename T>
Vector3<T> getSkinnedLocatorPosition(
    const SkinnedLocator& locator,
    const Vector3<T>& restPosition,
    const TransformationList& inverseBindPose,
    const SkeletonStateT<T>& state) {
  Vector3<T> worldPos = Vector3<T>::Zero();
  T weightSum = 0;
  for (int k = 0; k < locator.skinWeights.size(); ++k) {
    const auto weight = static_cast<T>(locator.skinWeights[k]);
    if (weight == 0) {
      break;
    }
    const auto boneIndex = locator.parents[k];
    const auto& jointState = state.jointState[boneIndex];

    worldPos +=
        weight * (jointState.transform * (inverseBindPose[boneIndex].cast<T>() * restPosition));
    weightSum += weight;
  }

  return worldPos / weightSum;
}

template Vector3<float> getSkinnedLocatorPosition<float>(
    const SkinnedLocator& locator,
    const Vector3<float>& restPosition,
    const TransformationList& inverseBindPose,
    const SkeletonStateT<float>& state);
template Vector3<double> getSkinnedLocatorPosition<double>(
    const SkinnedLocator& locator,
    const Vector3<double>& restPosition,
    const TransformationList& inverseBindPose,
    const SkeletonStateT<double>& state);

template <typename T>
Vector3<T> getSkinnedLocatorPosition(
    const SkinnedLocator& locator,
    const TransformationList& inverseBindPose,
    const SkeletonStateT<T>& state) {
  const Vector3<T> restPosition = locator.position.template cast<T>();
  return getSkinnedLocatorPosition(locator, restPosition, inverseBindPose, state);
}

template Vector3<float> getSkinnedLocatorPosition<float>(
    const SkinnedLocator& locator,
    const TransformationList& inverseBindPose,
    const SkeletonStateT<float>& state);
template Vector3<double> getSkinnedLocatorPosition<double>(
    const SkinnedLocator& locator,
    const TransformationList& inverseBindPose,
    const SkeletonStateT<double>& state);

} // namespace momentum
