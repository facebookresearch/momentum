/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/linear_skinning.h>

#include <momentum/character/skeleton_state.h>
#include <momentum/character/skin_weights.h>
#include <momentum/common/checks.h>
#include <momentum/common/profile.h>
#include <momentum/math/mesh.h>

#include <dispenso/parallel_for.h>

namespace momentum {

template <typename T>
std::vector<Vector3<T>> applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<T>>::type points,
    const SkeletonStateT<T>& state) {
  MT_PROFILE_FUNCTION();

  // some sanity checks
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

  std::vector<Vector3<T>> result(points.size(), Vector3<T>::Zero());

  // first create a list of transformations from bindpose to final output pose
  std::vector<Matrix4<T>> transformations(state.jointState.size());
  for (size_t i = 0; i < state.jointState.size(); i++) {
    transformations[i] = (state.jointState[i].transformation * inverseBindPose[i]).matrix();
  }

  // go over all vertices and perform transformation
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
            MT_CHECK(
                skin.index(i, j) < transformations.size(),
                "skin.index({}, {}): {} vs {}",
                i,
                j,
                skin.index(i, j),
                transformations.size());

            // get pointer to transformation and weight float
            const auto& weight = skin.weight(i, j);
            if (weight == 0.0f) {
              break;
            }
            const auto& transformation = transformations[skin.index(i, j)];

            // add up transforms: outputp += (transformation * (pos, 1)) * weight
            Eigen::Vector3<T> temp = transformation.template topRightCorner<3, 1>();
            temp.noalias() += transformation.template topLeftCorner<3, 3>() * pos;
            output.noalias() += temp * weight;
          }
        }
      });

  return result;
}

template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    const SkeletonStateT<T>& state,
    MeshT<T>& outputMesh) {
  return applySSD(inverseBindPose, skin, mesh, state.jointState, outputMesh);
}

template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    const JointStateListT<T>& jointState,
    MeshT<T>& outputMesh) {
  MT_PROFILE_FUNCTION();

  // some sanity checks
  MT_CHECK(
      jointState.size() >= inverseBindPose.size(),
      "{} vs {}",
      jointState.size(),
      inverseBindPose.size());
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

  // first create a list of transformations from bindpose to final output pose
  std::vector<Eigen::Matrix4<T>> transformations(inverseBindPose.size());
  for (size_t i = 0; i < inverseBindPose.size(); i++) {
    transformations[i].noalias() = (jointState[i].transformation * inverseBindPose[i]).matrix();
  }

  // go over all vertices and perform transformation
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
            MT_CHECK(
                skin.index(i, j) < transformations.size(),
                "skin.index({}, {}): {} vs {}",
                i,
                j,
                skin.index(i, j),
                transformations.size());

            // get pointer to transformation and weight float
            const auto& weight = skin.weight(i, j);
            if (weight == 0.0f)
              break;
            const auto& transformation = transformations[skin.index(i, j)];

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
      });
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

  // storage
  Affine3f transform;
  transform.matrix().setZero();

  // loop over the weights
  for (size_t j = 0; j < kMaxSkinJoints; j++) {
    // get pointer to transformation and weight float
    const auto& weight = skin.weight(index, j);
    if (weight == 0.0f)
      break;

    auto jointIndex = skin.index(index, j);
    const auto transformation =
        state.jointState[jointIndex].transformation * inverseBindPose[jointIndex];

    // add up transforms
    transform.matrix().noalias() += transformation.matrix() * weight;
  }

  return transform.inverse();
}

void applyInverseSSD(
    const TransformationList& inverseBindPose,
    const SkinWeights& skin,
    gsl::span<const Vector3f> points,
    const SkeletonState& state,
    Mesh& mesh) {
  // some sanity checks
  MT_CHECK(points.size() == mesh.vertices.size(), "{} vs {}", points.size(), mesh.vertices.size());

  mesh.vertices = applyInverseSSD(inverseBindPose, skin, points, state);
}

std::vector<Vector3f> applyInverseSSD(
    const TransformationList& inverseBindPose,
    const SkinWeights& skin,
    gsl::span<const Vector3f> points,
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

  // first create a list of transformations from bindpose to final output pose
  TransformationList transformations(state.jointState.size());
  for (size_t i = 0; i < state.jointState.size(); i++) {
    transformations[i] = state.jointState[i].transformation * inverseBindPose[i];
  }

  // go over all vertices and perform transformation
  for (int i = 0; i != (int)skin.index.rows(); i++) {
    // grab vertex
    const Vector3f& pos = points[i];

    // storage
    Affine3f transform;
    transform.matrix().setZero();

    // loop over the weights
    for (size_t j = 0; j < kMaxSkinJoints; j++) {
      MT_CHECK(
          skin.index(i, j) < transformations.size(),
          "skin.index({}, {}): {} vs {}",
          i,
          j,
          skin.index(i, j),
          transformations.size());

      // get pointer to transformation and weight float
      const auto& weight = skin.weight(i, j);
      if (weight == 0.0f)
        break;
      const auto& transformation = transformations[skin.index(i, j)];

      // add up transforms
      transform.matrix().noalias() += transformation.matrix() * weight;
    }

    // store in new mesh
    res[i].noalias() = transform.inverse() * pos;
  }

  return res;
}

template std::vector<Vector3f> applySSD<float>(
    const TransformationListT<float>& inverseBindPose,
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<float>>::type points,
    const SkeletonStateT<float>& state);
template std::vector<Vector3d> applySSD<double>(
    const TransformationListT<double>& inverseBindPose,
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<double>>::type points,
    const SkeletonStateT<double>& state);

template void applySSD<float>(
    const TransformationList& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<float>& mesh,
    const SkeletonStateT<float>& jointState,
    MeshT<float>& outputMesh);
template void applySSD<double>(
    const TransformationListT<double>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<double>& mesh,
    const SkeletonStateT<double>& jointState,
    MeshT<double>& outputMesh);

template void applySSD<float>(
    const TransformationListT<float>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<float>& mesh,
    const JointStateListT<float>& jointState,
    MeshT<float>& outputMesh);
template void applySSD<double>(
    const TransformationListT<double>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<double>& mesh,
    const JointStateListT<double>& jointState,
    MeshT<double>& outputMesh);

} // namespace momentum
