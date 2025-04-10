/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/mesh.h"

#include "momentum/common/profile.h"
#include "momentum/math/utility.h"

#include <dispenso/parallel_for.h>

#include <algorithm>

namespace momentum {

template <typename T>
void MeshT<T>::updateNormals() {
  // calculate normals
  normals.resize(vertices.size());
  std::fill(normals.begin(), normals.end(), Eigen::Vector3<T>::Zero());
  const auto verticesNum = static_cast<int>(vertices.size());
  for (const auto& face : faces) {
    // Skip faces with out-of-boundaries indexes
    if (std::any_of(face.begin(), face.end(), [verticesNum](int idx) {
          return idx < 0 || idx >= verticesNum;
        }))
      continue;
    // calculate normal and add for each vertex
    const Eigen::Vector3<T> normal =
        (vertices[face[1]] - vertices[face[0]]).cross(vertices[face[2]] - vertices[face[0]]);
    if (IsNanNoOpt(normal[0]))
      continue;
    for (const auto& faceIdx : face) {
      normals[faceIdx] += normal;
    }
  }
  // re-normalize normals
  for (auto& normal : normals) {
    normal.normalize(); // if the input vector is too small, the this is left unchanged
  }
}

template <typename T>
void MeshT<T>::updateNormalsMt(size_t maxThreads) {
  // fill vector for faces per vertex
  facePerVertex_.clear(); // need to clear previous data
  facePerVertex_.resize(vertices.size());
  for (size_t i = 0; i < faces.size(); i++) {
    const auto& face = faces[i];
    for (size_t j = 0; j < 3; j++) {
      facePerVertex_[face[j]].push_back(i);
    }
  }

  // resize face normals
  faceNormals_.resize(faces.size());

  // set options for parallel for
  auto dispensoOptions = dispenso::ParForOptions();
  dispensoOptions.maxThreads = maxThreads;

  // compute normals per face
  MT_PROFILE_PUSH("Compute face normals");
  const auto verticesNum = static_cast<int>(vertices.size());
  dispenso::parallel_for(
      0,
      faces.size(),
      [&](const size_t i) {
        // Skip faces with out-of-boundaries indexes
        const auto& face = faces[i];
        if (std::any_of(face.begin(), face.end(), [verticesNum](int idx) {
              return idx < 0 || idx >= verticesNum;
            })) {
          faceNormals_[i].setZero();
          return;
        }

        // calculate normal
        faceNormals_[i].noalias() =
            (vertices[face[1]] - vertices[face[0]]).cross(vertices[face[2]] - vertices[face[0]]);
      },
      dispensoOptions);
  MT_PROFILE_POP();

  // for each vertex, add the face normals
  MT_PROFILE_PUSH("add face normals");
  normals.resize(vertices.size());
  std::fill(normals.begin(), normals.end(), Eigen::Vector3<T>::Zero());
  dispenso::parallel_for(
      0,
      facePerVertex_.size(),
      [&](const size_t i) {
        for (const auto& faceIdx : facePerVertex_[i]) {
          if (IsNanNoOpt(faceNormals_[faceIdx][0])) {
            continue;
          }

          normals[i].noalias() += faceNormals_[faceIdx];
        }

        normals[i].normalize(); // if the input vector is too small, the this is left unchanged
      },
      dispensoOptions);
  MT_PROFILE_POP();
}

template <typename T, typename T2, int N>
std::vector<Eigen::Vector<T2, N>> castVectors(const std::vector<Eigen::Vector<T, N>>& vec_in) {
  std::vector<Eigen::Vector<T2, N>> result;
  result.reserve(vec_in.size());
  for (const auto& v : vec_in) {
    result.push_back(v.template cast<T2>());
  }
  return result;
}

template <typename T>
template <typename T2>
MeshT<T2> MeshT<T>::cast() const {
  MeshT<T2> result;
  result.vertices = castVectors<T, T2, 3>(this->vertices);
  result.normals = castVectors<T, T2, 3>(this->normals);
  result.faces = this->faces;
  result.lines = this->lines;
  result.colors = this->colors;
  result.confidence = std::vector<T2>(this->confidence.begin(), this->confidence.end());
  result.texcoords = this->texcoords;
  result.texcoord_faces = this->texcoord_faces;
  result.texcoord_lines = this->texcoord_lines;
  return result;
}

template <typename T>
void MeshT<T>::reset() {
  vertices.clear();
  normals.clear();
  faces.clear();
  lines.clear();
  colors.clear();
  confidence.clear();
  texcoords.clear();
  texcoord_faces.clear();
  texcoord_lines.clear();
  facePerVertex_.clear();
  faceNormals_.clear();
}

template MeshT<float> MeshT<float>::cast<float>() const;
template MeshT<double> MeshT<float>::cast<double>() const;
template MeshT<float> MeshT<double>::cast<float>() const;
template MeshT<double> MeshT<double>::cast<double>() const;

template struct MeshT<float>;
template struct MeshT<double>;

} // namespace momentum
