/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/mesh.h"

#include "momentum/math/utility.h"

#include <algorithm>

namespace momentum {

template <typename T>
void MeshT<T>::updateNormals() {
  normals.resize(vertices.size());
  std::fill(normals.begin(), normals.end(), Eigen::Vector3<T>::Zero());
  const auto verticesNum = static_cast<int>(vertices.size());
  for (const auto& face : faces) {
    // Skip faces with out-of-bounds vertex indices.
    if (std::any_of(face.begin(), face.end(), [verticesNum](int idx) {
          return idx < 0 || idx >= verticesNum;
        })) {
      continue;
    }
    // The unnormalized cross product has magnitude equal to 2 * triangle area, so accumulating it
    // across faces produces area-weighted vertex normals (larger faces contribute more). This is
    // the standard Gouraud-style averaging and tends to produce better results than uniform
    // averaging for meshes with mixed triangle sizes.
    const Eigen::Vector3<T> normal =
        (vertices[face[1]] - vertices[face[0]]).cross(vertices[face[2]] - vertices[face[0]]);
    // Skip degenerate faces that produced NaN (e.g., from non-finite vertex coordinates).
    // Collinear/zero-area triangles produce a zero vector, which contributes nothing and is safe.
    // TODO: only checking normal[0] misses NaNs that appear only in y or z components.
    if (IsNanNoOpt(normal[0])) {
      continue;
    }
    for (const auto& faceIdx : face) {
      normals[faceIdx] += normal;
    }
  }
  for (size_t i = 0; i < normals.size(); i++) {
    const T len = normals[i].norm();
    // Leave isolated/unreferenced vertices with a zero normal rather than dividing by zero.
    if (len != 0) {
      normals[i] /= len;
    }
  }
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
  result.polyFaces = this->polyFaces;
  result.polyFaceSizes = this->polyFaceSizes;
  result.polyTexcoordFaces = this->polyTexcoordFaces;
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
  polyFaces.clear();
  polyFaceSizes.clear();
  polyTexcoordFaces.clear();
}

template MeshT<float> MeshT<float>::cast<float>() const;
template MeshT<double> MeshT<float>::cast<double>() const;
template MeshT<float> MeshT<double>::cast<float>() const;
template MeshT<double> MeshT<double>::cast<double>() const;

template struct MeshT<float>;
template struct MeshT<double>;

} // namespace momentum
