/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/usd/usd_mesh_io.h"

#include "momentum/common/checks.h"
#include "momentum/common/exception.h"

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdSkel/bindingAPI.h>

#include <algorithm>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace momentum {

namespace {

std::vector<Vector3f> loadVerticesFromMeshPrim(const UsdGeomMesh& meshPrim) {
  std::vector<Vector3f> vertices;
  VtArray<GfVec3f> points;
  if (meshPrim.GetPointsAttr().Get(&points)) {
    vertices.reserve(points.size());
    for (const auto& point : points) {
      vertices.emplace_back(point[0], point[1], point[2]);
    }
  }
  return vertices;
}

std::vector<Vector3b> loadColorsFromMeshPrim(const UsdGeomMesh& meshPrim, size_t numVertices) {
  std::vector<Vector3b> colors;
  UsdGeomPrimvarsAPI primvarsAPI(meshPrim);

  const std::vector<std::string> colorPrimvarNames = {
      "displayColor", "Cd", "color", "vertexColor", "diffuseColor"};

  for (const auto& colorName : colorPrimvarNames) {
    UsdGeomPrimvar colorPrimvar = primvarsAPI.GetPrimvar(TfToken(colorName));
    if (!colorPrimvar || !colorPrimvar.HasValue()) {
      continue;
    }

    VtValue colorValue;
    if (!colorPrimvar.Get(&colorValue)) {
      continue;
    }

    if (colorValue.IsHolding<VtArray<GfVec3f>>()) {
      VtArray<GfVec3f> rawColors = colorValue.Get<VtArray<GfVec3f>>();
      if (rawColors.size() == numVertices) {
        colors.reserve(rawColors.size());
        for (const auto& color : rawColors) {
          colors.emplace_back(color[0], color[1], color[2]);
        }
        break;
      }
    } else if (colorValue.IsHolding<VtArray<GfVec4f>>()) {
      VtArray<GfVec4f> rawColors = colorValue.Get<VtArray<GfVec4f>>();
      if (rawColors.size() == numVertices) {
        colors.reserve(rawColors.size());
        for (const auto& color : rawColors) {
          colors.emplace_back(color[0], color[1], color[2]);
        }
        break;
      }
    }
  }

  return colors;
}

std::vector<Vector3i> loadFacesFromMeshPrim(const UsdGeomMesh& meshPrim, size_t numPoints) {
  std::vector<Vector3i> faces;
  VtArray<int> faceVertexCounts;
  VtArray<int> faceVertexIndices;

  if (!meshPrim.GetFaceVertexCountsAttr().Get(&faceVertexCounts) ||
      !meshPrim.GetFaceVertexIndicesAttr().Get(&faceVertexIndices)) {
    return faces;
  }

  // Validate that faceVertexCounts and faceVertexIndices are consistent
  // (e.g., sum of counts == indices length) before accessing indices.
  std::string validationReason;
  MT_THROW_IF(
      !UsdGeomMesh::ValidateTopology(
          faceVertexIndices, faceVertexCounts, numPoints, &validationReason),
      "Invalid mesh topology: {}",
      validationReason);

  const bool allTriangles = std::all_of(
      faceVertexCounts.cbegin(), faceVertexCounts.cend(), [](int count) { return count == 3; });

  if (allTriangles && faceVertexIndices.size() == faceVertexCounts.size() * 3) {
    faces.reserve(faceVertexCounts.size());
    const int* indices = faceVertexIndices.cdata();
    for (size_t i = 0; i < faceVertexCounts.size(); ++i) {
      faces.emplace_back(indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]);
    }
    return faces;
  }

  size_t indexOffset = 0;
  for (int faceVertexCount : faceVertexCounts) {
    if (faceVertexCount == 3) {
      faces.emplace_back(
          faceVertexIndices[indexOffset],
          faceVertexIndices[indexOffset + 1],
          faceVertexIndices[indexOffset + 2]);
    } else if (faceVertexCount == 4) {
      faces.emplace_back(
          faceVertexIndices[indexOffset],
          faceVertexIndices[indexOffset + 1],
          faceVertexIndices[indexOffset + 2]);
      faces.emplace_back(
          faceVertexIndices[indexOffset],
          faceVertexIndices[indexOffset + 2],
          faceVertexIndices[indexOffset + 3]);
    }
    indexOffset += faceVertexCount;
  }

  return faces;
}

} // namespace

Mesh loadMeshFromUsd(const UsdStageRefPtr& stage) {
  Mesh mesh;

  for (const auto& prim : stage->Traverse()) {
    if (!prim.IsA<UsdGeomMesh>()) {
      continue;
    }

    UsdGeomMesh meshPrim(prim);
    mesh.vertices = loadVerticesFromMeshPrim(meshPrim);
    mesh.colors = loadColorsFromMeshPrim(meshPrim, mesh.vertices.size());
    mesh.faces = loadFacesFromMeshPrim(meshPrim, mesh.vertices.size());
    break; // Use first mesh found
  }

  return mesh;
}

SkinWeights loadSkinWeightsFromUsd(const UsdStageRefPtr& stage, size_t numVertices) {
  SkinWeights skinWeights;

  skinWeights.index.resize(numVertices, kMaxSkinJoints);
  skinWeights.weight.resize(numVertices, kMaxSkinJoints);
  skinWeights.index.setZero();
  skinWeights.weight.setZero();

  for (const auto& prim : stage->Traverse()) {
    if (!prim.IsA<UsdGeomMesh>()) {
      continue;
    }

    UsdGeomMesh meshPrim(prim);
    UsdSkelBindingAPI bindingAPI(meshPrim.GetPrim());
    if (!bindingAPI) {
      continue;
    }

    VtArray<int> jointIndices;
    VtArray<float> jointWeights;

    if (bindingAPI.GetJointIndicesAttr().Get(&jointIndices) &&
        bindingAPI.GetJointWeightsAttr().Get(&jointWeights)) {
      if (!jointIndices.empty() && !jointWeights.empty() &&
          jointIndices.size() == jointWeights.size()) {
        const int influencesPerVertex = jointIndices.size() / numVertices;

        if (influencesPerVertex > 0 && jointIndices.size() == numVertices * influencesPerVertex) {
          for (size_t v = 0; v < numVertices; ++v) {
            int validInfluences = 0;
            for (int i = 0;
                 i < influencesPerVertex && validInfluences < static_cast<int>(kMaxSkinJoints);
                 ++i) {
              size_t idx = v * influencesPerVertex + i;
              int jointIndex = jointIndices[idx];
              float weight = jointWeights[idx];

              if (weight > 0.0f && jointIndex >= 0) {
                skinWeights.index(v, validInfluences) = static_cast<uint32_t>(jointIndex);
                skinWeights.weight(v, validInfluences) = weight;
                validInfluences++;
              }
            }
          }
        }
      }
    }

    break; // Use first skinned mesh found
  }

  return skinWeights;
}

void saveMeshToUsd(const Mesh& mesh, UsdGeomMesh& meshPrim) {
  // Write vertices
  VtArray<GfVec3f> points;
  points.reserve(mesh.vertices.size());
  for (const auto& vertex : mesh.vertices) {
    points.push_back(GfVec3f(vertex.x(), vertex.y(), vertex.z()));
  }
  meshPrim.GetPointsAttr().Set(points);

  // Write faces
  VtArray<int> faceVertexCounts;
  VtArray<int> faceVertexIndices;
  faceVertexCounts.reserve(mesh.faces.size());
  faceVertexIndices.reserve(mesh.faces.size() * 3);

  for (const auto& face : mesh.faces) {
    faceVertexCounts.push_back(3);
    faceVertexIndices.push_back(face.x());
    faceVertexIndices.push_back(face.y());
    faceVertexIndices.push_back(face.z());
  }

  meshPrim.GetFaceVertexCountsAttr().Set(faceVertexCounts);
  meshPrim.GetFaceVertexIndicesAttr().Set(faceVertexIndices);
}

void saveSkinWeightsToUsd(
    const SkinWeights& skinWeights,
    UsdGeomMesh& meshPrim,
    const UsdSkelSkeleton& skelPrim) {
  if (skinWeights.index.rows() == 0) {
    return;
  }

  UsdSkelBindingAPI bindingAPI = UsdSkelBindingAPI::Apply(meshPrim.GetPrim());
  bindingAPI.GetSkeletonRel().SetTargets({skelPrim.GetPath()});

  VtArray<int> jointIndices;
  VtArray<float> jointWeights;

  const int maxInfluences = 4;
  const int numVertices = skinWeights.index.rows();
  const int numJointsPerVertex =
      std::min(maxInfluences, static_cast<int>(skinWeights.index.cols()));

  jointIndices.reserve(numVertices * maxInfluences);
  jointWeights.reserve(numVertices * maxInfluences);

  for (int v = 0; v < numVertices; ++v) {
    for (int i = 0; i < numJointsPerVertex; ++i) {
      jointIndices.push_back(skinWeights.index(v, i));
      jointWeights.push_back(skinWeights.weight(v, i));
    }

    for (int i = numJointsPerVertex; i < maxInfluences; ++i) {
      jointIndices.push_back(0);
      jointWeights.push_back(0.0f);
    }
  }

  bindingAPI.GetJointIndicesAttr().Set(jointIndices);
  bindingAPI.GetJointWeightsAttr().Set(jointWeights);
}

} // namespace momentum
