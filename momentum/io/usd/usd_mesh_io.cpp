/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/usd/usd_mesh_io.h"

#include "momentum/common/checks.h"
#include "momentum/common/exception.h"
#include "momentum/common/log.h"

#include <pxr/base/tf/staticTokens.h>

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/sdf/valueTypeName.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdSkel/bindingAPI.h>
#include <pxr/usd/usdSkel/blendShape.h>

#include <algorithm>
#include <array>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

TF_DEFINE_PRIVATE_TOKENS(
    _tokens,
    (displayColor)(Cd)(color)(vertexColor)(diffuseColor)((
        momentumConfidence,
        "momentum:confidence")));

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

Vector3b convertColorToBytes(float r, float g, float b) {
  // Detect 0-1 float range vs 0-255 byte range
  const bool isNormalized = (r <= 1.0f && g <= 1.0f && b <= 1.0f);
  const float scale = isNormalized ? 255.0f : 1.0f;
  return {
      static_cast<uint8_t>(std::clamp(r * scale, 0.0f, 255.0f)),
      static_cast<uint8_t>(std::clamp(g * scale, 0.0f, 255.0f)),
      static_cast<uint8_t>(std::clamp(b * scale, 0.0f, 255.0f))};
}

std::vector<Vector3b> loadColorsFromMeshPrim(const UsdGeomMesh& meshPrim, size_t numVertices) {
  std::vector<Vector3b> colors;
  UsdGeomPrimvarsAPI primvarsAPI(meshPrim);

  const std::array<TfToken, 5> colorPrimvarNames = {
      _tokens->displayColor,
      _tokens->Cd,
      _tokens->color,
      _tokens->vertexColor,
      _tokens->diffuseColor};

  for (const auto& colorToken : colorPrimvarNames) {
    UsdGeomPrimvar colorPrimvar = primvarsAPI.GetPrimvar(colorToken);
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
          colors.push_back(convertColorToBytes(color[0], color[1], color[2]));
        }
        break;
      }
    } else if (colorValue.IsHolding<VtArray<GfVec4f>>()) {
      VtArray<GfVec4f> rawColors = colorValue.Get<VtArray<GfVec4f>>();
      if (rawColors.size() == numVertices) {
        colors.reserve(rawColors.size());
        for (const auto& color : rawColors) {
          colors.push_back(convertColorToBytes(color[0], color[1], color[2]));
        }
        break;
      }
    }
  }

  return colors;
}

std::vector<Vector3f> loadNormalsFromMeshPrim(const UsdGeomMesh& meshPrim, size_t numVertices) {
  std::vector<Vector3f> normals;
  VtArray<GfVec3f> rawNormals;
  if (meshPrim.GetNormalsAttr().Get(&rawNormals) && rawNormals.size() == numVertices) {
    normals.reserve(rawNormals.size());
    for (const auto& normal : rawNormals) {
      normals.emplace_back(normal[0], normal[1], normal[2]);
    }
  }
  return normals;
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
    mesh.normals = loadNormalsFromMeshPrim(meshPrim, mesh.vertices.size());
    mesh.faces = loadFacesFromMeshPrim(meshPrim, mesh.vertices.size());

    // Load per-vertex confidence from custom primvar
    UsdGeomPrimvarsAPI primvarsAPI(meshPrim);
    UsdGeomPrimvar confidencePrimvar = primvarsAPI.GetPrimvar(_tokens->momentumConfidence);
    if (confidencePrimvar && confidencePrimvar.HasValue()) {
      VtArray<float> confidenceValues;
      if (confidencePrimvar.Get(&confidenceValues)) {
        if (confidenceValues.size() == mesh.vertices.size()) {
          mesh.confidence.assign(confidenceValues.begin(), confidenceValues.end());
        } else {
          MT_LOGW(
              "Confidence primvar size ({}) != vertex count ({}), skipping",
              confidenceValues.size(),
              mesh.vertices.size());
        }
      }
    }

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
        const int influencesPerVertex = static_cast<int>(jointIndices.size() / numVertices);

        if (influencesPerVertex > 0 && jointIndices.size() == numVertices * influencesPerVertex) {
          for (size_t v = 0; v < numVertices; ++v) {
            int validInfluences = 0;
            for (int i = 0;
                 i < influencesPerVertex && validInfluences < static_cast<int>(kMaxSkinJoints);
                 ++i) {
              const size_t idx = v * influencesPerVertex + i;
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

  // Write normals
  if (!mesh.normals.empty()) {
    VtArray<GfVec3f> normals;
    normals.reserve(mesh.normals.size());
    for (const auto& normal : mesh.normals) {
      normals.push_back(GfVec3f(normal.x(), normal.y(), normal.z()));
    }
    meshPrim.GetNormalsAttr().Set(normals);
    meshPrim.SetNormalsInterpolation(UsdGeomTokens->vertex);
  }

  // Write vertex colors
  if (!mesh.colors.empty()) {
    UsdGeomPrimvarsAPI primvarsAPI(meshPrim);
    auto colorPrimvar = primvarsAPI.CreatePrimvar(
        _tokens->displayColor, SdfValueTypeNames->Color3fArray, UsdGeomTokens->vertex);

    VtArray<GfVec3f> colors;
    colors.reserve(mesh.colors.size());
    for (const auto& color : mesh.colors) {
      colors.push_back(GfVec3f(color.x() / 255.0f, color.y() / 255.0f, color.z() / 255.0f));
    }
    colorPrimvar.Set(colors);
  }

  // Write per-vertex confidence
  if (!mesh.confidence.empty()) {
    UsdGeomPrimvarsAPI primvarsAPI(meshPrim);
    auto confidencePrimvar = primvarsAPI.CreatePrimvar(
        _tokens->momentumConfidence, SdfValueTypeNames->FloatArray, UsdGeomTokens->vertex);
    VtArray<float> confidenceValues(mesh.confidence.begin(), mesh.confidence.end());
    confidencePrimvar.Set(confidenceValues);
  }
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
  const int numVertices = static_cast<int>(skinWeights.index.rows());
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

void saveBlendShapesToUsd(const BlendShape& blendShape, UsdGeomMesh& meshPrim) {
  const auto numShapes = blendShape.shapeSize();
  const auto numVertices = blendShape.modelSize();
  if (numShapes == 0 || numVertices == 0) {
    return;
  }

  const auto& shapeVectors = blendShape.getShapeVectors();
  const auto& shapeNames = blendShape.getShapeNames();

  SdfPathVector blendShapePaths;
  blendShapePaths.reserve(numShapes);
  VtArray<TfToken> blendShapeTokens;
  blendShapeTokens.reserve(numShapes);

  auto stage = meshPrim.GetPrim().GetStage();
  const auto meshPath = meshPrim.GetPath();

  VtArray<GfVec3f> offsets;
  offsets.reserve(numVertices);

  for (Eigen::Index s = 0; s < numShapes; ++s) {
    const std::string name =
        (s < static_cast<Eigen::Index>(shapeNames.size()) && !shapeNames[s].empty())
        ? shapeNames[s]
        : "shape_" + std::to_string(s);

    auto shapePath = meshPath.AppendChild(TfToken(name));
    auto shapePrim = UsdSkelBlendShape::Define(stage, shapePath);

    // Extract per-vertex offsets from the shape vector column
    offsets.clear();
    for (size_t v = 0; v < numVertices; ++v) {
      offsets.push_back(GfVec3f(
          shapeVectors(3 * v + 0, s), shapeVectors(3 * v + 1, s), shapeVectors(3 * v + 2, s)));
    }
    shapePrim.GetOffsetsAttr().Set(offsets);

    blendShapePaths.push_back(shapePath);
    blendShapeTokens.push_back(TfToken(name));
  }

  // Bind blend shapes to the mesh
  UsdSkelBindingAPI bindingAPI = UsdSkelBindingAPI::Apply(meshPrim.GetPrim());
  bindingAPI.GetBlendShapesAttr().Set(blendShapeTokens);
  bindingAPI.GetBlendShapeTargetsRel().SetTargets(blendShapePaths);
}

std::shared_ptr<BlendShape> loadBlendShapesFromUsd(
    const UsdStageRefPtr& stage,
    size_t numVertices) {
  for (const auto& prim : stage->Traverse()) {
    if (!prim.IsA<UsdGeomMesh>()) {
      continue;
    }

    UsdSkelBindingAPI bindingAPI(prim);
    VtArray<TfToken> blendShapeTokens;
    if (!bindingAPI.GetBlendShapesAttr().Get(&blendShapeTokens) || blendShapeTokens.empty()) {
      continue;
    }

    SdfPathVector blendShapePaths;
    if (!bindingAPI.GetBlendShapeTargetsRel().GetTargets(&blendShapePaths) ||
        blendShapePaths.empty()) {
      continue;
    }

    const auto numShapes = static_cast<Eigen::Index>(blendShapePaths.size());

    // Collect shape names and offsets
    std::vector<std::string> shapeNames;
    shapeNames.reserve(numShapes);
    MatrixXf shapeVectors = MatrixXf::Zero(3 * numVertices, numShapes);

    VtArray<GfVec3f> offsets;
    for (Eigen::Index s = 0; s < numShapes; ++s) {
      shapeNames.push_back(blendShapePaths[s].GetName());

      auto shapePrim = UsdSkelBlendShape::Get(stage, blendShapePaths[s]);
      if (!shapePrim) {
        continue;
      }

      offsets.clear();
      if (shapePrim.GetOffsetsAttr().Get(&offsets)) {
        const size_t count = std::min(offsets.size(), numVertices);
        for (size_t v = 0; v < count; ++v) {
          shapeVectors(3 * v + 0, s) = offsets[v][0];
          shapeVectors(3 * v + 1, s) = offsets[v][1];
          shapeVectors(3 * v + 2, s) = offsets[v][2];
        }
      }
    }

    // Build base shape from mesh vertices
    UsdGeomMesh meshPrim(prim);
    VtArray<GfVec3f> points;
    if (!meshPrim.GetPointsAttr().Get(&points) || points.size() != numVertices) {
      continue;
    }
    std::vector<Vector3f> baseShape;
    baseShape.reserve(points.size());
    for (const auto& p : points) {
      baseShape.emplace_back(p[0], p[1], p[2]);
    }

    auto result = std::make_shared<BlendShape>(baseShape, numShapes, shapeNames);
    result->setShapeVectors(shapeVectors, shapeNames);

    return result;
  }

  return nullptr;
}

} // namespace momentum
