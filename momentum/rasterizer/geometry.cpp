/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <tuple>
#include <unordered_map>

#include <momentum/rasterizer/geometry.h>
#include <Eigen/Core>

#include <momentum/common/exception.h>

namespace momentum::rasterizer {

Mesh fromFlatVectors(
    const Eigen::VectorXf& positions,
    const Eigen::VectorXf& normals,
    const Eigen::VectorXi& triangles) {
  Mesh mesh;

  // Convert positions
  const size_t numVerts = positions.size() / 3;
  mesh.vertices.reserve(numVerts);
  for (size_t i = 0; i < numVerts; ++i) {
    mesh.vertices.emplace_back(positions.segment<3>(3 * i));
  }

  // Convert normals
  mesh.normals.reserve(numVerts);
  for (size_t i = 0; i < numVerts; ++i) {
    mesh.normals.emplace_back(normals.segment<3>(3 * i));
  }

  // Convert triangles
  const size_t numTris = triangles.size() / 3;
  mesh.faces.reserve(numTris);
  for (size_t i = 0; i < numTris; ++i) {
    mesh.faces.emplace_back(triangles.segment<3>(3 * i));
  }

  return mesh;
}

Mesh mergeMeshes(const std::initializer_list<Mesh>& meshes) {
  Mesh result;

  size_t totalVerts = 0;
  size_t totalTris = 0;
  for (const auto& m : meshes) {
    totalVerts += m.vertices.size();
    totalTris += m.faces.size();
  }

  result.vertices.reserve(totalVerts);
  result.normals.reserve(totalVerts);
  result.faces.reserve(totalTris);

  for (const auto& m : meshes) {
    const size_t vertOffset = result.vertices.size();

    // Add vertices and normals
    result.vertices.insert(result.vertices.end(), m.vertices.begin(), m.vertices.end());
    result.normals.insert(result.normals.end(), m.normals.begin(), m.normals.end());

    // Add faces with offset
    for (const auto& face : m.faces) {
      result.faces.emplace_back(
          face.x() + vertOffset, face.y() + vertOffset, face.z() + vertOffset);
    }
  }

  return result;
}

Mesh makeSphere(int subdivisionLevel) {
  // Use subdivision level to determine resolution
  // Higher subdivision level = more subdivisions
  const int numAzimuthSubdivisions = std::max(8, 4 * (1 << subdivisionLevel));
  const int numPolarSubdivisions = std::max(4, 2 * (1 << subdivisionLevel));

  // Pre-cache azimuth (longitude) angles
  std::vector<Eigen::Vector2f> azimuthPoints;
  azimuthPoints.reserve(numAzimuthSubdivisions);
  for (int i = 0; i < numAzimuthSubdivisions; ++i) {
    const float angle = 2.0f * M_PI * ((float)i / (float)numAzimuthSubdivisions);
    azimuthPoints.emplace_back(std::cos(angle), std::sin(angle));
  }

  Mesh mesh;

  // Total vertices: 1 north pole + middle bands + 1 south pole
  const int numMiddleBands = numPolarSubdivisions - 1;
  mesh.vertices.reserve(2 + numMiddleBands * numAzimuthSubdivisions);
  mesh.normals.reserve(2 + numMiddleBands * numAzimuthSubdivisions);

  // Add north pole vertex (index 0)
  mesh.vertices.emplace_back(0.0f, 0.0f, 1.0f);
  mesh.normals.emplace_back(0.0f, 0.0f, 1.0f);

  // Add middle band vertices (indices 1 to numMiddleBands * numAzimuthSubdivisions)
  for (int iPolar = 1; iPolar < numPolarSubdivisions; ++iPolar) {
    const float polarAngle = M_PI * ((float)iPolar / (float)numPolarSubdivisions);
    const float cosPolar = std::cos(polarAngle);
    const float sinPolar = std::sin(polarAngle);

    for (int iAzimuth = 0; iAzimuth < numAzimuthSubdivisions; ++iAzimuth) {
      const auto cosAzimuth = azimuthPoints.at(iAzimuth).x();
      const auto sinAzimuth = azimuthPoints.at(iAzimuth).y();

      // Spherical coordinates: x = sin(polar) * cos(azimuth), y = sin(polar) * sin(azimuth), z =
      // cos(polar)
      Eigen::Vector3f pos(sinPolar * cosAzimuth, sinPolar * sinAzimuth, cosPolar);
      mesh.vertices.push_back(pos);
      mesh.normals.push_back(pos.normalized());
    }
  }

  // Add south pole vertex (last index)
  mesh.vertices.emplace_back(0.0f, 0.0f, -1.0f);
  mesh.normals.emplace_back(0.0f, 0.0f, -1.0f);

  const int northPoleIdx = 0;
  const int southPoleIdx = static_cast<int>(mesh.vertices.size() - 1);

  // Helper function to get vertex index in middle bands
  auto getMiddleVertexIdx = [&](int polarBand, int azimuthIdx) -> int {
    return 1 + polarBand * numAzimuthSubdivisions + azimuthIdx;
  };

  // Generate triangles connecting north pole to first ring
  for (int iAzimuth = 0; iAzimuth < numAzimuthSubdivisions; ++iAzimuth) {
    const int iAzimuthNext = (iAzimuth + 1) % numAzimuthSubdivisions;
    const int idx1 = getMiddleVertexIdx(0, iAzimuth);
    const int idx2 = getMiddleVertexIdx(0, iAzimuthNext);
    mesh.faces.emplace_back(northPoleIdx, idx1, idx2);
  }

  // Generate triangles for middle bands
  for (int iPolar = 0; iPolar < numMiddleBands - 1; ++iPolar) {
    for (int iAzimuth = 0; iAzimuth < numAzimuthSubdivisions; ++iAzimuth) {
      const int iAzimuthNext = (iAzimuth + 1) % numAzimuthSubdivisions;

      const int idx1 = getMiddleVertexIdx(iPolar, iAzimuth);
      const int idx2 = getMiddleVertexIdx(iPolar, iAzimuthNext);
      const int idx3 = getMiddleVertexIdx(iPolar + 1, iAzimuthNext);
      const int idx4 = getMiddleVertexIdx(iPolar + 1, iAzimuth);

      // Create two triangles for each quad with correct winding
      mesh.faces.emplace_back(idx1, idx3, idx2);
      mesh.faces.emplace_back(idx1, idx4, idx3);
    }
  }

  // Generate triangles connecting last ring to south pole
  const int lastBandIdx = numMiddleBands - 1;
  for (int iAzimuth = 0; iAzimuth < numAzimuthSubdivisions; ++iAzimuth) {
    const int iAzimuthNext = (iAzimuth + 1) % numAzimuthSubdivisions;
    const int idx1 = getMiddleVertexIdx(lastBandIdx, iAzimuth);
    const int idx2 = getMiddleVertexIdx(lastBandIdx, iAzimuthNext);
    mesh.faces.emplace_back(southPoleIdx, idx2, idx1);
  }

  return mesh;
}

// Builds a cap normal to the x axis
Mesh makeCylinderCap(int numCircleSubdivisions, bool top, float radius = 1.0) {
  Mesh result;

  // Reserve space for vertices, normals and faces
  result.vertices.reserve(1 + numCircleSubdivisions);
  result.normals.reserve(1 + numCircleSubdivisions);
  result.faces.reserve(numCircleSubdivisions);

  const float xValue = top ? 1.0f : 0.0f;
  const Eigen::Vector3f normal = (top ? 1.0f : -1.0f) * Eigen::Vector3f::UnitX();

  // Add center vertex
  result.vertices.emplace_back(xValue, 0, 0);
  result.normals.push_back(normal);

  // Add circle vertices
  for (int i = 0; i < numCircleSubdivisions; ++i) {
    const float angle = 2.0f * M_PI * float(i) / float(numCircleSubdivisions);
    result.vertices.emplace_back(xValue, radius * std::cos(angle), radius * std::sin(angle));
    result.normals.push_back(normal);
  }

  // Add triangles
  for (int i = 0; i < numCircleSubdivisions; ++i) {
    const int cur = i + 1;
    const int next = (i + 1) % numCircleSubdivisions + 1;
    result.faces.emplace_back(top ? Eigen::Vector3i(0, cur, next) : Eigen::Vector3i(0, next, cur));
  }

  return result;
}

// Builds an uncapped cylinder oriented along the x axis:
Mesh makeCylinderBody(
    int numCircleSubdivisions,
    int numLengthSubdivisions,
    float radius = 1.0,
    float length = 1.0) {
  // Pre-cache the sines and cosines:
  std::vector<Eigen::Vector2f> circlePoints;
  circlePoints.reserve(numCircleSubdivisions);
  for (int i = 0; i < numCircleSubdivisions; ++i) {
    const float angle = 2.0f * M_PI * ((float)i / (float)numCircleSubdivisions);
    circlePoints.emplace_back(std::cos(angle), std::sin(angle));
  }

  auto vertexIndex = [&](int iLength, int jRadius) {
    return iLength * numCircleSubdivisions + (jRadius % numCircleSubdivisions);
  };

  Mesh result;
  result.vertices.reserve((numLengthSubdivisions + 1) * numCircleSubdivisions);
  result.normals.reserve((numLengthSubdivisions + 1) * numCircleSubdivisions);
  result.faces.reserve(2 * numLengthSubdivisions * numCircleSubdivisions);

  // Build the vertices along the length:
  for (int iLength = 0; iLength <= numLengthSubdivisions; ++iLength) {
    float xPos = float(iLength) / float(numLengthSubdivisions) * length;
    for (int jRadius = 0; jRadius < numCircleSubdivisions; ++jRadius) {
      const Eigen::Vector2f circlePt = circlePoints.at(jRadius);
      result.vertices.emplace_back(xPos, radius * circlePt.x(), radius * circlePt.y());
      // normal points outward:
      result.normals.emplace_back(Eigen::Vector3f(0, circlePt.x(), circlePt.y()).normalized());
    }
  }

  // Build the triangles:
  for (int iLength = 0; iLength < numLengthSubdivisions; ++iLength) {
    for (int jRadius = 0; jRadius < numCircleSubdivisions; ++jRadius) {
      result.faces.emplace_back(
          vertexIndex(iLength, jRadius),
          vertexIndex(iLength, jRadius + 1),
          vertexIndex(iLength + 1, jRadius + 1));
      result.faces.emplace_back(
          vertexIndex(iLength, jRadius),
          vertexIndex(iLength + 1, jRadius + 1),
          vertexIndex(iLength + 1, jRadius));
    }
  }

  return result;
}

Mesh makeCylinder(int numCircleSubdivisions, int numLengthSubdivisions) {
  return mergeMeshes(
      {makeCylinderBody(numCircleSubdivisions, numLengthSubdivisions),
       makeCylinderCap(numCircleSubdivisions, true),
       makeCylinderCap(numCircleSubdivisions, false)});
}

// Builds a capsule with spherical end capped, oriented along the x axis:
Mesh makeCapsule(
    int numCircleSubdivisions,
    int numLengthSubdivisions,
    float startRadius,
    float endRadius,
    float length) {
  // Pre-cache the sines and cosines:
  std::vector<Eigen::Vector2f> circlePoints;
  circlePoints.reserve(numCircleSubdivisions);
  for (int i = 0; i < numCircleSubdivisions; ++i) {
    const float angle = 2.0f * M_PI * ((float)i / (float)numCircleSubdivisions);
    circlePoints.emplace_back(std::cos(angle), std::sin(angle));
  }

  const auto numPolarSubdivisions = numCircleSubdivisions / 2 + 1;
  std::vector<Eigen::Vector2f> polarPoints;
  polarPoints.reserve(numPolarSubdivisions);
  for (int i = 0; i < numPolarSubdivisions; ++i) {
    const float angle = M_PI * ((float)i / (float)numCircleSubdivisions);
    polarPoints.emplace_back(std::cos(angle), std::sin(angle));
  }

  Mesh result;

  // Reserve space for all vertices
  const size_t totalVertices = (numLengthSubdivisions + 1) * numCircleSubdivisions +
      2 * numCircleSubdivisions * numPolarSubdivisions;
  result.vertices.resize(totalVertices, Eigen::Vector3f::Zero());
  result.normals.resize(totalVertices, Eigen::Vector3f::Zero());

  // If I have my math correct, this is the offset of the normal along the x axis.
  // Explanation: theta is the angle subtended by the vector v along the surface of the
  // cylinder with the x axis, and because the normal n is perpendicular to v it must
  // have the same angle with the y axis.  Therefore theta = phi = atan(rEnd - rStart / length)
  // and the length of the normals x component is tan(theta) = rEnd - rStart / length.
  //       | y axis
  //       |           ____
  //   phi |     v ___/    |
  //   \   |    __/        | <- (rEnd - rStart)
  //    \  | __/  theta    |
  //     \ |/______________|__ x axis
  const float normalX = -(endRadius - startRadius) / length;

  // Build the main cylinder body vertices
  size_t vertexOffset = 0;
  auto vertexIndex = [&](int iLength, int jRadius) {
    return vertexOffset + iLength * numCircleSubdivisions + (jRadius % numCircleSubdivisions);
  };

  // Build the vertices along the length:
  for (int iLength = 0; iLength <= numLengthSubdivisions; ++iLength) {
    const float fraction = float(iLength) / float(numLengthSubdivisions);
    const float xPos = length * fraction;
    const float radius = std::lerp(startRadius, endRadius, fraction);
    for (int jRadius = 0; jRadius < numCircleSubdivisions; ++jRadius) {
      const Eigen::Vector2f& circlePt = circlePoints.at(jRadius);
      result.vertices[vertexIndex(iLength, jRadius)] =
          Eigen::Vector3f(xPos, radius * circlePt.x(), radius * circlePt.y());
      // normal points outward:
      result.normals[vertexIndex(iLength, jRadius)] =
          Eigen::Vector3f(normalX, circlePt.x(), circlePt.y()).normalized();
    }
  }

  // Build the triangles for cylinder body:
  for (int iLength = 0; iLength < numLengthSubdivisions; ++iLength) {
    for (int jRadius = 0; jRadius < numCircleSubdivisions; ++jRadius) {
      result.faces.emplace_back(
          vertexIndex(iLength, jRadius),
          vertexIndex(iLength, jRadius + 1),
          vertexIndex(iLength + 1, jRadius + 1));
      result.faces.emplace_back(
          vertexIndex(iLength, jRadius),
          vertexIndex(iLength + 1, jRadius + 1),
          vertexIndex(iLength + 1, jRadius));
    }
  }

  vertexOffset += (numLengthSubdivisions + 1) * numCircleSubdivisions;

  // Construct the lower cap:
  for (int iAzimuth = 0; iAzimuth < numCircleSubdivisions; ++iAzimuth) {
    for (int iPolar = 0; iPolar < numPolarSubdivisions; ++iPolar) {
      const auto cosAzimuth = circlePoints.at(iAzimuth).x();
      const auto sinAzimuth = circlePoints.at(iAzimuth).y();
      const auto cosPolar = polarPoints.at(iPolar).x();
      const auto sinPolar = polarPoints.at(iPolar).y();

      Eigen::Vector3f pos =
          Eigen::Vector3f(-cosPolar, sinPolar * cosAzimuth, sinPolar * sinAzimuth);

      result.vertices[vertexOffset + iPolar * numCircleSubdivisions + iAzimuth] = startRadius * pos;
      result.normals[vertexOffset + iPolar * numCircleSubdivisions + iAzimuth] = pos.normalized();
    }
  }

  for (int iAzimuth = 0; iAzimuth < numCircleSubdivisions; ++iAzimuth) {
    for (int jPolar = 0; jPolar < (numPolarSubdivisions - 1); ++jPolar) {
      const int iAzimuthNext = (iAzimuth + 1) % numCircleSubdivisions;
      const int jPolarNext = (jPolar + 1);

      int idx1 = vertexOffset + jPolar * numCircleSubdivisions + iAzimuth;
      int idx2 = vertexOffset + jPolar * numCircleSubdivisions + iAzimuthNext;
      int idx3 = vertexOffset + jPolarNext * numCircleSubdivisions + iAzimuthNext;
      int idx4 = vertexOffset + jPolarNext * numCircleSubdivisions + iAzimuth;

      if (jPolar != 0) {
        result.faces.emplace_back(idx1, idx2, idx3);
      }
      result.faces.emplace_back(idx1, idx3, idx4);
    }
  }

  vertexOffset += numCircleSubdivisions * numPolarSubdivisions;

  // Construct the upper cap:
  for (int iPolar = 0; iPolar < numPolarSubdivisions; ++iPolar) {
    for (int iAzimuth = 0; iAzimuth < numCircleSubdivisions; ++iAzimuth) {
      const auto cosAzimuth = circlePoints.at(iAzimuth).x();
      const auto sinAzimuth = circlePoints.at(iAzimuth).y();
      const auto cosPolar = polarPoints.at(iPolar).x();
      const auto sinPolar = polarPoints.at(iPolar).y();

      Eigen::Vector3f pos = Eigen::Vector3f(cosPolar, sinPolar * cosAzimuth, sinPolar * sinAzimuth);
      result.vertices[vertexOffset + iPolar * numCircleSubdivisions + iAzimuth] =
          Eigen::Vector3f(length, 0, 0) + endRadius * pos;
      result.normals[vertexOffset + iPolar * numCircleSubdivisions + iAzimuth] = pos.normalized();
      result.vertices.push_back(Eigen::Vector3f(length, 0, 0) + endRadius * pos);
      result.normals.push_back(pos.normalized());
    }
  }

  for (int jPolar = 0; jPolar < (numPolarSubdivisions - 1); ++jPolar) {
    for (int iAzimuth = 0; iAzimuth < numCircleSubdivisions; ++iAzimuth) {
      const int iAzimuthNext = (iAzimuth + 1) % numCircleSubdivisions;
      const int jPolarNext = jPolar + 1;

      int idx1 = vertexOffset + jPolar * numCircleSubdivisions + iAzimuth;
      int idx2 = vertexOffset + jPolar * numCircleSubdivisions + iAzimuthNext;
      int idx3 = vertexOffset + jPolarNext * numCircleSubdivisions + iAzimuthNext;
      int idx4 = vertexOffset + jPolarNext * numCircleSubdivisions + iAzimuth;

      if (jPolar != 0) {
        result.faces.emplace_back(idx3, idx2, idx1);
      }
      result.faces.emplace_back(idx4, idx3, idx1);
    }
  }

  return result;
}

// Builds an arrowhead pointing down the x axis:
Mesh makeArrowhead(
    int numCircleSubdivisions,
    float innerRadius,
    float outerRadius,
    float length,
    float translation) {
  // Pre-cache the sines and cosines:
  std::vector<Eigen::Vector2f> circlePoints;
  circlePoints.reserve(numCircleSubdivisions);
  for (int i = 0; i < numCircleSubdivisions; ++i) {
    const float angle = 2.0f * M_PI * ((float)i / (float)numCircleSubdivisions);
    circlePoints.emplace_back(std::cos(angle), std::sin(angle));
  }

  const float endRadius = 0.0f;
  const float startRadius = outerRadius;
  const size_t numLengthSubdivisions = 1;

  Mesh result;

  // Reserve space for vertices, normals and faces
  // Arrowhead body: (numLengthSubdivisions + 1) * numCircleSubdivisions
  // Cap: 2 * numCircleSubdivisions (inner and outer ring)
  const size_t totalVertices =
      (numLengthSubdivisions + 1) * numCircleSubdivisions + 2 * numCircleSubdivisions;
  result.vertices.reserve(totalVertices);
  result.normals.reserve(totalVertices);

  // Estimate triangles: body triangles + cap triangles
  const size_t totalTriangles =
      numLengthSubdivisions * numCircleSubdivisions + 2 * numCircleSubdivisions;
  result.faces.reserve(totalTriangles);

  size_t vertexOffset = 0;

  // If I have my math correct, this is the offset of the normal along the x axis.
  // Explanation: theta is the angle subtended by the vector v along the surface of the
  // cylinder with the x axis, and because the normal n is perpendicular to v it must
  // have the same angle with the y axis.  Therefore theta = phi = atan(rEnd - rStart / length)
  // and the length of the normals x component is tan(theta) = rEnd - rStart / length.
  //       | y axis
  //       |           ____
  //   phi |     v ___/    |
  //   \   |    __/        | <- (rEnd - rStart)
  //    \  | __/  theta    |
  //     \ |/______________|__ x axis
  const float normalX = -(endRadius - startRadius) / length;

  auto vertexIndex = [&](int iLength, int jRadius) {
    return vertexOffset + iLength * numCircleSubdivisions + (jRadius % numCircleSubdivisions);
  };

  // Build the vertices along the length:
  for (int iLength = 0; iLength <= numLengthSubdivisions; ++iLength) {
    const float fraction = float(iLength) / float(numLengthSubdivisions);
    const float xPos = translation + length * fraction;
    const float radiusCur = std::lerp(startRadius, endRadius, fraction);
    for (int jRadius = 0; jRadius < numCircleSubdivisions; ++jRadius) {
      const Eigen::Vector2f circlePt = circlePoints.at(jRadius);
      result.vertices.emplace_back(xPos, radiusCur * circlePt.x(), radiusCur * circlePt.y());
      // normal points outward:
      result.normals.emplace_back(
          Eigen::Vector3f(normalX, circlePt.x(), circlePt.y()).normalized());
    }
  }

  // Build the triangles:
  for (int iLength = 0; iLength < numLengthSubdivisions; ++iLength) {
    for (int jRadius = 0; jRadius < numCircleSubdivisions; ++jRadius) {
      result.faces.emplace_back(
          vertexIndex(iLength, jRadius),
          vertexIndex(iLength, jRadius + 1),
          vertexIndex(iLength + 1, jRadius + 1));
    }
  }

  vertexOffset += (numLengthSubdivisions + 1) * numCircleSubdivisions;

  // Construct the lower cap:
  for (int iCircle = 0; iCircle < numCircleSubdivisions; ++iCircle) {
    const auto cosTheta = circlePoints.at(iCircle).x();
    const auto sinTheta = circlePoints.at(iCircle).y();

    result.vertices.emplace_back(translation, innerRadius * cosTheta, innerRadius * sinTheta);
    result.vertices.emplace_back(translation, outerRadius * cosTheta, outerRadius * sinTheta);
    result.normals.push_back(-Eigen::Vector3f::UnitX());
    result.normals.push_back(-Eigen::Vector3f::UnitX());
  }

  for (int iCircle = 0; iCircle < numCircleSubdivisions; ++iCircle) {
    const int iCircleNext = (iCircle + 1) % numCircleSubdivisions;

    result.faces.emplace_back(
        vertexOffset + 2 * iCircle + 0,
        vertexOffset + 2 * iCircleNext + 0,
        vertexOffset + 2 * iCircle + 1);
    result.faces.emplace_back(
        vertexOffset + 2 * iCircle + 1,
        vertexOffset + 2 * iCircleNext + 0,
        vertexOffset + 2 * iCircleNext + 1);
  }

  return result;
}

Mesh makeArrow(
    int numCircleSubdivisions,
    int numLengthSubdivisions,
    float innerRadius,
    float outerRadius,
    float tipLength,
    float cylinderLength) {
  return mergeMeshes(
      {makeCylinderBody(numCircleSubdivisions, numLengthSubdivisions, innerRadius, cylinderLength),
       makeCylinderCap(numCircleSubdivisions, false, innerRadius),
       makeArrowhead(numCircleSubdivisions, innerRadius, outerRadius, tipLength, cylinderLength)});
}

Eigen::Matrix4f makeCylinderTransform(
    const Eigen::Vector3f& startPos,
    const Eigen::Vector3f& endPos,
    float radius) {
  Eigen::Affine3f cylinderXF = Eigen::Affine3f::Identity();
  cylinderXF.translate(startPos);
  cylinderXF.rotate(Eigen::Quaternionf::FromTwoVectors(
      Eigen::Vector3f::UnitX(), (endPos - startPos).normalized()));
  cylinderXF.scale(Eigen::Vector3f((endPos - startPos).norm(), radius, radius));
  return cylinderXF.matrix();
}

Eigen::Matrix4f makeSphereTransform(const Eigen::Vector3f& center, float radius) {
  Eigen::Affine3f sphereXF = Eigen::Affine3f::Identity();
  sphereXF.translate(center);
  sphereXF.scale(radius);
  return sphereXF.matrix();
}

std::array<Mesh, 2> makeCheckerboard(float width, int numChecks, int subdivisions) {
  // Precompute all the divisions of the plane to guarantee exact equality between
  // adjacent triangles:
  std::vector<float> xPositions;
  xPositions.reserve(numChecks * subdivisions + 1);
  for (int i = 0; i <= (numChecks * subdivisions); ++i) {
    xPositions.push_back(
        width * (static_cast<float>(i) / static_cast<float>(numChecks * subdivisions) - 0.5f));
  }

  std::array<Mesh, 2> result;

  for (size_t iMesh = 0; iMesh < 2; ++iMesh) {
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3i> triangles;

    for (int iCheck = 0; iCheck < numChecks; ++iCheck) {
      for (int jCheck = 0; jCheck < numChecks; ++jCheck) {
        if ((iCheck + jCheck) % 2 != iMesh) {
          continue;
        }

        const int positionsOffset = static_cast<int>(positions.size());
        auto vertexIndex = [&](int i, int j) {
          return positionsOffset + (i * (subdivisions + 1)) + j;
        };

        for (int iSubd = 0; iSubd <= subdivisions; ++iSubd) {
          for (int jSubd = 0; jSubd <= subdivisions; ++jSubd) {
            const float xPos = xPositions.at(iCheck * subdivisions + iSubd);
            const float zPos = xPositions.at(jCheck * subdivisions + jSubd);

            positions.emplace_back(xPos, 0, zPos);

            if (iSubd >= 1 && jSubd >= 1) {
              triangles.emplace_back(
                  vertexIndex(iSubd, jSubd),
                  vertexIndex(iSubd - 1, jSubd - 1),
                  vertexIndex(iSubd - 1, jSubd));
              triangles.emplace_back(
                  vertexIndex(iSubd, jSubd),
                  vertexIndex(iSubd, jSubd - 1),
                  vertexIndex(iSubd - 1, jSubd - 1));
            }
          }
        }
      }
    }

    std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f::UnitY());

    // Create mesh using momentum MeshT format
    result[iMesh].vertices = positions;
    result[iMesh].normals = normals;
    result[iMesh].faces = triangles;
  }

  return result;
}

std::vector<Eigen::Vector3f>
subdivideLines(const std::vector<Eigen::Vector3f>& lines, float maxLength, size_t maxSubdivisions) {
  std::vector<Eigen::Vector3f> result;
  result.reserve(lines.size());

  const size_t nLines = lines.size() / 2;
  MT_THROW_IF(nLines * 2 != lines.size(), "Expected lines size to be twice the number of lines");

  for (size_t i = 0; i < nLines; i++) {
    const Eigen::Vector3f& lineStart = lines[2 * i + 0];
    const Eigen::Vector3f& lineEnd = lines[2 * i + 1];
    const Eigen::Vector3f diff = lineEnd - lineStart;

    const float length = (lineEnd - lineStart).norm();
    size_t numSubdivisions =
        std::clamp<size_t>(static_cast<size_t>(std::ceil(length / maxLength)), 1, maxSubdivisions);

    for (size_t j = 0; j < numSubdivisions; j++) {
      result.emplace_back(
          lineStart + (static_cast<float>(j) / static_cast<float>(numSubdivisions)) * diff);
      result.emplace_back(
          lineStart + (static_cast<float>(j + 1) / static_cast<float>(numSubdivisions)) * diff);
    }
  }

  return result;
}

std::tuple<Mesh, std::vector<Eigen::Vector3f>> makeOctahedron(float radius, float midFraction) {
  std::array<Eigen::Vector3f, 4> midPoints = {
      Eigen::Vector3f(midFraction, -radius, -radius),
      Eigen::Vector3f(midFraction, radius, -radius),
      Eigen::Vector3f(midFraction, radius, radius),
      Eigen::Vector3f(midFraction, -radius, radius),
  };

  std::vector<Eigen::Vector3f> positions;
  std::vector<Eigen::Vector3f> normals;
  std::vector<Eigen::Vector3i> triangles;

  std::vector<Eigen::Vector3f> lines;

  // Reserve space for octahedron
  // Octahedron has 8 triangular faces, so 8 * 3 = 24 vertices (before subdivision)
  positions.reserve(24);
  normals.reserve(24);
  triangles.reserve(8);

  // Reserve space for lines: 4 midpoint edges + 4 center lines + 4 edge lines
  lines.reserve(16);

  for (int i = 0; i < 4; ++i) {
    lines.push_back(midPoints[i]);
    lines.push_back(midPoints[(i + 1) % 4]);
  }

  auto addVertex = [&positions, &normals](
                       const Eigen::Vector3f& p, const Eigen::Vector3f& n) -> int {
    const int idx = static_cast<int>(positions.size());
    positions.push_back(p);
    normals.push_back(n);
    return idx;
  };

  for (size_t iSide = 0; iSide < 4; ++iSide) {
    const Eigen::Vector3f p_cur = midPoints[iSide];
    const Eigen::Vector3f p_next = midPoints[(iSide + 1) % 4];

    lines.emplace_back(0, 0, 0);
    lines.push_back(midPoints[iSide]);
    lines.push_back(midPoints[iSide]);
    lines.emplace_back(1, 0, 0);

    {
      // left triangle:
      const Eigen::Vector3f& p1 = p_cur;
      const Eigen::Vector3f p2 = Eigen::Vector3f(0, 0, 0);
      const Eigen::Vector3f& p3 = p_next;

      const Eigen::Vector3f normal = (p2 - p1).cross(p3 - p1).normalized();

      MT_THROW_IF(
          normal.dot(Eigen::Vector3f(0, p_cur.y(), p_cur.z())) <= 0,
          "Normal check failed for p_cur");
      MT_THROW_IF(
          normal.dot(Eigen::Vector3f(0, p_next.y(), p_next.z())) <= 0,
          "Normal check failed for p_next");

      const int p1_idx = addVertex(p1, normal);
      const int p2_idx = addVertex(p2, normal);
      const int p3_idx = addVertex(p3, normal);

      triangles.emplace_back(p1_idx, p2_idx, p3_idx);
    }

    {
      // right triangle:
      const Eigen::Vector3f& p1 = p_cur;
      const Eigen::Vector3f& p2 = p_next;
      const Eigen::Vector3f p3 = Eigen::Vector3f(1, 0, 0);

      const Eigen::Vector3f normal = (p2 - p1).cross(p3 - p1).normalized();

      MT_THROW_IF(
          normal.dot(Eigen::Vector3f(0, p_cur.y(), p_cur.z())) <= 0,
          "Normal check failed for p_cur");
      MT_THROW_IF(
          normal.dot(Eigen::Vector3f(0, p_next.y(), p_next.z())) <= 0,
          "Normal check failed for p_next");

      const int p1_idx = addVertex(p1, normal);
      const int p2_idx = addVertex(p2, normal);
      const int p3_idx = addVertex(p3, normal);

      triangles.emplace_back(p1_idx, p2_idx, p3_idx);
    }
  }

  Mesh mesh;
  mesh.vertices = positions;
  mesh.normals = normals;
  mesh.faces = triangles;

  float max_edge_length = 0.1f;
  return {subdivideMeshNoSmoothing(mesh, max_edge_length), subdivideLines(lines, max_edge_length)};
}

namespace {

template <typename T1, typename T2>
Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> toMatrix(
    const std::vector<Eigen::Vector3<T1>>& v) {
  Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(v.size(), 3);
  for (int i = 0; i < v.size(); ++i) {
    result.row(i) = v[i].template cast<T2>();
  }
  return result;
}

using Edge = std::pair<int, int>;

Edge makeEdge(int v1, int v2) {
  if (v1 > v2) {
    std::swap(v1, v2);
  }
  return {v1, v2};
};

struct HashEdge {
  std::size_t operator()(const Edge& e) const {
    // Custom hash function combining two integers
    // Uses a variation of the FNV hash algorithm
    std::size_t h1 = std::hash<int>{}(e.first);
    std::size_t h2 = std::hash<int>{}(e.second);

    // Mix the hashes using a technique similar to boost::hash_combine
    // but without the boost dependency
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

struct Vertex {
  explicit Vertex(const Eigen::Vector3f& pos, const Eigen::Vector3f& n)
      : position(pos), normal(n) {}

  Eigen::Vector3f position;
  Eigen::Vector3f normal;
};

Vertex blend(const Vertex& a, const Vertex& b) {
  return Vertex(
      0.5f * a.position + 0.5f * b.position, (0.5f * a.normal + 0.5f * b.normal).normalized());
}

struct TextureVertex {
  explicit TextureVertex(const Eigen::Vector2f& uv) : uv(uv) {}

  Eigen::Vector2f uv;
};

TextureVertex blend(const TextureVertex& a, const TextureVertex& b) {
  return TextureVertex(0.5f * a.uv + 0.5f * b.uv);
}

std::vector<Vertex> toVertexList(
    Eigen::Ref<const RowMatrixXf> vertices_orig,
    Eigen::Ref<const RowMatrixXf> normals_orig) {
  const auto nVerts = vertices_orig.rows();
  if (normals_orig.rows() != nVerts) {
    throw std::runtime_error("Number of vertices must match number of normals.");
  }

  std::vector<Vertex> result;
  result.reserve(nVerts);
  for (Eigen::Index iRow = 0; iRow < nVerts; ++iRow) {
    result.emplace_back(vertices_orig.row(iRow), normals_orig.row(iRow));
  }
  return result;
}

std::vector<TextureVertex> toTextureVertexList(Eigen::Ref<const RowMatrixXf> textureCoords_orig) {
  const auto nVerts = textureCoords_orig.rows();

  std::vector<TextureVertex> result;
  result.reserve(nVerts);
  for (Eigen::Index iRow = 0; iRow < nVerts; ++iRow) {
    result.emplace_back(textureCoords_orig.row(iRow));
  }
  return result;
}

template <typename VertexType>
struct ExtendableTriangleMesh {
  explicit ExtendableTriangleMesh(std::vector<VertexType> vertices) : vertices(vertices) {}

  std::vector<Eigen::Vector3i> triangles;
  std::vector<VertexType> vertices;
  std::unordered_map<Edge, int, HashEdge> edgeVerts;

  int edgeVertex(Edge edge) {
    auto edgeItr = edgeVerts.find(edge);
    if (edgeItr != edgeVerts.end()) {
      return edgeItr->second;
    }

    bool inserted = false;
    std::tie(edgeItr, inserted) = edgeVerts.insert(std::make_pair(edge, vertices.size()));
    vertices.push_back(blend(vertices.at(edge.first), vertices.at(edge.second)));
    return edgeItr->second;
  }

  void addTriangle(int64_t i, int64_t j, int64_t k) {
    triangles.emplace_back(i, j, k);
  }
};

} // namespace

std::tuple<RowMatrixXf, RowMatrixXf, RowMatrixXi, RowMatrixXf, RowMatrixXi>
subdivideMeshNoSmoothing(
    Eigen::Ref<const RowMatrixXf> vertices_orig,
    Eigen::Ref<const RowMatrixXf> normals_orig,
    Eigen::Ref<const RowMatrixXi> triangles_orig,
    Eigen::Ref<const RowMatrixXf> textureCoords_orig,
    Eigen::Ref<const RowMatrixXi> textureTriangles_orig,
    float max_edge_length) {
  const auto nVertOrig = vertices_orig.rows();
  const auto nTrianglesOrig = triangles_orig.rows();

  if (nVertOrig == 0 || nTrianglesOrig == 0) {
    return {{}, {}, {}, {}, {}};
  }

  if (vertices_orig.cols() != 3) {
    throw std::runtime_error("Expected n x 3 vertex array.");
  }
  if (normals_orig.cols() != 3) {
    throw std::runtime_error("Expected n x 3 normals array.");
  }
  if (textureCoords_orig.rows() != 0 && textureCoords_orig.cols() != 2) {
    throw std::runtime_error("Expected n x 2 texture coordinates array.");
  }

  if (triangles_orig.cols() != 3) {
    throw std::runtime_error("Expected n x 3 triangles array.");
  }
  if (textureTriangles_orig.rows() != 0 && textureTriangles_orig.cols() != 3) {
    throw std::runtime_error("Expected n x 3 texture triangles array.");
  }
  if (textureTriangles_orig.rows() != 0 && textureTriangles_orig.rows() != nTrianglesOrig) {
    throw std::runtime_error("Texture triangles length must match triangles length.");
  }

  ExtendableTriangleMesh<Vertex> trimesh(toVertexList(vertices_orig, normals_orig));
  ExtendableTriangleMesh<TextureVertex> textureTrimesh(toTextureVertexList(
      textureCoords_orig.rows() == 0 ? Eigen::MatrixXf::Zero(nVertOrig, 2) : textureCoords_orig));

  for (Eigen::Index iTri = 0; iTri < nTrianglesOrig; ++iTri) {
    Eigen::Vector3i triangle = triangles_orig.row(iTri);
    Eigen::Vector3i textureTriangle =
        textureTriangles_orig.rows() == 0 ? triangle : textureTriangles_orig.row(iTri);

    // Edge vert is the vertex on the opposite edge:
    Eigen::Vector3i edgeVert(-1, -1, -1);
    Eigen::Vector3i textureEdgeVert(-1, -1, -1);
    for (int j = 0; j < 3; j++) {
      int vaI = triangle((j + 1) % 3);
      int vbI = triangle((j + 2) % 3);

      const float dist = (vertices_orig.row(vaI) - vertices_orig.row(vbI)).norm();
      if (dist > max_edge_length) {
        edgeVert(j) = trimesh.edgeVertex(makeEdge(vaI, vbI));
        textureEdgeVert(j) = textureTrimesh.edgeVertex(
            makeEdge(textureTriangle((j + 1) % 3), textureTriangle((j + 2) % 3)));
      }
    }

    // True if the missing edge vertices are sorted to the end of the vec:
    auto sortedEdgeVerts = [](const Eigen::Vector3i& v) {
      if (v.x() == -1) {
        return (v.y() == -1 || v.z() == -1);
      } else if (v.y() == -1) {
        return (v.z() == -1);
      } else {
        return true;
      }
    };

    auto rotateVec = [](const Eigen::Vector3i& v) { return Eigen::Vector3i(v.y(), v.z(), v.x()); };

    // Rotate until the missing edge vertices are at the end:
    while (!sortedEdgeVerts(edgeVert)) {
      triangle = rotateVec(triangle);
      edgeVert = rotateVec(edgeVert);
      textureTriangle = rotateVec(textureTriangle);
      textureEdgeVert = rotateVec(textureEdgeVert);
    }

    if (edgeVert.x() == -1) {
      // All edge vertices missing, return the original triangle:
      trimesh.triangles.push_back(triangle);
      textureTrimesh.triangles.push_back(textureTriangle);
    } else if (edgeVert.y() == -1) {
      // Last two edge vertices missing, so only the yz (opposite x) edge
      // vertex is there. Triangle looks like this:
      //
      //       v0
      //     / |  \
      //  v1 __|__ v2
      trimesh.triangles.emplace_back(triangle.x(), triangle.y(), edgeVert.x());
      trimesh.triangles.emplace_back(triangle.x(), edgeVert.x(), triangle.z());

      textureTrimesh.triangles.emplace_back(
          textureTriangle.x(), textureTriangle.y(), textureEdgeVert.x());
      textureTrimesh.triangles.emplace_back(
          textureTriangle.x(), textureEdgeVert.x(), textureTriangle.z());
    } else if (edgeVert.z() == -1) {
      // only one edge vertex missing (opposite z).  Ambiguous case where we
      // could construct the triangles in one of two ways:
      trimesh.triangles.emplace_back(triangle.x(), triangle.y(), edgeVert.y());
      trimesh.triangles.emplace_back(triangle.y(), edgeVert.x(), edgeVert.y());
      trimesh.triangles.emplace_back(edgeVert.y(), edgeVert.x(), triangle.z());

      textureTrimesh.triangles.emplace_back(
          textureTriangle.x(), textureTriangle.y(), textureEdgeVert.y());
      textureTrimesh.triangles.emplace_back(
          textureTriangle.y(), textureEdgeVert.x(), textureEdgeVert.y());
      textureTrimesh.triangles.emplace_back(
          textureEdgeVert.y(), textureEdgeVert.x(), textureTriangle.z());
    } else {
      trimesh.triangles.emplace_back(triangle.x(), edgeVert.z(), edgeVert.y());
      trimesh.triangles.emplace_back(edgeVert.z(), triangle.y(), edgeVert.x());
      trimesh.triangles.emplace_back(edgeVert.x(), triangle.z(), edgeVert.y());
      trimesh.triangles.emplace_back(edgeVert.z(), edgeVert.x(), edgeVert.y());

      textureTrimesh.triangles.emplace_back(
          textureTriangle.x(), textureEdgeVert.z(), textureEdgeVert.y());
      textureTrimesh.triangles.emplace_back(
          textureEdgeVert.z(), textureTriangle.y(), textureEdgeVert.x());
      textureTrimesh.triangles.emplace_back(
          textureEdgeVert.x(), textureTriangle.z(), textureEdgeVert.y());
      textureTrimesh.triangles.emplace_back(
          textureEdgeVert.z(), textureEdgeVert.x(), textureEdgeVert.y());
    }
  }

  Eigen::MatrixXf vertices_new(trimesh.vertices.size(), 3);
  Eigen::MatrixXf normals_new(trimesh.vertices.size(), 3);
  for (size_t i = 0; i < trimesh.vertices.size(); ++i) {
    vertices_new.row(i) = trimesh.vertices[i].position;
    normals_new.row(i) = trimesh.vertices[i].normal;
  }

  Eigen::MatrixXf texture_coords_new(textureTrimesh.vertices.size(), 2);
  for (size_t i = 0; i < textureTrimesh.vertices.size(); ++i) {
    texture_coords_new.row(i) = textureTrimesh.vertices[i].uv;
  }

  return {
      vertices_new,
      normals_new,
      toMatrix<int, int>(trimesh.triangles),
      texture_coords_new,
      toMatrix<int, int>(textureTrimesh.triangles)};
}

Mesh subdivideMeshNoSmoothing(const Mesh& mesh, float max_edge_length, size_t max_depth) {
  // Convert momentum mesh to row matrices
  RowMatrixXf positions(mesh.vertices.size(), 3);
  RowMatrixXf normals(mesh.normals.size(), 3);
  RowMatrixXi triangles(mesh.faces.size(), 3);

  for (size_t i = 0; i < mesh.vertices.size(); ++i) {
    positions.row(i) = mesh.vertices[i];
  }
  for (size_t i = 0; i < mesh.normals.size(); ++i) {
    normals.row(i) = mesh.normals[i];
  }
  for (size_t i = 0; i < mesh.faces.size(); ++i) {
    triangles.row(i) = mesh.faces[i];
  }

  for (size_t depth = 0; depth <= max_depth; depth++) {
    float maxEdgeLengthCur = 0;
    for (int iTri = 0; iTri < triangles.rows(); ++iTri) {
      for (int j = 0; j < 3; ++j) {
        const int vi = triangles(iTri, j);
        const int vj = triangles(iTri, (j + 1) % 3);
        maxEdgeLengthCur =
            std::max(maxEdgeLengthCur, (positions.row(vi) - positions.row(vj)).norm());
      }
    }

    if (maxEdgeLengthCur < max_edge_length) {
      break;
    }

    std::tie(positions, normals, triangles, std::ignore, std::ignore) = subdivideMeshNoSmoothing(
        positions, normals, triangles, RowMatrixXf{}, RowMatrixXi{}, max_edge_length);
  }

  // Convert back to momentum mesh format
  Mesh result;

  // Convert vertices
  result.vertices.reserve(positions.rows());
  for (int i = 0; i < positions.rows(); ++i) {
    result.vertices.emplace_back(positions.row(i));
  }

  // Convert normals
  result.normals.reserve(normals.rows());
  for (int i = 0; i < normals.rows(); ++i) {
    result.normals.emplace_back(normals.row(i));
  }

  // Convert triangles
  result.faces.reserve(triangles.rows());
  for (int i = 0; i < triangles.rows(); ++i) {
    result.faces.emplace_back(triangles.row(i));
  }

  return result;
}

std::vector<Eigen::Vector3f>
makeCameraFrustumLines(const Camera& camera, float distance, size_t nSamples) {
  auto windowToEyeWithLength =
      [&camera](const uint32_t x, const uint32_t y, const float length) -> Eigen::Vector3f {
    const auto [p_eye, valid] = camera.getIntrinsicsModel().unproject(Eigen::Vector3f(x, y, 1.0));
    const Eigen::Vector3f dir = p_eye.normalized().cast<float>();
    return (dir * length).eval();
  };

  const uint32_t w = camera.imageWidth();
  const uint32_t h = camera.imageHeight();

  const Eigen::Vector3f camOrigin = Eigen::Vector3f::Zero();
  const Eigen::Vector3f p1_far = windowToEyeWithLength(0, 0, distance).cast<float>();
  const Eigen::Vector3f p2_far = windowToEyeWithLength(0, h - 1, distance);
  const Eigen::Vector3f p3_far = windowToEyeWithLength(w - 1, h - 1, distance);
  const Eigen::Vector3f p4_far = windowToEyeWithLength(w - 1, 0, distance);

  std::vector<Eigen::Vector3f> linesEye;

  linesEye.push_back(camOrigin);
  linesEye.push_back(p1_far);
  linesEye.push_back(camOrigin);
  linesEye.push_back(p2_far);
  linesEye.push_back(camOrigin);
  linesEye.push_back(p3_far);
  linesEye.push_back(camOrigin);
  linesEye.push_back(p4_far);

  const uint32_t nCircles = 3;
  const uint32_t pixJump = std::min(w, h) / (2 * nCircles + 1);
  std::vector<Eigen::Vector3f> lineLoop;
  lineLoop.reserve(4 * nSamples);
  for (uint32_t iCircle = 0; iCircle < nCircles; ++iCircle) {
    lineLoop.clear();
    const uint32_t curOffset = iCircle * pixJump;
    const uint32_t cur_h = h - 2 * curOffset;
    const uint32_t cur_w = w - 2 * curOffset;
    for (uint32_t j = 0; j <= nSamples; ++j) {
      lineLoop.push_back(
          windowToEyeWithLength(curOffset, curOffset + j * (cur_h - 1) / nSamples, distance));
    }
    for (uint32_t j = 0; j <= nSamples; ++j) {
      lineLoop.push_back(windowToEyeWithLength(
          curOffset + j * (cur_w - 1) / nSamples, curOffset + cur_h - 1, distance));
    }
    for (uint32_t j = 0; j <= nSamples; ++j) {
      lineLoop.push_back(windowToEyeWithLength(
          curOffset + cur_w - 1, curOffset + (nSamples - j) * (cur_h - 1) / nSamples, distance));
    }
    for (uint32_t j = 0; j <= nSamples; ++j) {
      lineLoop.push_back(windowToEyeWithLength(
          curOffset + (nSamples - j) * (cur_w - 1) / nSamples, curOffset, distance));
    }

    for (size_t i = 0; i < lineLoop.size(); ++i) {
      linesEye.push_back(lineLoop.at(i));
      linesEye.push_back(lineLoop.at((i + 1) % lineLoop.size()));
    }
  }

  return linesEye;
}

} // namespace momentum::rasterizer
