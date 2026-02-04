/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <momentum/camera/camera.h>
#include <momentum/math/constants.h>
#include <momentum/rasterizer/geometry.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

using namespace momentum::rasterizer;

// Local implementation of angle between two normalized vectors
// (to avoid dependency on nimble::Math)
float angleBetween(const Eigen::Vector3f& u, const Eigen::Vector3f& v) {
  // From: http://www.plunk.org/~hatch/rightway.php
  return 2.0f * std::atan2(0.5f * (u - v).norm(), 0.5f * (u + v).norm());
}

void checkNormalsValid(const Mesh& mesh, float maxAngle) {
  for (size_t iVert = 0; iVert < mesh.vertices.size(); ++iVert) {
    const Eigen::Vector3f normal = mesh.normals[iVert];
    ASSERT_GT(normal.norm(), 0.0f);
  }

  for (size_t iTri = 0; iTri < mesh.faces.size(); ++iTri) {
    const Eigen::Vector3i tri = mesh.faces[iTri];
    const Eigen::Vector3f faceNormal = (mesh.vertices[tri[1]] - mesh.vertices[tri[0]])
                                           .cross(mesh.vertices[tri[2]] - mesh.vertices[tri[0]])
                                           .normalized();

    for (size_t iVert = 0; iVert < 3; ++iVert) {
      const Eigen::Vector3f& normal = mesh.normals[tri[iVert]];
      ASSERT_LE(angleBetween(normal, faceNormal), maxAngle)
          << "Face normal " << faceNormal.transpose() << " doesn't match vertex normal "
          << normal.transpose() << " for tri " << iTri << ": " << tri.transpose()
          << "; vertices are " << mesh.vertices[tri[0]].transpose() << " // "
          << mesh.vertices[tri[1]].transpose() << " // " << mesh.vertices[tri[2]].transpose();
    }
  }
}

void checkTrianglesValid(const Mesh& mesh) {
  for (size_t iTri = 0; iTri < mesh.faces.size(); ++iTri) {
    const auto& tri = mesh.faces[iTri];
    for (size_t iVert = 0; iVert < 3; ++iVert) {
      ASSERT_GE(tri[iVert], 0);
      ASSERT_LT(tri[iVert], mesh.vertices.size());
      ASSERT_NE(tri[iVert], tri[(iVert + 1) % 3]);
    }
  }
}

void dumpObjFile(const Mesh& mesh, const std::string& filename) {
  std::ofstream oss(filename.c_str());
  if (!oss.good()) {
    throw std::runtime_error("Unable to open file " + filename + " for writing.");
  }

  for (size_t iVert = 0; iVert < mesh.vertices.size(); ++iVert) {
    const auto& pos = mesh.vertices[iVert];
    const auto& n = mesh.normals[iVert];
    oss << "v " << pos.x() << " " << pos.y() << " " << pos.z() << "\n";
    oss << "vn " << n.x() << " " << n.y() << " " << n.z() << "\n";
  }

  for (size_t iTri = 0; iTri < mesh.faces.size(); ++iTri) {
    const Eigen::Vector3i tri = mesh.faces[iTri] + Eigen::Vector3i::Ones();
    oss << "f ";
    for (int k = 0; k < 3; ++k) {
      oss << tri[k] << "//" << tri[k] << " ";
    }
    oss << "\n";
  }
}

TEST(SoftwareRasterizer, CreateCylinder) {
  const auto cylinder = makeCylinder(20, 10);
  // dumpObjFile(cylinder, "/Users/cdtwigg/cylinder.obj");
  checkTrianglesValid(cylinder);
  checkNormalsValid(cylinder, momentum::pi<float>() / 16.0f);
}

TEST(SoftwareRasterizer, CreateCapsule) {
  const auto capsule = makeCapsule(20, 10, 1.0, 0.5, 2.0);
  // dumpObjFile(capsule, "/Users/cdtwigg/capsule.obj");
  checkTrianglesValid(capsule);
  checkNormalsValid(capsule, momentum::pi<float>() / 16.0f);
}

TEST(SoftwareRasterizer, CreateArrow) {
  const auto arrowhead = makeArrow(20, 10, 0.2, 0.4, 0.5, 1.0);
  // dumpObjFile(arrowhead, "/Users/cdtwigg/arrow.obj");
  checkTrianglesValid(arrowhead);
  checkNormalsValid(arrowhead, momentum::pi<float>() / 16.0f);
}

TEST(SoftwareRasterizer, CreateSphere) {
  const auto sphere = makeSphere(2);
  // dumpObjFile(sphere, "/Users/cdtwigg/sphere.obj");
  checkTrianglesValid(sphere);
  checkNormalsValid(
      sphere, momentum::pi<float>() / 8.0f); // Allow slightly larger angle tolerance for sphere

  // Additional checks for sphere-specific properties
  ASSERT_GT(sphere.vertices.size(), 0);
  ASSERT_GT(sphere.faces.size(), 0);

  // Check that all vertices are approximately on unit sphere (radius=1)
  const float radiusTolerance = 0.01f;
  for (size_t i = 0; i < sphere.vertices.size(); ++i) {
    const Eigen::Vector3f pos = sphere.vertices[i];
    const float radius = pos.norm();
    ASSERT_NEAR(radius, 1.0f, radiusTolerance)
        << "Vertex " << i << " at position " << pos.transpose() << " has radius " << radius
        << " (expected ~1.0)";

    // Check that normal points outward (should be same direction as position for unit sphere)
    const Eigen::Vector3f normal = sphere.normals[i];
    const float dotProduct = pos.normalized().dot(normal.normalized());
    ASSERT_GT(dotProduct, 0.9f) << "Normal " << normal.transpose() << " at vertex " << i
                                << " doesn't point outward from position " << pos.transpose();
  }
}
