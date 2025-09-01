/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/legacy_json/legacy_json_io.h"

#include <momentum/character/character.h>
#include <momentum/character/collision_geometry.h>
#include <momentum/character/joint.h>
#include <momentum/character/locator.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skin_weights.h>
#include <momentum/character/types.h>
#include <momentum/common/exception.h>
#include <momentum/io/common/stream_utils.h>
#include <momentum/math/mesh.h>
#include <momentum/math/transform.h>
#include <momentum/math/types.h>
#include <fstream>
#include <sstream>

namespace momentum {

namespace {

/// Helper function to convert Vector3 to JSON array
template <typename T>
nlohmann::json eigenToJsonArray(const Eigen::Matrix<T, 3, 1>& vec) {
  return nlohmann::json::array({vec.x(), vec.y(), vec.z()});
}

/// Helper function to convert Vector2 to JSON array
template <typename T>
nlohmann::json eigenToJsonArray(const Eigen::Matrix<T, 2, 1>& vec) {
  return nlohmann::json::array({vec.x(), vec.y()});
}

/// Helper function to convert Quaternion to JSON array (w, x, y, z)
template <typename T>
nlohmann::json eigenToJsonArray(const Eigen::Quaternion<T>& quat) {
  return nlohmann::json::array({quat.x(), quat.y(), quat.z(), quat.w()});
}

/// Helper function to convert Transform to JSON object
template <typename T>
nlohmann::json transformToJson(const TransformT<T>& transform) {
  nlohmann::json j;

  // Extract translation
  j["Translation"] = eigenToJsonArray(transform.translation);

  // Extract rotation as quaternion
  j["Rotation"] = eigenToJsonArray(transform.rotation);

  // Extract scale
  j["Scale"] = transform.scale;

  return j;
}

/// Helper function to convert JSON array to Vector3
template <typename T>
Eigen::Matrix<T, 3, 1> jsonArrayToEigenVector3(const nlohmann::json& j) {
  MT_THROW_IF(j.size() != 3, "Expected array of size 3 for Vector3");
  return Eigen::Matrix<T, 3, 1>(j[0].get<T>(), j[1].get<T>(), j[2].get<T>());
}

/// Helper function to convert JSON array to Vector2
template <typename T>
Eigen::Matrix<T, 2, 1> jsonArrayToEigenVector2(const nlohmann::json& j) {
  MT_THROW_IF(j.size() != 2, "Expected array of size 2 for Vector2");
  return Eigen::Matrix<T, 2, 1>(j[0].get<T>(), j[1].get<T>());
}

/// Helper function to convert JSON array to Quaternion (w, x, y, z)
template <typename T>
Eigen::Quaternion<T> jsonArrayToEigenQuaternion(const nlohmann::json& j) {
  MT_THROW_IF(j.size() != 4, "Expected array of size 4 for Quaternion");
  // Eigen constructor takes (w, x, y, z); json storage is (x, y, z, w)
  return Eigen::Quaternion<T>(j[3].get<T>(), j[0].get<T>(), j[1].get<T>(), j[2].get<T>());
}

/// Helper function to convert JSON object to Transform
template <typename T>
TransformT<T> jsonToTransform(const nlohmann::json& j) {
  TransformT<T> transform; // Identity by default

  if (j.contains("Translation")) {
    transform.translation = jsonArrayToEigenVector3<T>(j["Translation"]);
  }

  if (j.contains("Rotation")) {
    transform.rotation = jsonArrayToEigenQuaternion<T>(j["Rotation"]);
  }

  if (j.contains("Scale")) {
    transform.scale = j["Scale"].get<T>();
  }

  return transform;
}

/// Helper function to find field names with multiple possible variants
nlohmann::json::const_iterator findFieldNames(
    const nlohmann::json& json,
    const std::initializer_list<const char*>& names) {
  for (const auto& name : names) {
    auto itr = json.find(name);
    if (itr != json.end()) {
      return itr;
    }
  }
  return json.end();
}

Skeleton legacySkeletonToMomentum(const nlohmann::json& legacySkeleton) {
  MT_THROW_IF(!legacySkeleton.contains("Bones"), "Legacy skeleton JSON missing 'Bones' field");

  const auto& bones = legacySkeleton["Bones"];
  JointList joints;
  joints.reserve(bones.size());

  for (const auto& boneJson : bones) {
    Joint joint;

    // Extract joint name
    joint.name = boneJson["Name"].get<std::string>();

    // Extract parent index
    joint.parent = boneJson["Parent"].get<size_t>();

    // Extract pre-rotation quaternion
    if (boneJson.contains("PreRotation")) {
      joint.preRotation = jsonArrayToEigenQuaternion<float>(boneJson["PreRotation"]);
    } else {
      joint.preRotation = Eigen::Quaternionf::Identity();
    }

    // Extract translation offset
    if (boneJson.contains("TranslationOffset")) {
      joint.translationOffset = jsonArrayToEigenVector3<float>(boneJson["TranslationOffset"]);
    } else {
      joint.translationOffset = Eigen::Vector3f::Zero();
    }

    joints.push_back(joint);
  }

  return Skeleton(joints);
}

nlohmann::json momentumSkeletonToLegacy(const Skeleton& skeleton) {
  nlohmann::json legacySkeleton;
  nlohmann::json bones = nlohmann::json::array();

  for (const auto& joint : skeleton.joints) {
    nlohmann::json boneJson;

    boneJson["Name"] = joint.name;
    boneJson["Parent"] = joint.parent;
    boneJson["PreRotation"] = eigenToJsonArray(joint.preRotation);
    boneJson["TranslationOffset"] = eigenToJsonArray(joint.translationOffset);

    // Add default values for legacy compatibility
    boneJson["RestState"] = nlohmann::json{
        {"Rot", nlohmann::json::array({0.0, 0.0, 0.0})},
        {"Trans", nlohmann::json::array({0.0, 0.0, 0.0})},
        {"Scale", 0.0}};

    // Determine joint type
    if (joint.parent == kInvalidIndex) {
      boneJson["JointType"] = "Root";
    } else {
      boneJson["JointType"] = "Limb";
    }

    boneJson["RotationOrder"] = "XYZ";

    bones.push_back(boneJson);
  }

  legacySkeleton["Bones"] = bones;
  return legacySkeleton;
}

std::pair<Mesh, SkinWeights> legacySkinnedModelToMomentum(
    const nlohmann::json& legacySkinnedModel) {
  Mesh mesh;
  SkinWeights skinWeights;

  // Extract vertices - support multiple naming variants
  auto verticesItr = findFieldNames(legacySkinnedModel, {"RestPositions", "vertices"});
  if (verticesItr != legacySkinnedModel.end()) {
    const auto& vertices = *verticesItr;
    mesh.vertices.reserve(vertices.size());
    for (const auto& vertex : vertices) {
      mesh.vertices.push_back(jsonArrayToEigenVector3<float>(vertex));
    }
  }

  // Extract normals - support multiple naming variants
  auto normalsItr = findFieldNames(legacySkinnedModel, {"RestVertexNormals", "normals"});
  if (normalsItr != legacySkinnedModel.end()) {
    const auto& normals = *normalsItr;
    mesh.normals.reserve(normals.size());
    for (const auto& normal : normals) {
      mesh.normals.push_back(jsonArrayToEigenVector3<float>(normal));
    }
  }

  // Extract faces - support multiple naming variants and triangulate if needed
  auto facesItr = findFieldNames(legacySkinnedModel, {"Faces", "faces"});
  if (facesItr != legacySkinnedModel.end()) {
    const auto& facesObj = *facesItr;
    // Expecting "Indices", optional "TextureIndices", and "Offsets"
    MT_THROW_IF(!facesObj.contains("Indices"), "Faces object missing 'Indices' field");
    MT_THROW_IF(!facesObj.contains("Offsets"), "Faces object missing 'Offsets' field");

    const auto& indices = facesObj["Indices"];
    const auto& offsets = facesObj["Offsets"];
    // Optional texture indices
    const auto* texIndicesPtr =
        facesObj.contains("TextureIndices") ? &facesObj["TextureIndices"] : nullptr;

    size_t numFaces = offsets.size() - 1;
    mesh.faces.reserve(numFaces * 2); // Conservative estimate for triangulation

    for (size_t faceIdx = 0; faceIdx < numFaces; ++faceIdx) {
      size_t start = offsets[faceIdx].get<size_t>();
      size_t end = offsets[faceIdx + 1].get<size_t>();
      size_t faceSize = end - start;

      // Triangulate face (ngon) using fan triangulation
      if (faceSize < 3) {
        MT_THROW_IF(true, "Face must have at least 3 vertices");
      } else {
        int v0 = indices[start].get<int>();
        for (size_t i = 1; i < faceSize - 1; ++i) {
          int v1 = indices[start + i].get<int>();
          int v2 = indices[start + i + 1].get<int>();
          mesh.faces.emplace_back(v0, v1, v2);
        }
        if (texIndicesPtr && texIndicesPtr->is_array()) {
          int t0 = (*texIndicesPtr)[start].get<int>();
          for (size_t i = 1; i < faceSize - 1; ++i) {
            int t1 = (*texIndicesPtr)[start + i].get<int>();
            int t2 = (*texIndicesPtr)[start + i + 1].get<int>();
            mesh.texcoord_faces.emplace_back(t0, t1, t2);
          }
        }
      }
    }

    // Shrink to actual size after triangulation
    mesh.faces.shrink_to_fit();
    mesh.texcoord_faces.shrink_to_fit();
  }

  // Extract texture coordinates - support multiple naming variants
  auto texcoordsItr = findFieldNames(legacySkinnedModel, {"TextureCoordinates", "texcoords"});
  if (texcoordsItr != legacySkinnedModel.end()) {
    const auto& texcoords = *texcoordsItr;
    mesh.texcoords.reserve(texcoords.size());
    for (const auto& texcoord : texcoords) {
      mesh.texcoords.push_back(jsonArrayToEigenVector2<float>(texcoord));
    }
  }

  // Try Trinity rig format with separate SkinningWeights and SkinningOffsets
  auto skinningWeightsItr = findFieldNames(legacySkinnedModel, {"SkinningWeights"});
  auto skinningOffsetsItr = findFieldNames(legacySkinnedModel, {"SkinningOffsets"});

  if (skinningWeightsItr != legacySkinnedModel.end() &&
      skinningOffsetsItr != legacySkinnedModel.end()) {
    const auto& skinningWeights = *skinningWeightsItr;
    const auto& skinningOffsets = *skinningOffsetsItr;

    if (skinningWeights.is_array() && skinningOffsets.is_array()) {
      const size_t numVertices = skinningOffsets.size() - 1;
      skinWeights.index = IndexMatrix::Zero(numVertices, kMaxSkinJoints);
      skinWeights.weight = WeightMatrix::Zero(numVertices, kMaxSkinJoints);

      for (size_t i = 0; i < numVertices; ++i) {
        size_t start = skinningOffsets[i].get<size_t>();
        size_t end = skinningOffsets[i + 1].get<size_t>();
        size_t numInfluences = std::min(end - start, static_cast<size_t>(kMaxSkinJoints));

        for (size_t j = 0; j < numInfluences; ++j) {
          const auto& pair = skinningWeights[start + j];
          // Each pair is [joint index, skinning weight]
          skinWeights.index(i, j) = pair[0].get<uint32_t>();
          skinWeights.weight(i, j) = pair[1].get<float>();
        }
      }
    }
  }

  return std::make_pair(std::move(mesh), std::move(skinWeights));
}

nlohmann::json momentumSkinnedModelToLegacy(const Mesh& mesh, const SkinWeights& skinWeights) {
  nlohmann::json legacySkinnedModel;

  // Convert vertices
  nlohmann::json restPositions = nlohmann::json::array();
  for (const auto& vertex : mesh.vertices) {
    restPositions.push_back(eigenToJsonArray(vertex));
  }
  legacySkinnedModel["RestPositions"] = restPositions;

  // Convert normals
  if (!mesh.normals.empty()) {
    nlohmann::json restVertexNormals = nlohmann::json::array();
    for (const auto& normal : mesh.normals) {
      restVertexNormals.push_back(eigenToJsonArray(normal));
    }
    legacySkinnedModel["RestVertexNormals"] = restVertexNormals;
  }

  // Convert faces (legacy expects an object with Indices, Offsets, and optionally TextureIndices)
  nlohmann::json facesObj;
  nlohmann::json indices = nlohmann::json::array();
  nlohmann::json offsets = nlohmann::json::array();
  nlohmann::json textureIndices = nlohmann::json::array();

  // To reconstruct offsets, we need to group triangles into faces.
  // But since we only have triangles, we treat each triangle as a face.
  size_t numFaces = mesh.faces.size();
  offsets.push_back(0);
  for (size_t i = 0; i < numFaces; ++i) {
    const auto& face = mesh.faces[i];
    indices.push_back(face.x());
    indices.push_back(face.y());
    indices.push_back(face.z());
    offsets.push_back((i + 1) * 3);
    // If texcoord_faces exist, add them as TextureIndices
    if (!mesh.texcoord_faces.empty()) {
      const auto& tface = mesh.texcoord_faces[i];
      textureIndices.push_back(tface.x());
      textureIndices.push_back(tface.y());
      textureIndices.push_back(tface.z());
    }
  }
  facesObj["Indices"] = indices;
  facesObj["Offsets"] = offsets;
  if (!textureIndices.empty()) {
    facesObj["TextureIndices"] = textureIndices;
  }
  legacySkinnedModel["Faces"] = facesObj;

  // Convert texture coordinates
  if (!mesh.texcoords.empty()) {
    nlohmann::json textureCoordinates = nlohmann::json::array();
    for (const auto& texcoord : mesh.texcoords) {
      textureCoordinates.push_back(eigenToJsonArray(texcoord));
    }
    legacySkinnedModel["TextureCoordinates"] = textureCoordinates;
  }

  // Convert skin weights to Trinity format: SkinningWeights and SkinningOffsets
  nlohmann::json skinningWeights = nlohmann::json::array();
  nlohmann::json skinningOffsets = nlohmann::json::array();
  size_t runningIndex = 0;
  skinningOffsets.push_back(runningIndex);
  for (int i = 0; i < skinWeights.index.rows(); ++i) {
    int influences = 0;
    for (int j = 0; j < skinWeights.index.cols(); ++j) {
      if (skinWeights.weight(i, j) > 0.0f) {
        nlohmann::json pair = nlohmann::json::array();
        pair.push_back(skinWeights.index(i, j));
        pair.push_back(skinWeights.weight(i, j));
        skinningWeights.push_back(pair);
        ++influences;
      }
    }
    runningIndex += influences;
    skinningOffsets.push_back(runningIndex);
  }
  legacySkinnedModel["SkinningWeights"] = skinningWeights;
  legacySkinnedModel["SkinningOffsets"] = skinningOffsets;

  return legacySkinnedModel;
}

CollisionGeometry legacyCollisionToMomentum(const nlohmann::json& legacyCollision) {
  CollisionGeometry collision;

  if (legacyCollision.is_array()) {
    collision.reserve(legacyCollision.size());

    for (const auto& capsuleJson : legacyCollision) {
      TaperedCapsule capsule;

      if (capsuleJson.contains("transformation")) {
        TransformT<float> transform = jsonToTransform<float>(capsuleJson["transformation"]);
        capsule.transformation = transform.toAffine3();
      }

      if (capsuleJson.contains("radius")) {
        capsule.radius = jsonArrayToEigenVector2<float>(capsuleJson["radius"]);
      }

      if (capsuleJson.contains("parent")) {
        capsule.parent = capsuleJson["parent"].get<size_t>();
      }

      if (capsuleJson.contains("length")) {
        capsule.length = capsuleJson["length"].get<float>();
      }

      collision.push_back(capsule);
    }
  }

  return collision;
}

nlohmann::json momentumCollisionToLegacy(const CollisionGeometry& collision) {
  nlohmann::json legacyCollision = nlohmann::json::array();

  for (const auto& capsule : collision) {
    nlohmann::json capsuleJson;

    TransformT<float> transform = TransformT<float>::fromAffine3(capsule.transformation);
    capsuleJson["transformation"] = transformToJson(transform);
    capsuleJson["radius"] = eigenToJsonArray(capsule.radius);
    capsuleJson["parent"] = capsule.parent;
    capsuleJson["length"] = capsule.length;

    legacyCollision.push_back(capsuleJson);
  }

  return legacyCollision;
}

LocatorList legacyLocatorsToMomentum(const nlohmann::json& legacyLocators) {
  LocatorList locators;

  if (legacyLocators.is_array()) {
    locators.reserve(legacyLocators.size());

    for (const auto& locatorJson : legacyLocators) {
      Locator locator;

      if (locatorJson.contains("name")) {
        locator.name = locatorJson["name"].get<std::string>();
      }

      if (locatorJson.contains("parent")) {
        locator.parent = locatorJson["parent"].get<size_t>();
      }

      if (locatorJson.contains("offset")) {
        locator.offset = jsonArrayToEigenVector3<float>(locatorJson["offset"]);
      } else if (
          locatorJson.contains("offsetX") && locatorJson.contains("offsetY") &&
          locatorJson.contains("offsetZ")) {
        // Legacy format with separate X, Y, Z fields
        locator.offset = Eigen::Vector3f(
            locatorJson["offsetX"].get<float>(),
            locatorJson["offsetY"].get<float>(),
            locatorJson["offsetZ"].get<float>());
      }

      // Set default values for other fields
      locator.locked = Eigen::Vector3i::Zero();
      locator.weight = 1.0f;
      locator.limitOrigin = locator.offset;
      locator.limitWeight = Eigen::Vector3f::Zero();

      locators.push_back(locator);
    }
  }

  return locators;
}

nlohmann::json momentumLocatorsToLegacy(const LocatorList& locators) {
  nlohmann::json legacyLocators = nlohmann::json::array();

  for (const auto& locator : locators) {
    nlohmann::json locatorJson;

    locatorJson["name"] = locator.name;
    locatorJson["parent"] = locator.parent;
    locatorJson["offsetX"] = locator.offset.x();
    locatorJson["offsetY"] = locator.offset.y();
    locatorJson["offsetZ"] = locator.offset.z();

    legacyLocators.push_back(locatorJson);
  }

  return legacyLocators;
}

} // namespace

Character loadCharacterFromLegacyJson(const std::string& jsonPath) {
  std::ifstream file(jsonPath);
  MT_THROW_IF(!file.is_open(), "Failed to open legacy JSON file: {}", jsonPath);

  nlohmann::json j;
  file >> j;

  return loadCharacterFromLegacyJsonString(j.dump());
}

Character loadCharacterFromLegacyJsonBuffer(gsl::span<const std::byte> jsonBuffer) {
  std::string jsonString(reinterpret_cast<const char*>(jsonBuffer.data()), jsonBuffer.size());
  return loadCharacterFromLegacyJsonString(jsonString);
}

Character loadCharacterFromLegacyJsonString(const std::string& jsonString) {
  nlohmann::json j = nlohmann::json::parse(jsonString);

  // Extract skeleton - support multiple naming variants for backward compatibility
  auto skeletonItr = findFieldNames(j, {"Skeleton", "BodySkeleton", "skeleton"});
  MT_THROW_IF(skeletonItr == j.end(), "Legacy JSON missing skeleton field");
  Skeleton skeleton = legacySkeletonToMomentum(*skeletonItr);

  // Create default parameter transform
  ParameterTransform parameterTransform =
      ParameterTransform::empty(skeleton.joints.size() * kParametersPerJoint);

  // Extract mesh and skin weights if present - support multiple naming variants
  std::unique_ptr<Mesh> mesh;
  std::unique_ptr<SkinWeights> skinWeights;

  auto skinnedModelItr = findFieldNames(j, {"SkinnedModel", "BodySkinnedModel", "skinnedmodel"});
  if (skinnedModelItr != j.end()) {
    auto [meshData, skinWeightsData] = legacySkinnedModelToMomentum(*skinnedModelItr);
    mesh = std::make_unique<Mesh>(std::move(meshData));
    skinWeights = std::make_unique<SkinWeights>(std::move(skinWeightsData));
  }

  // Extract collision geometry if present - support multiple naming variants
  std::unique_ptr<CollisionGeometry> collision;
  auto collisionItr = findFieldNames(j, {"Collision", "collision"});
  if (collisionItr != j.end()) {
    CollisionGeometry collisionData = legacyCollisionToMomentum(*collisionItr);
    if (!collisionData.empty()) {
      collision = std::make_unique<CollisionGeometry>(std::move(collisionData));
    }
  }

  // Extract locators if present - support multiple naming variants
  LocatorList locators;
  auto locatorsItr = findFieldNames(j, {"Locators", "locators"});
  if (locatorsItr != j.end()) {
    locators = legacyLocatorsToMomentum(*locatorsItr);
  }

  // Create and return the character
  return {
      skeleton,
      parameterTransform,
      ParameterLimits(), // Default parameter limits
      locators,
      mesh.get(),
      skinWeights.get(),
      collision.get()};
}

void saveCharacterToLegacyJson(const Character& character, const std::string& jsonPath) {
  nlohmann::json j = characterToLegacyJson(character);

  std::ofstream file(jsonPath);
  MT_THROW_IF(!file.is_open(), "Failed to open file for writing: {}", jsonPath);

  file << j.dump(2); // Pretty print with 2-space indentation
}

std::string characterToLegacyJsonString(const Character& character) {
  nlohmann::json j = characterToLegacyJson(character);
  return j.dump(2);
}

nlohmann::json characterToLegacyJson(const Character& character) {
  nlohmann::json j;

  // Convert skeleton - use capitalized naming to match BasicBodyProfile
  j["Skeleton"] = momentumSkeletonToLegacy(character.skeleton);

  // Convert mesh and skin weights if present - use capitalized naming to match BasicBodyProfile
  if (character.mesh && character.skinWeights) {
    j["SkinnedModel"] = momentumSkinnedModelToLegacy(*character.mesh, *character.skinWeights);
  }

  // Convert collision geometry if present
  if (character.collision && !character.collision->empty()) {
    j["Collision"] = momentumCollisionToLegacy(*character.collision);
  }

  // Convert locators if present
  if (!character.locators.empty()) {
    j["Locators"] = momentumLocatorsToLegacy(character.locators);
  }

  return j;
}

} // namespace momentum
