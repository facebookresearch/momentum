/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/openfbx_loader.h"

#include "momentum/character/character.h"
#include "momentum/character/collision_geometry_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character/types.h"
#include "momentum/common/log.h"
#include "momentum/io/common/gsl_utils.h"
#include "momentum/io/fbx/polygon_data.h"
#include "momentum/io/skeleton/locator_io.h"
#include "momentum/io/skeleton/parameter_limits_io.h"
#include "momentum/io/skeleton/parameter_transform_io.h"
#include "momentum/io/skeleton/parameters_io.h"
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/math/utility.h"

#include <momentum/common/filesystem.h>
#include <ofbx.h>
#include <gsl/span_ext>

#include <cmath>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <variant>

namespace momentum {

namespace {

Eigen::Vector3d toEigen(const ofbx::DVec3& v) {
  return {v.x, v.y, v.z};
}

Eigen::Affine3d toEigen(const ofbx::DMatrix& mat) {
  return Eigen::Affine3d(Eigen::Matrix4d(mat.m));
}

Eigen::Vector3i getRotationOrder(ofbx::RotationOrder order) {
  switch (order) {
    case ofbx::RotationOrder::EULER_XYZ:
    default:
      return {0, 1, 2};
    case ofbx::RotationOrder::EULER_XZY:
      return {0, 2, 1};
    case ofbx::RotationOrder::EULER_YZX:
      return {1, 2, 0};
    case ofbx::RotationOrder::EULER_YXZ:
      return {1, 0, 2};
    case ofbx::RotationOrder::EULER_ZXY:
      return {2, 0, 1};
    case ofbx::RotationOrder::EULER_ZYX:
      return {2, 1, 0};
  }
}

const char* rotationOrderStr(ofbx::RotationOrder order) {
  switch (order) {
    case ofbx::RotationOrder::EULER_XYZ:
      return "EULER_XYZ";
    case ofbx::RotationOrder::EULER_XZY:
      return "EULER_XZY";
    case ofbx::RotationOrder::EULER_YZX:
      return "EULER_YZX";
    case ofbx::RotationOrder::EULER_YXZ:
      return "EULER_YXZ";
    case ofbx::RotationOrder::EULER_ZXY:
      return "EULER_ZXY";
    case ofbx::RotationOrder::EULER_ZYX:
      return "EULER_ZYX";
    case ofbx::RotationOrder::SPHERIC_XYZ:
      return "SPHERIC_XYZ";
    default:
      return "Unknown";
  }
}

const char* propertyTypeStr(ofbx::IElementProperty::Type type) {
  switch (type) {
    case ofbx::IElementProperty::LONG:
      return "LONG";
    case ofbx::IElementProperty::INTEGER:
      return "INTEGER";
    case ofbx::IElementProperty::STRING:
      return "STRING";
    case ofbx::IElementProperty::FLOAT:
      return "FLOAT";
    case ofbx::IElementProperty::DOUBLE:
      return "DOUBLE";
    case ofbx::IElementProperty::ARRAY_DOUBLE:
      return "ARRAY_DOUBLE";
    case ofbx::IElementProperty::ARRAY_INT:
      return "ARRAY_INT";
    case ofbx::IElementProperty::ARRAY_LONG:
      return "ARRAY_LONG";
    case ofbx::IElementProperty::ARRAY_FLOAT:
      return "ARRAY_FLOAT";
    default:
      return "Unknown";
  }
}

template <typename T>
Eigen::Quaternion<T> computeEulerRotation(
    const Eigen::Matrix<T, 3, 1>& angles,
    ofbx::RotationOrder order) {
  using VecType = Eigen::Matrix<T, 3, 1>;
  using QuaternionType = Eigen::Quaternion<T>;

  if (order == ofbx::RotationOrder::SPHERIC_XYZ) {
    // in the case of a ball-joint the 3 parameters are simply the xyz parameters of a quaternion
    return QuaternionType(
        std::sqrt(std::max(T(1) - angles.squaredNorm(), T(0))), angles.x(), angles.y(), angles.z());
  } else {
    const Eigen::Vector3i rotOrder = getRotationOrder(order);
    QuaternionType result = QuaternionType::Identity();
    for (uint32_t index = 0; index < 3; ++index) {
      const auto& rIndex = rotOrder(index);
      result = QuaternionType(Eigen::AngleAxis<T>(angles(rIndex), VecType::Unit(rIndex))) * result;
    }
    return result;
  }
}

Eigen::Quaterniond fbxEulerRotationToQuat(
    const ofbx::DVec3& eulerAngles,
    const ofbx::RotationOrder rotOrder) {
  const Eigen::Vector3d rotDegrees = toEigen(eulerAngles).head<3>();
  const Eigen::Vector3d rotRadians = toRad<double>() * rotDegrees;
  return computeEulerRotation<double>(rotRadians, rotOrder);
}

const ofbx::IElement* findChild(const ofbx::IElement& element, const char* id) {
  auto* curChild = element.getFirstChild();
  while (curChild) {
    if (curChild->getID() == id) {
      return curChild;
    }
    curChild = curChild->getSibling();
  }
  return nullptr;
}

bool equalsCaseInsensitive(const ofbx::DataView& data, const char* rhs) {
  const char* c = rhs;
  const char* c2 = (const char*)data.begin;
  while (*c && c2 != (const char*)data.end) {
    if (tolower(*c) != tolower(*c2)) {
      return 0;
    }
    ++c;
    ++c2;
  }
  return c2 == (const char*)data.end && *c == '\0';
}

// This function is more or less copied from the OpenFBX SDK.
ofbx::IElement* resolveProperty(const ofbx::Object& object, const char* name) {
  // This is black magic, but for some reason all the user properties are stored in an
  // element named Properties70.
  const ofbx::IElement* props = findChild(object.element, "Properties70");
  if (!props) {
    return nullptr;
  }

  ofbx::IElement* prop = props->getFirstChild();
  while (prop) {
    auto* firstProp = prop->getFirstProperty();
    MT_THROW_IF(
        firstProp->getType() != ofbx::IElementProperty::STRING,
        "Expected string for first property value but got {}.",
        propertyTypeStr(firstProp->getType()));

    // The Autodesk SDK appears to be case-insensitive wrt names, so we'll mirror that
    // behavior here.
    if (firstProp != nullptr && equalsCaseInsensitive(firstProp->getValue(), name)) {
      return prop;
    }
    prop = prop->getSibling();
  }
  return nullptr;
}

// Get the nth property, needed because elements store properties in the order [name, ?, ?, ?,
// value].
const ofbx::IElementProperty* getElementProperty(const ofbx::IElement* element, size_t iProp) {
  auto* prop = element->getFirstProperty();
  for (size_t j = 0; j < iProp; ++j) {
    if (prop == nullptr) {
      return nullptr;
    }
    prop = prop->getNext();
  }

  return prop;
}

double resolveDoubleProperty(const ofbx::Object& object, const char* name) {
  const ofbx::IElement* element = resolveProperty(object, name);
  MT_THROW_IF(element == nullptr, "Unable to find property element in {}", object.name);
  const ofbx::IElementProperty* x = getElementProperty(element, 4);
  MT_THROW_IF(x == nullptr, "Unable to find property {} in {}", name, object.name);
  if (x->getType() == ofbx::IElementProperty::DOUBLE) {
    return x->getValue().toDouble();
  } else if (x->getType() == ofbx::IElementProperty::FLOAT) {
    return x->getValue().toFloat();
  } else {
    MT_THROW(
        "For property {}, expected float/double but got {}.", name, propertyTypeStr(x->getType()));
  }
}

template <typename VecArray, typename EltType>
VecArray extractPropertyArrayImp(const ofbx::IElementProperty* prop, const char* what) {
  using VecType = typename VecArray::value_type;
  using Scalar = typename VecType::Scalar;

  const auto nScalar = prop->getCount();
  if (nScalar == 0) {
    return VecArray();
  }

  const auto vecLength = VecType::RowsAtCompileTime;
  const auto nVec = nScalar / vecLength;

  MT_THROW_IF(
      nVec * vecLength != nScalar,
      "For {}; expected to be divisible by {} but got {} as count",
      what,
      int(vecLength),
      nScalar);

  std::vector<double> vals(nScalar);
  prop->getValues(vals.data(), (int)(sizeof(EltType) * vals.size()));

  VecArray result;
  result.reserve(nVec);
  for (int iVec = 0; iVec < nVec; ++iVec) {
    VecType v;
    for (Eigen::DenseIndex k = 0; k < vecLength; ++k) {
      v[k] = (Scalar)vals[iVec * static_cast<size_t>(vecLength) + k];
    }
    result.push_back(v);
  }

  return result;
}

template <typename VecArray>
VecArray extractPropertyVecArray(const ofbx::IElement* element, const char* what) {
  const auto* prop = element->getFirstProperty();
  MT_THROW_IF(prop == nullptr, "For element {} found no property.", what);
  if (prop->getType() == ofbx::IElementProperty::ARRAY_DOUBLE) {
    return extractPropertyArrayImp<VecArray, double>(prop, what);
  } else if (prop->getType() == ofbx::IElementProperty::ARRAY_FLOAT) {
    return extractPropertyArrayImp<VecArray, float>(prop, what);
  } else {
    MT_THROW(
        "For property {} expected float/double array but got {}.",
        what,
        propertyTypeStr(prop->getType()));
  }
}

template <typename T>
ofbx::IElementProperty::Type propertyArrayType();
template <>
ofbx::IElementProperty::Type propertyArrayType<double>() {
  return ofbx::IElementProperty::ARRAY_DOUBLE;
}
template <>
ofbx::IElementProperty::Type propertyArrayType<float>() {
  return ofbx::IElementProperty::ARRAY_FLOAT;
}
template <>
ofbx::IElementProperty::Type propertyArrayType<int>() {
  return ofbx::IElementProperty::ARRAY_INT;
}

template <typename T>
std::vector<T> extractPropertyArray(const ofbx::IElement* element, const char* what) {
  const auto* prop = element->getFirstProperty();
  MT_THROW_IF(prop == nullptr, "For element {} found no property.", what);
  if (prop->getType() == propertyArrayType<T>()) {
    std::vector<T> vals(prop->getCount());
    prop->getValues(&vals[0], (int)(sizeof(T) * vals.size()));
    return vals;
  } else {
    MT_THROW(
        "For property {}, expected {} but got {}.",
        what,
        propertyTypeStr(propertyArrayType<T>()),
        propertyTypeStr(prop->getType()));
  }
}

std::vector<double> extractPropertyFloatArray(const ofbx::IElement* element, const char* what) {
  const auto* prop = element->getFirstProperty();
  MT_THROW_IF(prop == nullptr, "For element {}, found no property.", what);
  if (prop->getType() == ofbx::IElementProperty::ARRAY_DOUBLE) {
    return extractPropertyArray<double>(element, what);
  } else if (prop->getType() == ofbx::IElementProperty::ARRAY_FLOAT) {
    auto res = extractPropertyArray<float>(element, what);
    std::vector<double> result(res.begin(), res.end());
    return result;
  } else {
    MT_THROW(
        "For property {}, expected float array but got {}.",
        what,
        propertyTypeStr(prop->getType()));
  }
}

void parseSkeleton(
    const ofbx::Object* curSkelNode,
    const size_t parent,
    Skeleton& skeleton,
    std::vector<const ofbx::Object*>& fbxObjects,
    CollisionGeometry& capsules,
    LocatorList& locators,
    bool permissive) {
  MT_CHECK(curSkelNode, "Skeleton node for parent '{}' is null", parent);

  // Skip non-transform nodes:
  if (!curSkelNode->isNode()) {
    return;
  }
  const auto type = curSkelNode->getType();
  const std::string jointName = curSkelNode->name;

  // A transform node could be a lot of different objects.
  if (type == ofbx::Object::Type::NULL_NODE) {
    // Check for collision capsule in a custom attribute
    auto* res = resolveProperty(*curSkelNode, "col_type");

    if (res != nullptr) {
      // Extract the capsule if present:
      const double length = resolveDoubleProperty(*curSkelNode, "length");
      const double rad_a = resolveDoubleProperty(*curSkelNode, "rad_a");
      const double rad_b = resolveDoubleProperty(*curSkelNode, "rad_b");

      const auto xf = curSkelNode->getLocalTransform();

      TaperedCapsule capsule;
      capsule.parent = parent;
      capsule.length = length;
      capsule.radius = {rad_a, rad_b};
      capsule.transformation = toEigen(xf).cast<float>();
      capsules.push_back(capsule);
    } else if (parent != kInvalidIndex) {
      // It's a locator if it has a parent joint
      Locator locator;
      locator.name = jointName;
      const auto translationOffset = curSkelNode->getLocalTranslation();
      locator.offset = toEigen(translationOffset).cast<float>();
      locator.parent = parent;
      // TODO when we have rotations in the locator:
      // locator.rotationOffset = preRotation.cast<float>();
      locators.push_back(locator);
    } else if (parent == kInvalidIndex) {
      // If we haven't found the skeleton root yet, this could be a group node and we will try to
      // traverse it.
      int i = 0;
      while (ofbx::Object* child = curSkelNode->resolveObjectLink(i++)) {
        parseSkeleton(child, kInvalidIndex, skeleton, fbxObjects, capsules, locators, permissive);
      }
    }
  } else if (type == ofbx::Object::Type::LIMB_NODE) {
    // Get node transformation info
    // get rotation order
    auto order = curSkelNode->getRotationOrder();
    if (order != ofbx::RotationOrder::EULER_XYZ) {
      if (permissive) {
        MT_LOGW(
            "momentum supports only XYZ rotation; joint {} has {} rotation order.",
            jointName,
            rotationOrderStr(order));
      } else {
        MT_THROW(
            "momentum supports only XYZ rotation; joint {} has {} rotation order.",
            jointName,
            rotationOrderStr(order));
      }
    }

    // get local rotation
    auto localRot = fbxEulerRotationToQuat(curSkelNode->getLocalRotation(), order);
    if (Eigen::AngleAxisd(localRot).angle() > 1e-2) {
      if (permissive) {
        MT_LOGW(
            "{}: Node {} has nonzero rest rotation ({}, {}, {}), which will be baked into the skeleton.",
            __func__,
            jointName,
            curSkelNode->getLocalRotation().x,
            curSkelNode->getLocalRotation().y,
            curSkelNode->getLocalRotation().z);
      } else {
        MT_LOGE(
            "{}: Node {} has nonzero rest rotation ({}, {}, {}) and it will be ignored",
            __func__,
            jointName,
            curSkelNode->getLocalRotation().x,
            curSkelNode->getLocalRotation().y,
            curSkelNode->getLocalRotation().z);
        localRot = Eigen::Quaterniond::Identity();
      }
    }

    // get pre- and post-rotation
    const auto preRotEuler = curSkelNode->getPreRotation();
    // Pre-rot always uses XYZ order:
    const Eigen::Quaterniond preRotation =
        fbxEulerRotationToQuat(preRotEuler, ofbx::RotationOrder::EULER_XYZ) * localRot;

    MT_LOGE_IF(
        toEigen(curSkelNode->getPostRotation()).head<3>().norm() > 1e-8,
        "{}: Node {} has nonzero post-rotation; it will be ignored.",
        __func__,
        jointName);

    // get local offset
    const auto translationOffset = curSkelNode->getLocalTranslation();

    // ignore scaling values for now

    // Create a skeleton joint
    Joint joint;
    joint.name = jointName;
    joint.parent = parent;
    joint.preRotation = preRotation.cast<float>();
    joint.translationOffset = toEigen(translationOffset).cast<float>();
    const size_t jointIndex = skeleton.joints.size();
    skeleton.joints.push_back(joint);
    fbxObjects.push_back(curSkelNode);

    // Traverse the skeleton hierarchy.
    int i = 0;
    while (ofbx::Object* child = curSkelNode->resolveObjectLink(i++)) {
      parseSkeleton(child, jointIndex, skeleton, fbxObjects, capsules, locators, permissive);
    }
  }
}

std::tuple<Skeleton, std::vector<const ofbx::Object*>, LocatorList, CollisionGeometry>
parseSkeleton(const ofbx::Object* sceneRoot, const std::string& skelRoot, bool permissive) {
  MT_CHECK(sceneRoot, "Scene root with skel root '{}' is null", skelRoot);

  Skeleton skeleton;
  LocatorList locators;
  CollisionGeometry collision;
  std::vector<const ofbx::Object*>
      jointFbxNodes; // Contains the mapping between FbxNode* and indices in the FbxSkeleton

  int i = 0;
  while (ofbx::Object* child = sceneRoot->resolveObjectLink(i)) {
    i++;
    if (!skelRoot.empty() && child->name != skelRoot) {
      continue;
    }
    parseSkeleton(child, kInvalidIndex, skeleton, jointFbxNodes, collision, locators, permissive);
  }

  return {skeleton, jointFbxNodes, locators, collision};
}

void parseSkinnedModel(
    const ofbx::Mesh* meshRoot,
    const std::vector<const ofbx::Object*>& boneFbxNodes,
    Mesh& mesh,
    std::unique_ptr<SkinWeights>& skinWeights,
    TransformationList& inverseBindPoseTransforms,
    bool permissive) {
  enum EMapping {
    MappingUnknown,
    MappingByPolyVertex,
    MappingByVertex,
  };
  enum EReference {
    RefUnknown,
    RefIndexToDirect,
    RefDirect,
  };

  // We will parse out the geometry ourselves rather than using OpenFBX's
  // Geometry class, since the latter throws away a lot of useful information.
  const auto* geometry = meshRoot->getGeometry();
  const auto& geomElement = geometry->element;

  const auto* vertices_element = findChild(geomElement, "Vertices");
  MT_THROW_IF(
      vertices_element == nullptr || !vertices_element->getFirstProperty(),
      "No vertices found in mesh element.");
  const auto vertexPositions =
      extractPropertyVecArray<std::vector<Eigen::Vector3f>>(vertices_element, "Vertices");
  const auto nVerts = vertexPositions.size();

  const auto* polys_element = findChild(geomElement, "PolygonVertexIndex");
  MT_THROW_IF(
      polys_element == nullptr || !polys_element->getFirstProperty(),
      "No polygons found in mesh element.");
  const auto indices = extractPropertyArray<int>(polys_element, "PolygonVertexIndex");

  PolygonData vertices;
  vertices.indices.reserve(indices.size());
  for (const auto& i : indices) {
    if (i < 0) {
      // end of polygon is indicated by a negative index:
      vertices.indices.push_back(-(i + 1));
      vertices.offsets.push_back((uint32_t)vertices.indices.size());
    } else {
      vertices.indices.push_back(i);
    }
  }

  std::vector<Eigen::Vector2f> textureCoords;
  const auto* layer_uv_element = findChild(geomElement, "LayerElementUV");
  if (layer_uv_element != nullptr) {
    const auto* uvs_element = findChild(*layer_uv_element, "UV");
    if (uvs_element == nullptr || !uvs_element->getFirstProperty()) {
      // Some legitimate uses of fbxsdk (ie fbx_io.cpp:saveFbx) seem to be unable to write to this
      // element. So we are not hard-failing to remain "compatible".
      MT_LOGE("No UVs found in mesh element.");
    } else {
      EMapping mapping = MappingUnknown;
      const auto* mapping_element = findChild(*layer_uv_element, "MappingInformationType");
      if (mapping_element != nullptr && mapping_element->getFirstProperty() != nullptr) {
        const ofbx::DataView& view = mapping_element->getFirstProperty()->getValue();
        if (view == "ByPolygonVertex") {
          mapping = MappingByPolyVertex;
        } else if (view == "ByVertex" || view == "ByVertice") {
          mapping = MappingByVertex;
        }
      }
      MT_THROW_IF(
          mapping == MappingUnknown,
          "Don't currently know how to deal with mapping type that is not 'ByPolygonVertex' or 'ByVertex'.");

      EReference reference = RefUnknown;
      const auto* reference_element = findChild(*layer_uv_element, "ReferenceInformationType");
      if (reference_element != nullptr && reference_element->getFirstProperty() != nullptr) {
        const ofbx::DataView& view = reference_element->getFirstProperty()->getValue();
        if (view == "IndexToDirect") {
          reference = RefIndexToDirect;
        } else if (view == "Direct") {
          reference = RefDirect;
        }
      }
      MT_THROW_IF(
          reference == RefUnknown,
          "Don't currently know how to deal with reference type that is not 'IndexToDirect' or 'Direct'.");

      // Coords array is handled the same for either mapping type
      textureCoords = extractPropertyVecArray<std::vector<Eigen::Vector2f>>(uvs_element, "UV");

      if (reference == RefIndexToDirect) {
        // IndexToDirect means there is another mapping array which gives the order of the UVs in
        // the mesh
        const auto* indices_element = findChild(*layer_uv_element, "UVIndex");
        MT_THROW_IF(indices_element == nullptr, "Missing indices element.");
        const auto textureIndices =
            extractPropertyArray<int>(indices_element, "PolygonVertexIndex");

        MT_THROW_IF(
            textureIndices.size() != vertices.indices.size(),
            "Mismatch between texture indices size {} and vertex indices size {}.",
            textureIndices.size(),
            vertices.indices.size());
        std::copy(
            textureIndices.begin(),
            textureIndices.end(),
            std::back_inserter(vertices.textureIndices));
      } else if (reference == RefDirect) {
        // Direct means the UV array is already in order.
        if (mapping == MappingByPolyVertex) {
          MT_THROW_IF(
              textureCoords.size() != vertices.indices.size(),
              "Mismatch between 'Direct' texture coord array size {} and vertex indices size {}.",
              textureCoords.size(),
              vertices.indices.size());
          vertices.textureIndices.resize(vertices.indices.size());
        } else if (mapping == MappingByVertex) {
          MT_THROW_IF(
              textureCoords.size() != nVerts,
              "Mismatch between 'Direct' texture coord array size {} and vertex count {}.",
              textureCoords.size(),
              nVerts);
        }
        std::iota(vertices.textureIndices.begin(), vertices.textureIndices.end(), 0);
      } else {
        MT_THROW("UV reading code failed to handle a valid reference type. This is a bug.");
      }
    }
  }

  // Momentum wants the y coords flipped:
  for (auto& tc : textureCoords) {
    tc.y() = 1.0f - tc.y();
  }

  auto errMsg = vertices.errorMessage(nVerts);
  MT_THROW_IF(!errMsg.empty(), "Error reading polygons from FBX file: {}", errMsg);
  errMsg = vertices.warnMessage(textureCoords.size());
  MT_LOGW_IF(!errMsg.empty(), "Error reading polygon data from FBX file: {}", errMsg);

  const size_t vertexOffset = mesh.vertices.size();
  std::copy(
      std::begin(vertexPositions), std::end(vertexPositions), std::back_inserter(mesh.vertices));
  for (const auto& t : triangulate(vertices.indices, vertices.offsets)) {
    mesh.faces.emplace_back(t + Eigen::Vector3i::Constant(vertexOffset));
  }

  mesh.normals.resize(vertexOffset + nVerts, Vector3f::Zero());
  mesh.colors.resize(vertexOffset + nVerts, Vector3b::Zero());
  mesh.confidence.resize(vertexOffset + nVerts, 1);

  const size_t textureCoordOffset = mesh.texcoords.size();
  std::copy(std::begin(textureCoords), std::end(textureCoords), std::back_inserter(mesh.texcoords));
  for (const auto& t : triangulate(vertices.textureIndices, vertices.offsets)) {
    mesh.texcoord_faces.emplace_back(t + Eigen::Vector3i::Constant(textureCoordOffset));
  }

  const auto* fbxskin = geometry->getSkin();
  if (!fbxskin) {
    MT_THROW_IF(
        !permissive,
        "No skin found for geometry. Enable permissive mode to allow saving as a mesh-only character.");
    return; // Just return a mesh-only character.
  }

  // Need a fast map from an FbxNode to the bone index in our representation:
  std::unordered_map<const ofbx::Object*, size_t> boneMap;
  for (size_t i = 0; i < boneFbxNodes.size(); ++i) {
    boneMap.insert(std::make_pair(boneFbxNodes[i], i));
  }
  // The weights in the FBX file are stored by bone rather than by
  // vertex; we will cache them all as (vertex, bone, weight) in
  // this array and then sort it to get them in vertex order.
  using VertexBoneWithWeight = std::tuple<size_t, size_t, double>;
  std::vector<VertexBoneWithWeight> weights;
  weights.reserve(2 * nVerts);

  int clusterCount = fbxskin->getClusterCount();
  for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++) {
    const auto* cluster = fbxskin->getCluster(clusterIndex);
    MT_CHECK(cluster != nullptr);

    const auto* bone = cluster->getLink();

    const auto fbxJointItr = boneMap.find(bone);
    MT_THROW_IF(
        fbxJointItr == boneMap.end(),
        "Cluster {} references invalid bone: {}",
        cluster->name,
        bone->name);
    const size_t boneIndex = fbxJointItr->second;
    if (boneIndex < inverseBindPoseTransforms.size()) {
      inverseBindPoseTransforms[boneIndex] =
          toEigen(cluster->getTransformLinkMatrix()).inverse().cast<float>();
    }

    const auto* skinning_indices_element = findChild(cluster->element, "Indexes");
    if (skinning_indices_element == nullptr || !skinning_indices_element->getFirstProperty()) {
      MT_LOGT(
          "Skipping as no skinning indices found in cluster element {} (mesh is {}).",
          cluster->name,
          meshRoot->name);
      continue;
    }

    const auto skinningIndices = extractPropertyArray<int>(skinning_indices_element, "Indexes");

    const auto* skinning_weights_element = findChild(cluster->element, "Weights");
    MT_THROW_IF(
        skinning_weights_element == nullptr || !skinning_weights_element->getFirstProperty(),
        "No skinning weights found in cluster element {} (mesh is {}).",
        cluster->name,
        meshRoot->name);
    const auto skinningWeights = extractPropertyFloatArray(skinning_weights_element, "Weights");

    // iterate through all the vertices, which are affected by the bone
    MT_THROW_IF(
        skinningIndices.size() != skinningWeights.size(),
        "Mismatch between indices count ({}) and weight count ({}) in cluster {}",
        skinningIndices.size(),
        skinningWeights.size(),
        cluster->name);
    const auto numBoneVertexIndices = skinningIndices.size();
    for (size_t boneVIndex = 0; boneVIndex < numBoneVertexIndices; boneVIndex++) {
      const int boneVertexIndex = skinningIndices[boneVIndex];
      MT_THROW_IF(
          boneVertexIndex < 0 || static_cast<size_t>(boneVertexIndex) >= nVerts,
          "Invalid vertex index ({}) in cluster {}",
          boneVertexIndex,
          cluster->name);
      const auto boneWeight = skinningWeights[boneVIndex];
      if (boneWeight <= 0) {
        continue;
      }
      MT_LOGW_IF(
          boneWeight > 1.01,
          "{}: Bone weight of {} found; expected value between 0 and 1.",
          __func__,
          boneWeight);
      weights.emplace_back(boneVertexIndex, boneIndex, boneWeight);
    }
  }

  if (!skinWeights) {
    skinWeights = std::make_unique<SkinWeights>();
  }
  skinWeights->index.conservativeResize(vertexOffset + nVerts, Eigen::NoChange);
  skinWeights->weight.conservativeResize(vertexOffset + nVerts, Eigen::NoChange);
  skinWeights->index.bottomRows(nVerts).setZero();
  skinWeights->weight.bottomRows(nVerts).setZero();

  std::stable_sort(
      weights.begin(),
      weights.end(),
      [](const VertexBoneWithWeight& b1, const VertexBoneWithWeight& b2) {
        return std::get<0>(b1) < std::get<0>(b2);
      });
  auto weightItr = weights.begin();

  using BoneWeight = std::pair<size_t, double>;
  std::vector<BoneWeight> curBoneWeights;
  for (size_t iVertex = 0; iVertex < nVerts; ++iVertex) {
    curBoneWeights.clear();
    MT_THROW_IF(
        weightItr == weights.end() || std::get<0>(*weightItr) != iVertex,
        "No weights found for vertex {} in mesh {}",
        iVertex,
        meshRoot->name);

    auto vertWeightsBegin = weightItr;
    while (weightItr != weights.end() && std::get<0>(*weightItr) == iVertex) {
      ++weightItr;
    }
    auto vertWeightsEnd = weightItr;

    for (auto itr = vertWeightsBegin; itr != vertWeightsEnd; ++itr) {
      const auto boneIndex = std::get<1>(*itr);
      const auto boneWeight = std::clamp(std::get<2>(*itr), 0.0, 1.0);
      curBoneWeights.emplace_back(boneIndex, boneWeight);
    }

    std::stable_sort(
        curBoneWeights.begin(),
        curBoneWeights.end(),
        [](const BoneWeight& b1, const BoneWeight& b2) { return b1.second > b2.second; });
    curBoneWeights.resize(std::min<size_t>(curBoneWeights.size(), kMaxSkinJoints));

    double weightSum = 0;
    for (const auto& [idx, weight] : curBoneWeights) {
      weightSum += weight;
    }

    MT_THROW_IF(
        weightSum <= 0, "Empty weight sum found for vertex {} in mesh {}", iVertex, meshRoot->name);

    for (int iPr = 0; iPr < curBoneWeights.size() && iPr < kMaxSkinJoints; ++iPr) {
      const auto [boneIdx, weight] = curBoneWeights[iPr];
      skinWeights->index(vertexOffset + iVertex, iPr) = boneIdx;
      skinWeights->weight(vertexOffset + iVertex, iPr) = weight / weightSum;
    }
  }
}

double getMaxSeconds(const std::vector<const ofbx::AnimationCurveNode*>& curves) {
  ofbx::i64 fbxTime = 0;
  for (const ofbx::AnimationCurveNode* node : curves) {
    for (size_t iChannel = 0; iChannel < 3; ++iChannel) {
      const ofbx::AnimationCurve* channel = node->getCurve(iChannel);
      if (channel == nullptr) {
        continue;
      }
      const int count = channel->getKeyCount();
      if (count <= 0) {
        continue;
      }
      const ofbx::i64* time = channel->getKeyTime();
      if (time[count - 1] > fbxTime) {
        fbxTime = time[count - 1];
      }
    }
  }
  return ofbx::fbxTimeToSeconds(fbxTime);
}

size_t findJointIndex(
    const ofbx::Object* queryObj,
    const std::vector<const ofbx::Object*>& fbxNodes) {
  size_t result = kInvalidIndex;
  for (size_t iNode = 0; iNode < fbxNodes.size(); ++iNode) {
    if (queryObj == fbxNodes[iNode]) {
      result = iNode;
      break;
    }
  }

  return result;
}

MatrixXf parseAnimation(
    const ofbx::AnimationStack* animStack,
    const std::vector<const ofbx::Object*>& boneFbxNodes,
    const Skeleton& skeleton,
    const float fps,
    bool permissive) {
  // Return motion in numJointParameters X numFrames
  MatrixXf motion;
  std::vector<const ofbx::AnimationCurveNode*> animCurves;

  // Collect all anim curves
  int iLayer = 0;
  while (const ofbx::AnimationLayer* animLayer = animStack->getLayer(iLayer++)) {
    int iCurve = 0;
    while (const ofbx::AnimationCurveNode* curve = animLayer->getCurveNode(iCurve++)) {
      animCurves.push_back(curve);
    }
  }
  if (animCurves.empty()) {
    return {};
  }

  const double totalSeconds = getMaxSeconds(animCurves);
  const size_t numFrames = ceil(totalSeconds * fps) + 1;
  motion.setZero(skeleton.joints.size() * kParametersPerJoint, numFrames);

  for (size_t i = 0; i < skeleton.joints.size(); ++i) {
    // Load the rest pose into the animation; for channels that aren't animated, Maya
    // won't export a curve node and so this is the only way to get the rest pose set
    // up correctly.
    // Note that the rest translation is already baked into the translationOffset in
    // the skeleton so we don't want to set it here.
    if (!permissive) {
      // Local rotation is Euler angles:
      const auto localRot = boneFbxNodes[i]->getLocalRotation();
      motion.row(i * kParametersPerJoint + 3).setConstant(toRad(localRot.x));
      motion.row(i * kParametersPerJoint + 4).setConstant(toRad(localRot.y));
      motion.row(i * kParametersPerJoint + 5).setConstant(toRad(localRot.z));
    }
    motion.row(i * kParametersPerJoint + 6)
        .setConstant(std::log2(boneFbxNodes[i]->getLocalScaling().x));
  }

  std::vector<bool> writtenJoints(skeleton.joints.size() * kParametersPerJoint, false);

  // Curves on the same node will be added together.
  for (const ofbx::AnimationCurveNode* curveNode : animCurves) {
    // Find the skeleton joint
    const size_t jointIndex = findJointIndex(curveNode->getBone(), boneFbxNodes);
    if (jointIndex == kInvalidIndex) {
      continue;
    }
    // Find out which channels
    const auto& property = curveNode->getBoneLinkProperty();
    int mode = -1;
    if (property == "Lcl Translation") {
      mode = 0;
    } else if (property == "Lcl Rotation") {
      mode = 1;
    } else if (property == "Lcl Scaling") {
      mode = 2;
    } else {
      continue;
    }
    // Find the indices to write to
    const size_t startIndex = jointIndex * kParametersPerJoint + mode * 3;
    const size_t endIndex = startIndex + ((mode == 2) ? 1 : 3);
    for (size_t i = startIndex; i < endIndex; ++i) {
      if (writtenJoints.at(i)) {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        char propertyName[128];
        property.toString(propertyName);
        MT_LOGW(
            "Multiple curves affecting the same parameter {} in {}",
            propertyName,
            curveNode->getBone()->name);
      }

      writtenJoints.at(i) = true;
    }

    // Linear interpolation of the curve on timeline
    for (size_t iFrame = 0; iFrame < numFrames; ++iFrame) {
      const ofbx::DVec3 values =
          curveNode->getNodeLocalTransform(static_cast<double>(iFrame) / fps);

      // Aggregate the motion
      if (mode == 0) {
        // FBX unit is default to centimeter - the same as Momentum unit
        if (jointIndex < skeleton.joints.size()) {
          motion.col(iFrame).template segment<3>(startIndex).noalias() =
              toEigen(values).cast<float>() - skeleton.joints[jointIndex].translationOffset;
        }
      } else if (mode == 1) {
        // assume the correct rotation order
        motion.col(iFrame).template segment<3>(startIndex).noalias() =
            toRad() * toEigen(values).cast<float>();
      } else if (mode == 2) {
        float scale = 1.0;
        // Warn and average if non-uniform scale
        const float kMaxScaleDiff = 1e-4;
        if (std::abs(values.x - values.y) > kMaxScaleDiff ||
            std::abs(values.x - values.z) > kMaxScaleDiff) {
          MT_LOGW(
              "Animation has non-uniform scale, which is not supported: ({}, {}, {})",
              values.x,
              values.y,
              values.z);
          scale = (values.x + values.y + values.z) / 3.0;
        } else {
          scale = values.x;
        }
        // XXX not sure how to handle layers of scaling; we don't assume to support this case for
        // now.
        motion.col(iFrame)[startIndex] = std::log2(scale);
      }
    }
  }
  return motion;
}

std::tuple<std::unique_ptr<ofbx::u8[]>, size_t> readFileToBuffer(const filesystem::path& path) {
  // The FBX SDK returns a confusing error if the file doesn't actually
  // exist, so we should trap that case and return a more helpful error instead.
  std::ifstream ifs(path.string(), std::ios::binary | std::ios::ate);
  MT_THROW_IF(!ifs.good(), "Error reading FBX file from {}", path.string());

  auto length = ifs.tellg();
  ifs.seekg(0, std::ifstream::beg);

  MT_THROW_IF(length > INT32_MAX, "File too large for OpenFBX.");

  auto buffer = std::make_unique<ofbx::u8[]>(length);
  ifs.read((char*)buffer.get(), length);
  MT_THROW_IF(!ifs.good(), "Error reading the entire FBX file from {}", path.string());

  return std::make_tuple(std::move(buffer), length);
}

std::tuple<Character, std::vector<MatrixXf>, float> loadOpenFbx(
    const gsl::span<const std::byte> fbxDataRaw,
    bool keepLocators,
    bool loadAnim,
    bool permissive) {
  auto fbxCharDataRaw = cast_span<const unsigned char>(fbxDataRaw);
  const size_t length = fbxCharDataRaw.size();
  MT_THROW_IF(length > INT32_MAX, "File too large for OpenFBX.");

  auto ofbx_deleter = [](ofbx::IScene* s) { s->destroy(); };
  // We don't currently use blend shapes for anything and they can be very
  // expensive to load. Ignore stuff in the scene that we don't support.
  auto loadFlags = ofbx::LoadFlags::IGNORE_BLEND_SHAPES | ofbx::LoadFlags::IGNORE_CAMERAS |
      ofbx::LoadFlags::IGNORE_LIGHTS | ofbx::LoadFlags::IGNORE_VIDEOS |
      ofbx::LoadFlags::IGNORE_MATERIALS;
  if (!loadAnim) {
    loadFlags |= ofbx::LoadFlags::IGNORE_ANIMATIONS;
  }
  std::unique_ptr<ofbx::IScene, decltype(ofbx_deleter)> scene(
      ofbx::load(fbxCharDataRaw.data(), (int32_t)length, (ofbx::u16)loadFlags), ofbx_deleter);
  MT_THROW_IF(!scene, "Error reading FBX scene data. Error: {}", ofbx::getError());
  MT_THROW_IF(!scene->getRoot(), "FBX scene has no root node. Error: {}", ofbx::getError());

  const auto [skeleton, jointFbxNodes, locators, collision] =
      parseSkeleton(scene->getRoot(), {}, permissive);

  TransformationList inverseBindPoseTransforms;
  for (const auto& j : jointFbxNodes) {
    Eigen::Affine3d mat = Eigen::Affine3d::Identity();
    if (j) {
      mat = toEigen(j->getGlobalTransform());
    }
    inverseBindPoseTransforms.push_back(mat.inverse().cast<float>());
  }

  Mesh mesh;
  std::unique_ptr<SkinWeights> skinWeights;

  for (int iMesh = 0; iMesh < scene->getMeshCount(); ++iMesh) {
    parseSkinnedModel(
        scene->getMesh(iMesh),
        jointFbxNodes,
        mesh,
        skinWeights,
        inverseBindPoseTransforms,
        permissive);
  }
  mesh.updateNormals();

  std::vector<MatrixXf> jointParamMotions;
  if (loadAnim) {
    for (int iAnim = 0; iAnim < scene->getAnimationStackCount(); ++iAnim) {
      const ofbx::AnimationStack* stack = scene->getAnimationStack(iAnim);
      if (stack != nullptr) {
        jointParamMotions.emplace_back(
            parseAnimation(stack, jointFbxNodes, skeleton, scene->getSceneFrameRate(), permissive));
      }
    }
  }

  Character result(
      skeleton,
      ParameterTransform::empty(skeleton.joints.size() * kParametersPerJoint),
      ParameterLimits(),
      keepLocators ? locators : LocatorList(),
      &mesh,
      skinWeights.get(),
      collision.empty() ? nullptr : &collision);
  result.resetJointMap();
  result.inverseBindPose = inverseBindPoseTransforms;

  return {result, jointParamMotions, scene->getSceneFrameRate()};
}

} // namespace

Character loadOpenFbxCharacter(
    const gsl::span<const std::byte> fbxDataRaw,
    bool keepLocators,
    bool permissive) {
  auto [character, motion, fps] = loadOpenFbx(fbxDataRaw, keepLocators, false, permissive);
  return character;
}

Character loadOpenFbxCharacter(const filesystem::path& path, bool keepLocators, bool permissive) {
  auto [buffer, length] = readFileToBuffer(path);
  return loadOpenFbxCharacter(
      gsl::as_bytes(gsl::make_span(buffer.get(), length)), keepLocators, permissive);
}

std::tuple<Character, std::vector<MatrixXf>, float> loadOpenFbxCharacterWithMotion(
    gsl::span<const std::byte> inputSpan,
    bool keepLocators,
    bool permissive) {
  return loadOpenFbx(inputSpan, keepLocators, true, permissive);
}

std::tuple<Character, std::vector<MatrixXf>, float> loadOpenFbxCharacterWithMotion(
    const filesystem::path& inputPath,
    bool keepLocators,
    bool permissive) {
  auto [buffer, length] = readFileToBuffer(inputPath);
  return loadOpenFbxCharacterWithMotion(
      gsl::as_bytes(gsl::make_span(buffer.get(), length)), keepLocators, permissive);
}

} // namespace momentum
