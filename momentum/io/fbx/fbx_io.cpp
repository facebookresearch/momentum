/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/fbx_io.h"

#include "momentum/character/character.h"
#include "momentum/character/character_state.h"
#include "momentum/character/collision_geometry_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/exception.h"
#include "momentum/common/filesystem.h"
#include "momentum/common/log.h"
#include "momentum/io/fbx/fbx_memory_stream.h"
#include "momentum/io/skeleton/locator_io.h"
#include "momentum/io/skeleton/parameter_limits_io.h"
#include "momentum/io/skeleton/parameter_transform_io.h"
#include "momentum/io/skeleton/parameters_io.h"
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/math/utility.h"

#include <fbxsdk/scene/geometry/fbxcluster.h>

// **FBX SDK**
// They do the most awful things to isnan in here
#include <fbxsdk.h>
#include <fbxsdk/fileio/fbxiosettings.h>

#ifdef isnan
#undef isnan
#endif

#include <variant>

namespace momentum {

namespace {

[[nodiscard]] ::fbxsdk::FbxAxisSystem::EUpVector toFbx(const FBXUpVector upVector) {
  switch (upVector) {
    case FBXUpVector::XAxis:
      return ::fbxsdk::FbxAxisSystem::EUpVector::eXAxis;
    case FBXUpVector::YAxis:
      return ::fbxsdk::FbxAxisSystem::EUpVector::eYAxis;
    case FBXUpVector::ZAxis:
      return ::fbxsdk::FbxAxisSystem::EUpVector::eZAxis;
    default:
      MT_THROW("Unsupported up vector");
  }
}

[[nodiscard]] ::fbxsdk::FbxAxisSystem::EFrontVector toFbx(const FBXFrontVector frontVector) {
  switch (frontVector) {
    case FBXFrontVector::ParityEven:
      return ::fbxsdk::FbxAxisSystem::EFrontVector::eParityEven;
    case FBXFrontVector::ParityOdd:
      return ::fbxsdk::FbxAxisSystem::EFrontVector::eParityOdd;
    default:
      MT_THROW("Unsupported front vector");
  }
}

[[nodiscard]] ::fbxsdk::FbxAxisSystem::ECoordSystem toFbx(const FBXCoordSystem coordSystem) {
  switch (coordSystem) {
    case FBXCoordSystem::RightHanded:
      return ::fbxsdk::FbxAxisSystem::ECoordSystem::eRightHanded;
    case FBXCoordSystem::LeftHanded:
      return ::fbxsdk::FbxAxisSystem::ECoordSystem::eLeftHanded;
    default:
      MT_THROW("Unsupported coordinate system");
  }
}

[[nodiscard]] ::fbxsdk::FbxAxisSystem toFbx(const FBXCoordSystemInfo& coordSystemInfo) {
  return {
      toFbx(coordSystemInfo.upVector),
      toFbx(coordSystemInfo.frontVector),
      toFbx(coordSystemInfo.coordSystem)};
}

Character loadFbxCommon(::fbxsdk::FbxScene* scene, bool permissive) {
  Character result;

  // make sure we're y up
  ::fbxsdk::FbxAxisSystem axis = ::fbxsdk::FbxAxisSystem(::fbxsdk::FbxAxisSystem::eMayaYUp);
  axis.ConvertScene(scene);

  // ---------------------------------------------
  // parse all nodes
  // ---------------------------------------------
  std::map<fbxsdk::FbxNode*, size_t> nodeToJointMap;
  std::list<::fbxsdk::FbxNode*> nodes;

  // add root node to list
  nodes.push_back(scene->GetRootNode());
  do {
    // get current node
    auto* node = nodes.back();
    nodes.pop_back();

    // check if it' valid
    if (node->GetNodeAttribute() == nullptr) {
      // add all child nodes
      for (int i = 0; i < node->GetChildCount(); i++)
        nodes.push_back(node->GetChild(i));
      continue;
    }

    // check for skeleton Node
    if (node->GetNodeAttribute()->GetAttributeType() == ::fbxsdk::FbxNodeAttribute::eSkeleton) {
      auto* lSkeleton = (::fbxsdk::FbxSkeleton*)node->GetNodeAttribute();

      // create a new joint and add data from the fbx file to it
      Joint joint;
      joint.name = node->GetName();
      if (lSkeleton->IsSkeletonRoot())
        joint.parent = kInvalidIndex;
      else
        joint.parent = nodeToJointMap[node->GetParent()];

      // get pre- and post-rotation
      ::fbxsdk::FbxVector4 vec;
      vec = node->GetPreRotation(::fbxsdk::FbxNode::eSourcePivot);
      joint.preRotation = eulerToQuaternion<float>(
          Vector3f(vec[0], vec[1], vec[2]) * toRad(), 0, 1, 2, EulerConvention::Extrinsic);

      vec = node->GetPostRotation(::fbxsdk::FbxNode::eSourcePivot);
      MT_THROW_IF(
          Vector3d(vec[0], vec[1], vec[2]).norm() > 1e-8,
          "Skeleton files with post-rotation are not supported. Found post rotation in joint {}",
          joint.name);

      // load offset
      ::fbxsdk::FbxDouble3 val;
      val = node->LclTranslation.Get();
      joint.translationOffset = Vector3d(val[0], val[1], val[2]).cast<float>();

      // ignore rotation/scaling values for now
      val = node->LclRotation.Get();
      val = node->LclScaling.Get();

      // get rotation order
      ::fbxsdk::FbxEuler::EOrder order;
      node->GetRotationOrder(::fbxsdk::FbxNode::eSourcePivot, order);
      MT_THROW_IF(
          order != ::fbxsdk::FbxEuler::EOrder::eOrderXYZ,
          "Skeleton files with rotation orders other than XYZ are not supported. Found unsupported order in joint {}",
          joint.name);

      // get the index of the new joint
      const size_t index = result.skeleton.joints.size();

      // add to the skeleton
      result.skeleton.joints.push_back(joint);

      // add entry to nodeToJointMap
      nodeToJointMap[node] = index;
    }
    // check for collision geometry
    else if (node->FindProperty("Col_Type", false).IsValid()) {
      TaperedCapsule tc;
      tc.parent = nodeToJointMap[node->GetParent()];
      tc.length = node->FindProperty("Length", false).Get<float>();
      tc.radius[0] = node->FindProperty("Rad_A", false).Get<float>();
      tc.radius[1] = node->FindProperty("Rad_B", false).Get<float>();
      const ::fbxsdk::FbxAMatrix mat = node->EvaluateLocalTransform();
      tc.transformation = Map<const Matrix4d>(mat.Buffer()->Buffer()).cast<float>();
      if (!result.collision)
        result.collision = std::make_unique<CollisionGeometry>();
      result.collision->push_back(tc);
    }
    // add all child nodes
    for (int i = node->GetChildCount() - 1; i >= 0; i--)
      nodes.push_back(node->GetChild(i));
  } while (!nodes.empty());

  // do the same again, loading the mesh
  nodes.push_back(scene->GetRootNode());
  do {
    // get current node
    auto* node = nodes.front();
    nodes.pop_front();

    // check if it's valid
    if (node->GetNodeAttribute() == nullptr) {
      // add all child nodes
      for (int i = 0; i < node->GetChildCount(); i++)
        nodes.push_back(node->GetChild(i));
      continue;
    } else if (node->GetNodeAttribute()->GetAttributeType() == ::fbxsdk::FbxNodeAttribute::eMesh) {
      ::fbxsdk::FbxMesh* lMesh = (::fbxsdk::FbxMesh*)node->GetNodeAttribute();

      // found mesh, need to parse it
      if (!result.mesh) {
        result.mesh = std::make_unique<Mesh>();
      }

      // get vertices
      const int numVertices = lMesh->GetControlPointsCount();
      const int numFaces = lMesh->GetPolygonCount();
      const ::fbxsdk::FbxVector4* lControlPoints = lMesh->GetControlPoints();

      const size_t voffset = result.mesh->vertices.size();
      result.mesh->vertices.resize(voffset + numVertices);
      result.mesh->normals.resize(voffset + numVertices, Vector3f::Zero());
      result.mesh->colors.resize(voffset + numVertices, Vector3b::Constant(uint8_t{0}));
      result.mesh->confidence.resize(voffset + numVertices, 1.0f);

      for (int i = 0; i < numVertices; i++)
        result.mesh->vertices[voffset + i] =
            Vector3d(lControlPoints[i][0], lControlPoints[i][1], lControlPoints[i][2])
                .cast<float>();

      // load colors if present
      auto* const colorElement = lMesh->GetElementVertexColor();
      if (colorElement != nullptr) {
        const auto mode = colorElement->GetMappingMode();
        if (mode == ::fbxsdk::FbxLayerElement::eByPolygonVertex) {
          ::fbxsdk::FbxColor fcolor;

          int count = 0;
          for (int i = 0; i < numFaces; i++) {
            const int numV = lMesh->GetPolygonSize(i);
            for (int j = 0; j < numV; j++) {
              const auto vid = lMesh->GetPolygonVertex(i, j);
              fcolor = colorElement->GetDirectArray().GetAt(count++);
              result.mesh->colors[voffset + vid] = Vector3b(
                  gsl::narrow_cast<uint8_t>(fcolor.mRed * 255.0),
                  gsl::narrow_cast<uint8_t>(fcolor.mGreen * 255.0),
                  gsl::narrow_cast<uint8_t>(fcolor.mBlue * 255.0));
            }
          }
        }
      }

      fbxsdk::FbxLayerElement::EType uvType = fbxsdk::FbxLayerElement::eTextureDiffuse;
      const size_t tcoffset = result.mesh->texcoords.size();

      int numTexCoords = 0;
      bool hasTexCoords = false;
      ::fbxsdk::FbxLayerElementArrayTemplate<::fbxsdk::FbxVector2>* lTexCoords;
      if (lMesh->GetTextureUV(&lTexCoords, uvType)) {
        numTexCoords = lMesh->GetTextureUVCount(uvType);
        hasTexCoords = numTexCoords > 0;
        result.mesh->texcoords.resize(tcoffset + numTexCoords);
        for (int i = 0; i < numTexCoords; i++) {
          ::fbxsdk::FbxVector2 vec = lTexCoords->GetAt(i);
          // flip y to "momentum convention"?
          result.mesh->texcoords[tcoffset + i] = Vector2f(vec[0], 1.0f - vec[1]);
        }
      }

      // read faces (and triangulate if necessary)
      const size_t foffset = result.mesh->faces.size();

      // we need at least numfaces storage, possibly more
      // We want to make sure faces and texcoord_faces are the same size, in case there are multiple
      // meshes, and some have texture coords and some don't
      result.mesh->faces.reserve(foffset + numFaces);
      result.mesh->texcoord_faces.reserve(foffset + numFaces);

      for (int i = 0; i < numFaces; i++) {
        const int numV = lMesh->GetPolygonSize(i);
        Vector3i face(static_cast<int>(voffset) + lMesh->GetPolygonVertex(i, 0), -1, -1);
        Vector3i tcFace(-1, -1, -1);
        if (hasTexCoords) {
          tcFace = Vector3i(static_cast<int>(tcoffset) + lMesh->GetTextureUVIndex(i, 0), -1, -1);
        }

        // loop over all vertices
        for (int j = 1; j < numV - 1; j++) {
          // update face with indices
          face[1] = static_cast<int>(voffset) + lMesh->GetPolygonVertex(i, j);
          face[2] = static_cast<int>(voffset) + lMesh->GetPolygonVertex(i, j + 1);
          if (hasTexCoords) {
            tcFace[1] = static_cast<int>(tcoffset) + lMesh->GetTextureUVIndex(i, j);
            tcFace[2] = static_cast<int>(tcoffset) + lMesh->GetTextureUVIndex(i, j + 1);
          }

          // store face
          result.mesh->faces.push_back(face);

          // store tc face
          result.mesh->texcoord_faces.push_back(tcFace);
        }
      }

      // also load skinning data if present
      ::fbxsdk::FbxSkin* fbxskin =
          (::fbxsdk::FbxSkin*)lMesh->GetDeformer(0, ::fbxsdk::FbxDeformer::eSkin);
      // Create skin weights unconditionally, or only when FbxSkin is present in permissive mode.
      if (!result.skinWeights && (!permissive || fbxskin != nullptr)) {
        result.skinWeights = std::make_unique<SkinWeights>();
      }
      if (fbxskin != nullptr) {
        int clusterCount = fbxskin->GetClusterCount();
        // resize arrays
        result.skinWeights->index.conservativeResize(voffset + numVertices, Eigen::NoChange);
        result.skinWeights->weight.conservativeResize(voffset + numVertices, Eigen::NoChange);
        result.skinWeights->index.bottomRows(numVertices).setZero();
        result.skinWeights->weight.bottomRows(numVertices).setZero();

        for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++) {
          ::fbxsdk::FbxCluster* cluster = fbxskin->GetCluster(clusterIndex);
          ::fbxsdk::FbxNode* bone = cluster->GetLink();
          const size_t boneIndex = nodeToJointMap[bone];

          int* boneVertexIndices = cluster->GetControlPointIndices();
          double* boneVertexWeights = cluster->GetControlPointWeights();

          // iterate through all the vertices, which are affected by the bone
          int numBoneVertexIndices = cluster->GetControlPointIndicesCount();
          for (int boneVIndex = 0; boneVIndex < numBoneVertexIndices; boneVIndex++) {
            const int boneVertexIndex = boneVertexIndices[boneVIndex];
            const float boneWeight = static_cast<float>(boneVertexWeights[boneVIndex]);

            MT_CHECK(
                boneVertexIndex >= 0 && boneVertexIndex < numVertices,
                "{}: {}",
                boneVertexIndex,
                numVertices);

            const size_t bvi = boneVertexIndex + voffset;
            size_t insertAt = kInvalidIndex;
            for (size_t l = 0; l < kMaxSkinJoints; l++) {
              if (result.skinWeights->weight(bvi, l) < boneWeight) {
                insertAt = l;
                break;
              }
            }

            if (insertAt != kInvalidIndex) {
              if (insertAt < kMaxSkinJoints - 1) {
                auto t1 =
                    result.skinWeights->index.block(bvi, insertAt, 1, kMaxSkinJoints - insertAt - 1)
                        .eval();
                result.skinWeights->index.block(
                    bvi, insertAt + 1, 1, kMaxSkinJoints - insertAt - 1) = t1;
                auto t2 = result.skinWeights->weight
                              .block(bvi, insertAt, 1, kMaxSkinJoints - insertAt - 1)
                              .eval();
                result.skinWeights->weight.block(
                    bvi, insertAt + 1, 1, kMaxSkinJoints - insertAt - 1) = t2;
              }
              result.skinWeights->index(bvi, insertAt) = gsl::narrow<uint32_t>(boneIndex);
              result.skinWeights->weight(bvi, insertAt) = boneWeight;
            }
          }
        }

        for (int i = 0; i < numVertices; i++) {
          const int bvi = i + gsl::narrow_cast<int>(voffset);
          result.skinWeights->weight.row(bvi) /= result.skinWeights->weight.row(bvi).sum();
        }
      }
    }

    // add all child nodes
    for (int i = 0; i < node->GetChildCount(); i++)
      nodes.push_back(node->GetChild(i));
  } while (!nodes.empty());

  if (result.mesh)
    result.mesh->updateNormals();

  return result;
}

// Loads fbx scene either from a filepath or from an input buffer
Character loadFbxScene(
    const std::variant<filesystem::path, gsl::span<const std::byte>>& input,
    bool permissive) {
  // ---------------------------------------------
  // initialize FBX SDK and load data
  // ---------------------------------------------
  auto* manager = ::fbxsdk::FbxManager::Create();
  auto* ios = ::fbxsdk::FbxIOSettings::Create(manager, IOSROOT);
  manager->SetIOSettings(ios);

  std::unique_ptr<FbxStream> custom_stream; // needs to outlive importer
  auto* importer = ::fbxsdk::FbxImporter::Create(manager, "");
  bool initialized = false;

  std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, filesystem::path>) {
          std::string sInput = arg.string();
          initialized = importer->Initialize(sInput.c_str(), -1, manager->GetIOSettings());
        } else if constexpr (std::is_same_v<T, gsl::span<const std::byte>>) {
          FbxIOPluginRegistry* reg = manager->GetIOPluginRegistry();
          custom_stream =
              std::make_unique<FbxMemoryStream>(arg, reg->FindReaderIDByExtension("fbx"));
          initialized =
              importer->Initialize(custom_stream.get(), nullptr, -1, manager->GetIOSettings());
        }
      },
      input);

  // should never get here
  MT_THROW_IF(
      !initialized, "Unable to initialize fbx importer {}", importer->GetStatus().GetErrorString());

  auto* scene = ::fbxsdk::FbxScene::Create(manager, "myScene");
  importer->Import(scene);
  importer->Destroy();

  Character result = loadFbxCommon(scene, permissive);

  if (scene != nullptr)
    scene->Destroy();
  manager->Destroy();

  // Finalize Character
  result.parameterTransform =
      ParameterTransform::empty(kParametersPerJoint * result.skeleton.joints.size()),
  result.resetJointMap();
  result.initInverseBindPose();

  return result;
}

void createLocatorNodes(
    const Character& character,
    ::fbxsdk::FbxScene* scene,
    const std::vector<::fbxsdk::FbxNode*>& skeletonNodes) {
  for (const auto& loc : character.locators) {
    ::fbxsdk::FbxMarker* markerAttribute = ::fbxsdk::FbxMarker::Create(scene, loc.name.c_str());
    markerAttribute->Look.Set(::fbxsdk::FbxMarker::ELook::eHardCross);

    // create the node
    ::fbxsdk::FbxNode* locatorNode = ::fbxsdk::FbxNode::Create(scene, loc.name.c_str());
    locatorNode->SetNodeAttribute(markerAttribute);

    // set translation offset
    locatorNode->LclTranslation.Set(FbxVector4(loc.offset[0], loc.offset[1], loc.offset[2]));

    // set parent if it has one
    if (loc.parent != kInvalidIndex)
      skeletonNodes[loc.parent]->AddChild(locatorNode);
  }
}

void createCollisionGeometryNodes(
    const Character& character,
    ::fbxsdk::FbxScene* scene,
    const std::vector<::fbxsdk::FbxNode*>& skeletonNodes) {
  if (!character.collision) {
    MT_LOGD(
        "No collision geometry found in character, skipping creation of collision geometry nodes");
    return;
  }

  const auto& collisions = *character.collision;
  for (auto i = 0u; i < collisions.size(); ++i) {
    const TaperedCapsule& collision = collisions[i];

    ::fbxsdk::FbxNode* collisionNode =
        ::fbxsdk::FbxNode::Create(scene, ("Collision " + std::to_string(i)).c_str());
    auto* nullNodeAttr =
        ::fbxsdk::FbxNull::Create(scene, "Null"); // TODO: Find a good node attribute
    collisionNode->SetNodeAttribute(nullNodeAttr);

    ::fbxsdk::FbxProperty::Create(collisionNode, ::fbxsdk::FbxBoolDT, "Col_Type").Set(true);
    ::fbxsdk::FbxProperty::Create(collisionNode, ::fbxsdk::FbxFloatDT, "Length")
        .Set(collision.length);
    ::fbxsdk::FbxProperty::Create(collisionNode, ::fbxsdk::FbxFloatDT, "Rad_A")
        .Set(collision.radius[0]);
    ::fbxsdk::FbxProperty::Create(collisionNode, ::fbxsdk::FbxFloatDT, "Rad_B")
        .Set(collision.radius[1]);

    collisionNode->LclTranslation.Set(FbxVector4(
        collision.transformation.translation().x(),
        collision.transformation.translation().y(),
        collision.transformation.translation().z()));
    const Vector3f rot = rotationMatrixToEulerXYZ<float>(
        collision.transformation.rotation(), EulerConvention::Extrinsic);
    collisionNode->LclRotation.Set(FbxDouble3(toDeg(rot.x()), toDeg(rot.y()), toDeg(rot.z())));
    collisionNode->LclScaling.Set(FbxDouble3(1));

    if (collision.parent != kInvalidIndex) {
      skeletonNodes[collision.parent]->AddChild(collisionNode);
    } else {
      MT_LOGE("Found a collision node with no parent");
    }
  }
}

void setFrameRate(::fbxsdk::FbxScene* scene, const double framerate) {
  // enumerate common frame rates first, then resort to custom framerate
  if (std::abs(framerate - 30.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames30);
  } else if (std::abs(framerate - 24.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames24);
  } else if (std::abs(framerate - 48.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames48);
  } else if (std::abs(framerate - 50.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames50);
  } else if (std::abs(framerate - 60.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames60);
  } else if (std::abs(framerate - 72.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames72);
  } else if (std::abs(framerate - 96.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames96);
  } else if (std::abs(framerate - 100.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames100);
  } else if (std::abs(framerate - 120.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames120);
  } else if (std::abs(framerate - 1000.0) < 1e-6) {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eFrames1000);
  } else {
    scene->GetGlobalSettings().SetTimeMode(::fbxsdk::FbxTime::eCustom);
    scene->GetGlobalSettings().SetCustomFrameRate(framerate);
  }
}

// jointValues: (numJointParameters x numFrames) matrix of joint values
void createAnimationCurves(
    const Character& character,
    ::fbxsdk::FbxScene* scene,
    const std::vector<::fbxsdk::FbxNode*>& skeletonNodes,
    const MatrixXf& jointValues,
    const double framerate,
    const bool skipActiveJointParamCheck) {
  // set the framerate
  setFrameRate(scene, framerate);

  const auto& aj = character.parameterTransform.activeJointParams;

  // create animation stack
  ::fbxsdk::FbxAnimStack* animStack =
      ::fbxsdk::FbxAnimStack::Create(scene, "Skeleton Animation Stack");
  ::fbxsdk::FbxAnimLayer* animBaseLayer = ::fbxsdk::FbxAnimLayer::Create(scene, "Layer0");
  animStack->AddMember(animBaseLayer);

  // create anim curves for each joint and store them in an array
  std::vector<::fbxsdk::FbxAnimCurve*> animCurves(character.skeleton.joints.size() * 9, nullptr);
  std::vector<size_t> animCurvesIndex;
  for (size_t i = 0; i < character.skeleton.joints.size(); i++) {
    const size_t jointIndex = i * kParametersPerJoint;
    const size_t index = i * 9;
    skeletonNodes[i]->LclTranslation.GetCurveNode(true);
    if (skipActiveJointParamCheck || aj[jointIndex + 0]) {
      animCurves[index + 0] = skeletonNodes[i]->LclTranslation.GetCurve(
          animBaseLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
      animCurvesIndex.push_back(index + 0);
    }
    if (skipActiveJointParamCheck || aj[jointIndex + 1]) {
      animCurves[index + 1] = skeletonNodes[i]->LclTranslation.GetCurve(
          animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
      animCurvesIndex.push_back(index + 1);
    }
    if (skipActiveJointParamCheck || aj[jointIndex + 2]) {
      animCurves[index + 2] = skeletonNodes[i]->LclTranslation.GetCurve(
          animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);
      animCurvesIndex.push_back(index + 2);
    }
    skeletonNodes[i]->LclRotation.GetCurveNode(true);
    if (skipActiveJointParamCheck || aj[jointIndex + 3]) {
      animCurves[index + 3] =
          skeletonNodes[i]->LclRotation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
      animCurvesIndex.push_back(index + 3);
    }
    if (skipActiveJointParamCheck || aj[jointIndex + 4]) {
      animCurves[index + 4] =
          skeletonNodes[i]->LclRotation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
      animCurvesIndex.push_back(index + 4);
    }
    if (skipActiveJointParamCheck || aj[jointIndex + 5]) {
      animCurves[index + 5] =
          skeletonNodes[i]->LclRotation.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);
      animCurvesIndex.push_back(index + 5);
    }
    skeletonNodes[i]->LclScaling.GetCurveNode(true);
    if (skipActiveJointParamCheck || aj[jointIndex + 6]) {
      animCurves[index + 6] =
          skeletonNodes[i]->LclScaling.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
      animCurvesIndex.push_back(index + 6);
      animCurves[index + 7] =
          skeletonNodes[i]->LclScaling.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
      animCurvesIndex.push_back(index + 7);
      animCurves[index + 8] =
          skeletonNodes[i]->LclScaling.GetCurve(animBaseLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);
      animCurvesIndex.push_back(index + 8);
    }
  }

  // calculate the actual motion and set the keyframes
  ::fbxsdk::FbxTime time;
  // now go over each animCurveIndex and generate the curve
  for (const auto ai : animCurvesIndex) {
    const size_t jointIndex = ai / 9;
    const size_t jointOffset = ai % 9;
    const size_t parameterIndex =
        jointIndex * kParametersPerJoint + std::min(jointOffset, size_t(6));
    if (!skipActiveJointParamCheck && aj[parameterIndex] == 0)
      continue;

    animCurves[ai]->KeyModifyBegin();
    for (size_t f = 0; f < jointValues.cols(); f++) {
      // set keyframe time
      time.SetSecondDouble(static_cast<double>(f) / framerate);

      // get joint value
      float jointVal = jointValues(parameterIndex, f);

      // add translation offset for tx values
      if (jointOffset < 3)
        jointVal += character.skeleton.joints[jointIndex].translationOffset[jointOffset];
      // convert to degrees
      else if (jointOffset >= 3 && jointOffset <= 5)
        jointVal = toDeg(jointVal);
      // convert to non-exponential scaling
      else
        jointVal = std::pow(2.0f, jointVal);

      const auto keyIndex = animCurves[ai]->KeyAdd(time);
      animCurves[ai]->KeySet(keyIndex, time, jointVal);
    }
    animCurves[ai]->KeyModifyEnd();
  }
}

void saveFbxCommon(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& jointValues,
    const double framerate,
    const bool saveMesh,
    const bool skipActiveJointParamCheck,
    const FBXCoordSystemInfo& coordSystemInfo,
    bool permissive) {
  // ---------------------------------------------
  // initialize FBX SDK and prepare for export
  // ---------------------------------------------
  auto* manager = ::fbxsdk::FbxManager::Create();
  auto* ios = ::fbxsdk::FbxIOSettings::Create(manager, IOSROOT);
  manager->SetIOSettings(ios);

  // Create an exporter.
  ::fbxsdk::FbxExporter* lExporter = ::fbxsdk::FbxExporter::Create(manager, "");

  // Declare the path and filename of the file containing the scene.
  // In this case, we are assuming the file is in the same directory as the executable.
  // Going through string() because on windows, wchar_t (native filesystem path) are different from
  // char https://en.cppreference.com/w/cpp/language/types This avoids a build error on windows
  // only.
  std::string sFilename = filename.string();
  const char* lFilename = sFilename.c_str();

  // Initialize the exporter.
  bool lExportStatus = lExporter->Initialize(lFilename, -1, manager->GetIOSettings());

  MT_THROW_IF(
      !lExportStatus,
      "Unable to initialize fbx exporter {}",
      lExporter->GetStatus().GetErrorString());

  // ---------------------------------------------
  // create the scene
  // ---------------------------------------------
  ::fbxsdk::FbxScene* scene = ::fbxsdk::FbxScene::Create(manager, "momentum_scene");
  ::fbxsdk::FbxNode* root = scene->GetRootNode();

  // set the coordinate system
  ::fbxsdk::FbxAxisSystem axis = toFbx(coordSystemInfo);
  axis.ConvertScene(scene);

  // ---------------------------------------------
  // create the skeleton nodes
  // ---------------------------------------------
  std::vector<::fbxsdk::FbxNode*> skeletonNodes;
  std::unordered_map<size_t, fbxsdk::FbxNode*> jointToNodeMap;

  ::fbxsdk::FbxNode* skeletonRootNode;

  for (size_t i = 0; i < character.skeleton.joints.size(); i++) {
    const auto& joint = character.skeleton.joints[i];

    // create the node
    ::fbxsdk::FbxNode* skeletonNode = ::fbxsdk::FbxNode::Create(scene, joint.name.c_str());
    // create node attribute
    ::fbxsdk::FbxSkeleton* skeletonAttribute =
        ::fbxsdk::FbxSkeleton::Create(scene, joint.name.c_str());

    if (joint.parent == kInvalidIndex) {
      skeletonRootNode = skeletonNode;
      skeletonAttribute->SetSkeletonType(::fbxsdk::FbxSkeleton::eRoot);
    } else {
      skeletonAttribute->SetSkeletonType(::fbxsdk::FbxSkeleton::eLimbNode);
    }
    skeletonNode->SetNodeAttribute(skeletonAttribute);
    jointToNodeMap[i] = skeletonNode;

    // set translation offset
    skeletonNode->LclTranslation.Set(FbxDouble3(
        joint.translationOffset[0], joint.translationOffset[1], joint.translationOffset[2]));

    // set pre-rotation
    const auto angles = rotationMatrixToEulerZYX(joint.preRotation.toRotationMatrix());
    skeletonNode->SetPivotState(FbxNode::eSourcePivot, FbxNode::ePivotActive);
    skeletonNode->SetRotationActive(true);
    skeletonNode->SetPreRotation(
        ::fbxsdk::FbxNode::eSourcePivot,
        FbxDouble3(toDeg(angles[2]), toDeg(angles[1]), toDeg(angles[0])));

    // add to list
    skeletonNodes.emplace_back(skeletonNode);
  }

  // Second pass: handle the parenting, in case the parents are not in order
  for (size_t i = 0; i < character.skeleton.joints.size(); i++) {
    const auto& joint = character.skeleton.joints[i];

    // set parent if it has one
    auto* skeletonNode = jointToNodeMap[i];
    if (joint.parent != kInvalidIndex) {
      skeletonNodes[joint.parent]->AddChild(skeletonNode);
    }
  }

  // ---------------------------------------------
  // create the locator nodes
  // ---------------------------------------------
  createLocatorNodes(character, scene, skeletonNodes);

  // ---------------------------------------------
  // create the collision geometry nodes
  // ---------------------------------------------
  createCollisionGeometryNodes(character, scene, skeletonNodes);

  // ---------------------------------------------
  // create the mesh nodes
  // ---------------------------------------------
  if (saveMesh && character.mesh != nullptr) {
    // Add the mesh
    const int numVertices = character.mesh.get()->vertices.size();
    const int numFaces = character.mesh.get()->faces.size();
    ::fbxsdk::FbxNode* meshNode = ::fbxsdk::FbxNode::Create(scene, "body_mesh");
    ::fbxsdk::FbxMesh* lMesh = ::fbxsdk::FbxMesh::Create(scene, "mesh");
    lMesh->SetControlPointCount(numVertices);
    lMesh->InitNormals(numVertices);
    for (int i = 0; i < numVertices; i++) {
      FbxVector4 point(
          character.mesh.get()->vertices[i].x(),
          character.mesh.get()->vertices[i].y(),
          character.mesh.get()->vertices[i].z());
      FbxVector4 normal(
          character.mesh.get()->normals[i].x(),
          character.mesh.get()->normals[i].y(),
          character.mesh.get()->normals[i].z());
      lMesh->SetControlPointAt(point, normal, i);
    }
    // Add polygons to lMesh
    for (int iFace = 0; iFace < numFaces; iFace++) {
      lMesh->BeginPolygon();
      for (int i = 0; i < 3; i++) { // We have tris for models. This could be extended for
                                    // supporting Quads or npoly if needed.
        lMesh->AddPolygon(character.mesh.get()->faces[iFace][i]);
      }
      lMesh->EndPolygon();
    }
    lMesh->BuildMeshEdgeArray();
    meshNode->SetNodeAttribute(lMesh);

    // ---------------------------------------------
    // add texture coordinates
    // ---------------------------------------------
    if (!character.mesh->texcoords.empty()) {
      const fbxsdk::FbxLayerElement::EType uvType = fbxsdk::FbxLayerElement::eTextureDiffuse;

      // Initialize UV set and indices first. Both functions must be called before adding UVs
      // and UV indices.
      lMesh->InitTextureUV(0, uvType);
      lMesh->InitTextureUVIndices(
          ::fbxsdk::FbxLayerElement::EMappingMode::eByPolygonVertex, uvType);

      // Add UVs
      for (const auto& texcoords : character.mesh->texcoords) {
        // flip y back to fbx convention - refer to reading code
        lMesh->AddTextureUV(::fbxsdk::FbxVector2(texcoords[0], 1.0f - texcoords[1]), uvType);
      }

      // Set UV indices for each face. We only have triangles.
      int faceCount = 0;
      for (const auto& texcoords : character.mesh->texcoord_faces) {
        lMesh->SetTextureUVIndex(faceCount, 0, texcoords[0], uvType);
        lMesh->SetTextureUVIndex(faceCount, 1, texcoords[1], uvType);
        lMesh->SetTextureUVIndex(faceCount++, 2, texcoords[2], uvType);
      }
    }

    // ---------------------------------------------
    // create the skinning
    // ---------------------------------------------
    // Add the mesh skinning
    // Momentum skinning is saved in two matrices: index and weight (size numvertices x
    // not-ordered-joints). The index contains the joint index and the weight is the normalized
    // weight the vertex for that joint.

    MT_THROW_IF(
        !permissive && !character.skinWeights,
        " Failed to save the character '{}' to {}. The character has no skinning weights and permissive mode is not enabled. Only mesh-only characters are allowed in permissive mode.",
        character.name,
        filename.string());

    if (character.skinWeights != nullptr) {
      ::fbxsdk::FbxSkin* fbxskin = ::fbxsdk::FbxSkin::Create(scene, "meshskinning");
      fbxskin->SetSkinningType(::fbxsdk::FbxSkin::eLinear);
      fbxskin->SetGeometry(lMesh);
      FbxAMatrix meshTransform;
      meshTransform.SetIdentity();
      for (const auto& jointNode : jointToNodeMap) {
        size_t jointIdx = jointNode.first;
        auto* fbxJointNode = jointNode.second;

        std::ostringstream s;
        s << "skinningcluster_" << jointIdx;
        FbxCluster* pCluster = ::fbxsdk::FbxCluster::Create(scene, s.str().c_str());
        pCluster->SetLinkMode(::fbxsdk::FbxCluster::ELinkMode::eNormalize);
        pCluster->SetLink(fbxJointNode);

        ::fbxsdk::FbxAMatrix globalMatrix = fbxJointNode->EvaluateLocalTransform();
        ::fbxsdk::FbxNode* pParent = fbxJointNode->GetParent();
        // TODO: should use inverse bind transform from character instead.
        while (pParent != nullptr) {
          globalMatrix = pParent->EvaluateLocalTransform() * globalMatrix;
          pParent = pParent->GetParent();
        }
        pCluster->SetTransformLinkMatrix(globalMatrix);
        pCluster->SetTransformMatrix(meshTransform);

        for (int i = 0; i < character.skinWeights->index.rows(); i++) {
          for (int j = 0; j < character.skinWeights->index.cols(); j++) {
            auto boneIndex = character.skinWeights->index(i, j);
            if (boneIndex == jointNode.first && character.skinWeights->weight(i, j) > 0) {
              pCluster->AddControlPointIndex(i, character.skinWeights->weight(i, j));
            }
          }
        }
        fbxskin->AddCluster(pCluster);
      }
      lMesh->AddDeformer(fbxskin);
    }
    // Add the mesh under the root
    root->AddChild(meshNode);
  }

  // ---------------------------------------------
  // add the skeleton to the root
  // ---------------------------------------------
  if (!skeletonNodes.empty()) {
    root->AddChild(skeletonRootNode);
  }

  // ---------------------------------------------
  // create animation curves if we have motion
  // ---------------------------------------------
  if (jointValues.cols() != 0) {
    if (jointValues.rows() == character.parameterTransform.numJointParameters()) {
      createAnimationCurves(
          character, scene, skeletonNodes, jointValues, framerate, skipActiveJointParamCheck);
    } else {
      MT_LOGE(
          "Rows of joint values {} do not match joint parameter dimension {} so not saving any motion.",
          jointValues.rows(),
          character.parameterTransform.numJointParameters());
    }
  }

  // ---------------------------------------------
  // close the fbx exporter
  // ---------------------------------------------

  // finally export the scene
  lExporter->Export(scene);
  lExporter->Destroy();

  // destroy the scene and the manager
  if (scene != nullptr)
    scene->Destroy();
  manager->Destroy();
}

} // namespace

Character loadFbxCharacter(const filesystem::path& inputPath, bool permissive) {
  return loadFbxScene(inputPath, permissive);
}

Character loadFbxCharacter(gsl::span<const std::byte> inputSpan, bool permissive) {
  return loadFbxScene(inputSpan, permissive);
}

void saveFbx(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& poses, // model parameters
    const VectorXf& identity,
    const double framerate,
    const bool saveMesh,
    const FBXCoordSystemInfo& coordSystemInfo,
    bool permissive) {
  CharacterParameters params;
  if (identity.size() == character.parameterTransform.numJointParameters()) {
    params.offsets = identity;
  } else {
    params.offsets = character.parameterTransform.bindPose();
  }

  // first convert model parameters to joint values
  CharacterState state;
  MatrixXf jointValues;
  if (poses.cols() > 0) {
    // Set the initial pose to initialize the state
    params.pose = poses.col(0);
    state.set(params, character, false, false, false);

    // Resize the jointValues matrix based on the size of joint parameters and number of poses
    jointValues.resize(state.skeletonState.jointParameters.v.size(), poses.cols());

    // Store the joint parameters for the initial pose
    jointValues.col(0) = state.skeletonState.jointParameters.v;

    // Iterate through each subsequent pose
    for (Eigen::Index f = 1; f < poses.cols(); f++) {
      // set the current pose
      params.pose = poses.col(f);
      state.set(params, character, false, false, false);
      jointValues.col(f) = state.skeletonState.jointParameters.v;
    }
  }

  // Call the helper function to save FBX file with joint values
  saveFbxCommon(
      filename, character, jointValues, framerate, saveMesh, false, coordSystemInfo, permissive);
}

void saveFbxWithJointParams(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& jointParams,
    const double framerate,
    const bool saveMesh,
    const FBXCoordSystemInfo& coordSystemInfo,
    bool permissive) {
  // Call the helper function to save FBX file with joint values.
  // Set skipActiveJointParamCheck=true to skip the active joint param check as the joint params are
  // passed in directly from user.
  saveFbxCommon(
      filename, character, jointParams, framerate, saveMesh, true, coordSystemInfo, permissive);
}

void saveFbxModel(
    const filesystem::path& filename,
    const Character& character,
    const FBXCoordSystemInfo& coordSystemInfo,
    bool permissive) {
  saveFbx(filename, character, MatrixXf(), VectorXf(), 120.0, true, coordSystemInfo, permissive);
}

} // namespace momentum
