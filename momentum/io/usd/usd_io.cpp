/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/usd/usd_io.h"

#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character/types.h"
#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/math/mesh.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnosticMgr.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/base/vt/array.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdSkel/bindingAPI.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/skeleton.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/skinningQuery.h>

#include <tbb/global_control.h>

// TBB version compatibility:
// - TBB 2019 and older: uses task_scheduler_init.h
// - TBB 2020.3 and newer: uses global_control.h only (task_scheduler_init is deprecated)
// - TBB 2021 and newer: task_scheduler_init.h completely removed
#ifdef TBB_INTERFACE_VERSION
#if TBB_INTERFACE_VERSION < 12000 // TBB 2020.3 has interface version 12000
#include <tbb/task_scheduler_init.h>
#define MOMENTUM_USE_TBB_TASK_SCHEDULER_INIT 1
#endif
#else
// Try to include the header and define if successful
#if __has_include(<tbb/task_scheduler_init.h>)
#include <tbb/task_scheduler_init.h>
#define MOMENTUM_USE_TBB_TASK_SCHEDULER_INIT 1
#endif
#endif

// Conditional include for internal Meta environment vs open source
// In open source builds, this header won't exist and MOMENTUM_WITH_USD_PLUGIN_INIT will be
// undefined
#ifdef MOMENTUM_WITH_USD_PLUGIN_INIT
#include <UsdPluginInit.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Import USD namespace
PXR_NAMESPACE_USING_DIRECTIVE

namespace momentum {

namespace {

class ResolverWarningsSuppressor : public TfDiagnosticMgr::Delegate {
 public:
  void IssueError(TfError const& err) override {
    const std::string& msg = err.GetCommentary();
    if (msg.find("Failed to manufacture asset resolver") != std::string::npos) {
      return;
    }
    std::cerr << "USD Error: " << msg << std::endl;
  }

  void IssueWarning(TfWarning const& warning) override {
    const std::string& msg = warning.GetCommentary();
    if (msg.find("Failed to manufacture asset resolver") != std::string::npos) {
      return;
    }
    std::cerr << "USD Warning: " << msg << std::endl;
  }

  void IssueStatus(TfStatus const& status) override {
    std::cerr << "USD Status: " << status.GetCommentary() << std::endl;
  }

  void IssueFatalError(TfCallContext const& /*context*/, std::string const& msg) override {
    std::cerr << "USD Fatal Error: " << msg << std::endl;
  }
};

std::mutex g_usdInitMutex;
std::mutex g_usdOperationMutex;
bool g_usdInitialized = false;
std::unique_ptr<ResolverWarningsSuppressor> g_suppressor;
#ifdef MOMENTUM_WITH_USD_PLUGIN_INIT
std::unique_ptr<UsdPluginInit> g_usdPluginInit;
#endif
std::unique_ptr<tbb::global_control> g_tbbControl;

// TBB compatibility variables for older versions
#ifdef MOMENTUM_USE_TBB_TASK_SCHEDULER_INIT
std::unique_ptr<tbb::task_scheduler_init> g_tbbTaskScheduler;
#endif

void initializeUsdWithSuppressedWarnings() {
  std::lock_guard<std::mutex> lock(g_usdInitMutex);

  if (g_usdInitialized) {
    return;
  }

  // Initialize TBB with the appropriate method based on version
#ifdef MOMENTUM_USE_TBB_TASK_SCHEDULER_INIT
  // Use legacy task_scheduler_init for older TBB versions
  g_tbbTaskScheduler = std::make_unique<tbb::task_scheduler_init>(1);
#else
  // Use modern global_control for newer TBB versions
  g_tbbControl =
      std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, 1);
#endif

  g_suppressor = std::make_unique<ResolverWarningsSuppressor>();
  TfDiagnosticMgr::GetInstance().AddDelegate(g_suppressor.get());

#ifdef MOMENTUM_WITH_USD_PLUGIN_INIT
  // Internal Meta environment: Use UsdPluginInit for embedded plugins
  auto tempDir = filesystem::temp_directory_path();

  // Try a few fixed paths to avoid accumulating many plugin folders
  std::vector<std::string> fixedPaths = {"usd_plugin", "usd_momentum_plugin"};

  bool pluginDirCreated = false;
  filesystem::path pluginDir;

  for (const auto& pathName : fixedPaths) {
    pluginDir = tempDir / pathName;
    std::error_code ec;
    // Use create_directory (not create_directories) to fail if directory exists
    if (filesystem::create_directory(pluginDir, ec) && !ec) {
      pluginDirCreated = true;
      break;
    }
  }

  if (pluginDirCreated) {
    g_usdPluginInit = std::make_unique<UsdPluginInit>(pluginDir);
  } else {
    g_usdPluginInit = std::make_unique<UsdPluginInit>();
  }
#else
  // Open source environment: USD is already initialized via standard discovery
  // The pxr package config automatically handles plugin paths via PXR_PLUGINPATH_NAME
  MT_LOGI("USD I/O initialized with system USD installation");
#endif

  g_usdInitialized = true;
}

Eigen::Matrix4f toEigenMatrix4f(const GfMatrix4d& gfMatrix) {
  return Eigen::Map<const Eigen::Matrix4d>(gfMatrix.GetArray()).cast<float>();
}

GfMatrix4d toGfMatrix4d(const Eigen::Matrix4f& eigenMatrix) {
  GfMatrix4d gfMatrix{};
  Eigen::Map<Eigen::Matrix4d>(gfMatrix.GetArray()) = eigenMatrix.cast<double>();
  return gfMatrix;
}

/// Load skeleton from USD stage
Skeleton loadSkeletonFromUsd(const UsdStageRefPtr& stage) {
  Skeleton skeleton;

  // Use USD Skeleton Cache for proper hierarchy handling
  UsdSkelCache skelCache;

  // Find skeleton prims
  for (const auto& prim : stage->Traverse()) {
    if (prim.IsA<UsdSkelSkeleton>()) {
      UsdSkelSkeleton skelPrim(prim);

      // Create skeleton query for proper data access
      UsdSkelSkeletonQuery skelQuery = skelCache.GetSkelQuery(skelPrim);
      if (!skelQuery) {
        continue;
      }

      // Get joint names and topology
      VtArray<TfToken> jointNames = skelQuery.GetJointOrder();
      if (jointNames.empty()) {
        continue;
      }

      skeleton.joints.reserve(jointNames.size());

      // Build joint hierarchy using USD's topology
      VtArray<int> topology = skelQuery.GetTopology().GetParentIndices();

      for (size_t i = 0; i < jointNames.size(); ++i) {
        Joint joint;
        joint.name = jointNames[i].GetString();

        // Set parent from topology
        if (i < topology.size() && topology[i] >= 0) {
          joint.parent = topology[i];
        } else {
          joint.parent = kInvalidIndex;
        }

        joint.preRotation = Eigen::Quaternionf::Identity();
        joint.translationOffset = Eigen::Vector3f::Zero();

        skeleton.joints.push_back(joint);
      }

      // Get bind transforms (rest pose)
      VtArray<GfMatrix4d> bindTransforms;
      if (skelQuery.GetJointWorldBindTransforms(&bindTransforms)) {
        MT_CHECK(bindTransforms.size() == skeleton.joints.size());

        for (size_t i = 0; i < bindTransforms.size(); ++i) {
          auto matrix = toEigenMatrix4f(bindTransforms[i]);

          // Extract translation
          skeleton.joints[i].translationOffset = matrix.block<3, 1>(0, 3);

          // Extract rotation (simplified - could be improved)
          Eigen::Matrix3f rotMatrix = matrix.block<3, 3>(0, 0);
          Eigen::Quaternionf quat(rotMatrix);
          skeleton.joints[i].preRotation = quat.normalized();
        }
      }

      break; // Use first skeleton found
    }
  }

  return skeleton;
}

/// Load mesh from USD stage
Mesh loadMeshFromUsd(const UsdStageRefPtr& stage) {
  Mesh mesh;

  // Find mesh prims
  for (const auto& prim : stage->Traverse()) {
    if (prim.IsA<UsdGeomMesh>()) {
      UsdGeomMesh meshPrim(prim);

      // Get vertices
      VtArray<GfVec3f> points;
      if (meshPrim.GetPointsAttr().Get(&points)) {
        mesh.vertices.reserve(points.size());
        for (const auto& point : points) {
          mesh.vertices.emplace_back(point[0], point[1], point[2]);
        }
      }

      // Try to load vertex colors from primvars
      UsdGeomPrimvarsAPI primvarsAPI(meshPrim);

      // Common color primvar names to check
      std::vector<std::string> colorPrimvarNames = {
          "displayColor", "Cd", "color", "vertexColor", "diffuseColor"};

      for (const auto& colorName : colorPrimvarNames) {
        UsdGeomPrimvar colorPrimvar = primvarsAPI.GetPrimvar(TfToken(colorName));
        if (colorPrimvar && colorPrimvar.HasValue()) {
          VtValue colorValue;
          if (colorPrimvar.Get(&colorValue)) {
            // Handle different color formats
            if (colorValue.IsHolding<VtArray<GfVec3f>>()) {
              VtArray<GfVec3f> colors = colorValue.Get<VtArray<GfVec3f>>();
              if (colors.size() == points.size()) {
                mesh.colors.reserve(colors.size());
                for (const auto& color : colors) {
                  mesh.colors.emplace_back(color[0], color[1], color[2]);
                }
                break; // Found valid colors, stop searching
              }
            } else if (colorValue.IsHolding<VtArray<GfVec4f>>()) {
              VtArray<GfVec4f> colors = colorValue.Get<VtArray<GfVec4f>>();
              if (colors.size() == points.size()) {
                mesh.colors.reserve(colors.size());
                for (const auto& color : colors) {
                  mesh.colors.emplace_back(color[0], color[1], color[2]); // Ignore alpha
                }
                break; // Found valid colors, stop searching
              }
            }
          }
        }
      }

      // Get face vertex counts and indices
      VtArray<int> faceVertexCounts;
      VtArray<int> faceVertexIndices;

      if (meshPrim.GetFaceVertexCountsAttr().Get(&faceVertexCounts) &&
          meshPrim.GetFaceVertexIndicesAttr().Get(&faceVertexIndices)) {
        // Check if all faces are triangles (most common case)
        bool allTriangles = true;
        for (int count : faceVertexCounts) {
          if (count != 3) {
            allTriangles = false;
            break;
          }
        }

        if (allTriangles && faceVertexIndices.size() == faceVertexCounts.size() * 3) {
          // Fast path for triangulated meshes
          mesh.faces.reserve(faceVertexCounts.size());
          const int* indices = faceVertexIndices.cdata();
          for (size_t i = 0; i < faceVertexCounts.size(); ++i) {
            mesh.faces.emplace_back(indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]);
          }
        } else {
          // Handle mixed polygons (triangles and quads)
          size_t indexOffset = 0;
          for (int faceVertexCount : faceVertexCounts) {
            if (faceVertexCount == 3) {
              // Triangle
              mesh.faces.emplace_back(
                  faceVertexIndices[indexOffset],
                  faceVertexIndices[indexOffset + 1],
                  faceVertexIndices[indexOffset + 2]);
            } else if (faceVertexCount == 4) {
              // Quad - split into two triangles
              mesh.faces.emplace_back(
                  faceVertexIndices[indexOffset],
                  faceVertexIndices[indexOffset + 1],
                  faceVertexIndices[indexOffset + 2]);
              mesh.faces.emplace_back(
                  faceVertexIndices[indexOffset],
                  faceVertexIndices[indexOffset + 2],
                  faceVertexIndices[indexOffset + 3]);
            }
            // Skip faces with more than 4 vertices for now
            indexOffset += faceVertexCount;
          }
        }
      }

      break; // Use first mesh found
    }
  }

  return mesh;
}

/// Load skin weights from USD stage
SkinWeights loadSkinWeightsFromUsd(const UsdStageRefPtr& stage, size_t numVertices) {
  SkinWeights skinWeights;

  // Initialize matrices with proper dimensions
  skinWeights.index.resize(numVertices, kMaxSkinJoints);
  skinWeights.weight.resize(numVertices, kMaxSkinJoints);

  // Initialize with zeros
  skinWeights.index.setZero();
  skinWeights.weight.setZero();

  // Find skinned mesh prims
  for (const auto& prim : stage->Traverse()) {
    if (prim.IsA<UsdGeomMesh>()) {
      UsdGeomMesh meshPrim(prim);

      // Check if mesh has skinning
      UsdSkelBindingAPI bindingAPI(meshPrim.GetPrim());
      if (bindingAPI) {
        // Get joint indices and weights
        VtArray<int> jointIndices;
        VtArray<float> jointWeights;

        if (bindingAPI.GetJointIndicesAttr().Get(&jointIndices) &&
            bindingAPI.GetJointWeightsAttr().Get(&jointWeights)) {
          // Determine influences per vertex from data size
          if (!jointIndices.empty() && !jointWeights.empty() &&
              jointIndices.size() == jointWeights.size()) {
            const int influencesPerVertex = jointIndices.size() / numVertices;

            if (influencesPerVertex > 0 &&
                jointIndices.size() == numVertices * influencesPerVertex) {
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
      }

      break; // Use first skinned mesh found
    }
  }

  return skinWeights;
}

Character loadUsdCharacterFromStage(const UsdStageRefPtr& stage) {
  Character character;

  character.skeleton = loadSkeletonFromUsd(stage);

  auto mesh = loadMeshFromUsd(stage);
  character.mesh = std::make_unique<Mesh>(std::move(mesh));

  if (!character.mesh->vertices.empty()) {
    auto skinWeights = loadSkinWeightsFromUsd(stage, character.mesh->vertices.size());
    character.skinWeights = std::make_unique<SkinWeights>(std::move(skinWeights));
  }

  if (!character.skeleton.joints.empty()) {
    const size_t numJoints = character.skeleton.joints.size();
    const size_t numJointParams = numJoints * kParametersPerJoint;

    character.parameterTransform = ParameterTransform::empty(numJointParams);

    character.parameterTransform.name.reserve(numJointParams);
    for (const auto& joint : character.skeleton.joints) {
      for (const auto& paramName : kJointParameterNames) {
        character.parameterTransform.name.push_back(joint.name + "_" + paramName);
      }
    }

    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(numJointParams);
    for (size_t i = 0; i < numJointParams; ++i) {
      triplets.emplace_back(i, i, 1.0f);
    }

    character.parameterTransform.transform.resize(numJointParams, numJointParams);
    character.parameterTransform.transform.setFromTriplets(triplets.begin(), triplets.end());

    character.parameterTransform.activeJointParams = VectorX<bool>::Constant(numJointParams, true);
  }

  return character;
}

} // namespace

Character loadUsdCharacter(const filesystem::path& inputPath) {
  initializeUsdWithSuppressedWarnings();
  std::lock_guard<std::mutex> lock(g_usdOperationMutex);

  auto stage = UsdStage::Open(inputPath.string());
  MT_THROW_IF(!stage, "Failed to open USD stage: {}", inputPath.string());

  return loadUsdCharacterFromStage(stage);
}

Character loadUsdCharacter(std::span<const std::byte> inputSpan) {
  initializeUsdWithSuppressedWarnings();
  std::lock_guard<std::mutex> lock(g_usdOperationMutex);

  // Create a unique temporary file name using thread ID and timestamp
  auto tempDir = filesystem::temp_directory_path();
  auto tempPath = tempDir /
      ("momentum_usd_" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) +
       "_" + std::to_string(std::time(nullptr)) + ".usd");

  // Use RAII for automatic cleanup
  struct TempFileGuard {
    filesystem::path path;
    ~TempFileGuard() {
      std::error_code ec;
      filesystem::remove(path, ec); // Don't throw on cleanup
    }
  } tempGuard{tempPath};

  {
    std::ofstream tempFile(tempPath, std::ios::binary);
    MT_THROW_IF(!tempFile.is_open(), "Failed to create temporary file: {}", tempPath.string());
    tempFile.write(reinterpret_cast<const char*>(inputSpan.data()), inputSpan.size());
  } // File closed automatically

  auto stage = UsdStage::Open(tempPath.string());
  MT_THROW_IF(!stage, "Failed to open USD stage from buffer");

  return loadUsdCharacterFromStage(stage); // tempGuard destructor will clean up the file
}

void saveUsd(const filesystem::path& filename, const Character& character) {
  initializeUsdWithSuppressedWarnings();
  std::lock_guard<std::mutex> lock(g_usdOperationMutex);

  auto stage = UsdStage::CreateNew(filename.string());
  MT_THROW_IF(!stage, "Failed to create USD stage: {}", filename.string());

  auto skelRoot = UsdSkelRoot::Define(stage, SdfPath("/SkelRoot"));
  auto skeleton = UsdSkelSkeleton::Define(stage, SdfPath("/SkelRoot/Skeleton"));

  VtArray<TfToken> jointNames;
  jointNames.reserve(character.skeleton.joints.size());
  for (const auto& joint : character.skeleton.joints) {
    jointNames.push_back(TfToken(joint.name));
  }
  skeleton.GetJointsAttr().Set(jointNames);

  VtArray<GfMatrix4d> bindTransforms;
  bindTransforms.reserve(character.skeleton.joints.size());
  for (const auto& joint : character.skeleton.joints) {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 1>(0, 3) = joint.translationOffset;
    bindTransforms.push_back(toGfMatrix4d(transform));
  }
  skeleton.GetBindTransformsAttr().Set(bindTransforms);

  auto mesh = UsdGeomMesh::Define(stage, SdfPath("/SkelRoot/Mesh"));

  VtArray<GfVec3f> points;
  if (character.mesh) {
    points.reserve(character.mesh->vertices.size());
    for (const auto& vertex : character.mesh->vertices) {
      points.push_back(GfVec3f(vertex.x(), vertex.y(), vertex.z()));
    }
  }
  mesh.GetPointsAttr().Set(points);

  VtArray<int> faceVertexCounts;
  VtArray<int> faceVertexIndices;

  if (character.mesh) {
    faceVertexCounts.reserve(character.mesh->faces.size());
    faceVertexIndices.reserve(character.mesh->faces.size() * 3);

    for (const auto& face : character.mesh->faces) {
      faceVertexCounts.push_back(3);
      faceVertexIndices.push_back(face.x());
      faceVertexIndices.push_back(face.y());
      faceVertexIndices.push_back(face.z());
    }
  }

  mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts);
  mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices);

  if (character.skinWeights && character.skinWeights->index.rows() > 0) {
    UsdSkelBindingAPI bindingAPI = UsdSkelBindingAPI::Apply(mesh.GetPrim());

    bindingAPI.GetSkeletonRel().SetTargets({skeleton.GetPath()});

    VtArray<int> jointIndices;
    VtArray<float> jointWeights;

    const int maxInfluences = 4;
    const int numVertices = character.skinWeights->index.rows();
    const int numJointsPerVertex =
        std::min(maxInfluences, static_cast<int>(character.skinWeights->index.cols()));

    jointIndices.reserve(numVertices * maxInfluences);
    jointWeights.reserve(numVertices * maxInfluences);

    for (int v = 0; v < numVertices; ++v) {
      for (int i = 0; i < numJointsPerVertex; ++i) {
        jointIndices.push_back(character.skinWeights->index(v, i));
        jointWeights.push_back(character.skinWeights->weight(v, i));
      }

      for (int i = numJointsPerVertex; i < maxInfluences; ++i) {
        jointIndices.push_back(0);
        jointWeights.push_back(0.0f);
      }
    }

    bindingAPI.GetJointIndicesAttr().Set(jointIndices);
    bindingAPI.GetJointWeightsAttr().Set(jointWeights);
  }

  stage->GetRootLayer()->Save();
}

} // namespace momentum
