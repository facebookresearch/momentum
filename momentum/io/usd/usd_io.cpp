/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/usd/usd_io.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character/types.h"
#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/io/usd/usd_mesh_io.h"
#include "momentum/io/usd/usd_skeleton_io.h"
#include "momentum/math/mesh.h"

#include <pxr/base/tf/diagnosticMgr.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdSkel/root.h>

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

Character loadUsdCharacterFromStage(const UsdStageRefPtr& stage) {
  Character character;

  character.skeleton = loadSkeletonFromUsd(stage);

  auto mesh = loadMeshFromUsd(stage);
  character.mesh = std::make_unique<Mesh>(std::move(mesh));

  if (!character.mesh->vertices.empty()) {
    auto skinWeights = loadSkinWeightsFromUsd(stage, character.mesh->vertices.size());
    character.skinWeights = std::make_unique<SkinWeights>(std::move(skinWeights));

    auto blendShapes = loadBlendShapesFromUsd(stage, character.mesh->vertices.size());
    if (blendShapes) {
      character.blendShape = std::move(blendShapes);
    }
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

  saveSkeletonToUsd(character.skeleton, skeleton);

  auto mesh = UsdGeomMesh::Define(stage, SdfPath("/SkelRoot/Mesh"));

  if (character.mesh) {
    saveMeshToUsd(*character.mesh, mesh);
  }

  if (character.skinWeights) {
    saveSkinWeightsToUsd(*character.skinWeights, mesh, skeleton);
  }

  if (character.blendShape) {
    saveBlendShapesToUsd(*character.blendShape, mesh);
  }

  stage->GetRootLayer()->Save();
}

} // namespace momentum
