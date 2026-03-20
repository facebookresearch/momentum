/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/usd/usd_io.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/character.h"
#include "momentum/character/collision_geometry.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character/types.h"
#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/io/common/json_utils.h"
#include "momentum/io/usd/usd_animation_io.h"
#include "momentum/io/usd/usd_mesh_io.h"
#include "momentum/io/usd/usd_skeleton_io.h"
#include "momentum/math/mesh.h"

#include <nlohmann/json.hpp>

#include <pxr/base/tf/diagnosticMgr.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/valueTypeName.h>
#include <pxr/usd/usd/primRange.h>
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

void saveMomentumMetadata(const Character& character, const UsdPrim& skelRootPrim) {
  // Save character name
  if (!character.name.empty()) {
    auto nameAttr =
        skelRootPrim.CreateAttribute(TfToken("momentum:characterName"), SdfValueTypeNames->String);
    nameAttr.Set(std::string(character.name));
  }

  // Save parameter transform as JSON
  if (!character.parameterTransform.name.empty()) {
    nlohmann::json ptJson;
    parameterTransformToJson(character, ptJson);
    auto ptAttr = skelRootPrim.CreateAttribute(
        TfToken("momentum:parameterTransform"), SdfValueTypeNames->String);
    ptAttr.Set(ptJson.dump());
  }

  // Save parameter limits as JSON
  if (!character.parameterLimits.empty()) {
    nlohmann::json limJson;
    parameterLimitsToJson(character, limJson);
    auto limAttr = skelRootPrim.CreateAttribute(
        TfToken("momentum:parameterLimits"), SdfValueTypeNames->String);
    limAttr.Set(limJson.dump());
  }

  // Save parameter sets as JSON
  if (!character.parameterTransform.parameterSets.empty()) {
    nlohmann::json setsJson;
    parameterSetsToJson(character, setsJson);
    auto setsAttr =
        skelRootPrim.CreateAttribute(TfToken("momentum:parameterSets"), SdfValueTypeNames->String);
    setsAttr.Set(setsJson.dump());
  }

  // Save pose constraints as JSON
  if (!character.parameterTransform.poseConstraints.empty()) {
    nlohmann::json pcJson;
    poseConstraintsToJson(character, pcJson);
    auto pcAttr = skelRootPrim.CreateAttribute(
        TfToken("momentum:poseConstraints"), SdfValueTypeNames->String);
    pcAttr.Set(pcJson.dump());
  }
}

void loadMomentumMetadata(Character& character, const UsdPrim& skelRootPrim) {
  // Load character name
  auto nameAttr = skelRootPrim.GetAttribute(TfToken("momentum:characterName"));
  if (nameAttr) {
    std::string name;
    if (nameAttr.Get(&name)) {
      character.name = name;
    }
  }

  // Load parameter transform from JSON
  auto ptAttr = skelRootPrim.GetAttribute(TfToken("momentum:parameterTransform"));
  if (ptAttr) {
    std::string ptStr;
    if (ptAttr.Get(&ptStr)) {
      try {
        auto ptJson = nlohmann::json::parse(ptStr);
        character.parameterTransform = parameterTransformFromJson(character, ptJson);
      } catch (const nlohmann::json::parse_error& e) {
        MT_LOGW("Failed to parse momentum:parameterTransform: {}", e.what());
      }
    }
  }

  // Load parameter limits from JSON
  auto limAttr = skelRootPrim.GetAttribute(TfToken("momentum:parameterLimits"));
  if (limAttr) {
    std::string limStr;
    if (limAttr.Get(&limStr)) {
      try {
        auto limJson = nlohmann::json::parse(limStr);
        character.parameterLimits = parameterLimitsFromJson(character, limJson);
      } catch (const nlohmann::json::parse_error& e) {
        MT_LOGW("Failed to parse momentum:parameterLimits: {}", e.what());
      }
    }
  }

  // Load parameter sets from JSON
  auto setsAttr = skelRootPrim.GetAttribute(TfToken("momentum:parameterSets"));
  if (setsAttr) {
    std::string setsStr;
    if (setsAttr.Get(&setsStr)) {
      try {
        auto setsJson = nlohmann::json::parse(setsStr);
        character.parameterTransform.parameterSets = parameterSetsFromJson(character, setsJson);
      } catch (const nlohmann::json::parse_error& e) {
        MT_LOGW("Failed to parse momentum:parameterSets: {}", e.what());
      }
    }
  }

  // Load pose constraints from JSON
  auto pcAttr = skelRootPrim.GetAttribute(TfToken("momentum:poseConstraints"));
  if (pcAttr) {
    std::string pcStr;
    if (pcAttr.Get(&pcStr)) {
      try {
        auto pcJson = nlohmann::json::parse(pcStr);
        character.parameterTransform.poseConstraints = poseConstraintsFromJson(character, pcJson);
      } catch (const nlohmann::json::parse_error& e) {
        MT_LOGW("Failed to parse momentum:poseConstraints: {}", e.what());
      }
    }
  }
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

  // Load collision geometry and locators
  auto collision = loadCollisionGeometryFromUsd(stage, character.skeleton);
  if (!collision.empty()) {
    character.collision = std::make_unique<CollisionGeometry>(std::move(collision));
  }

  character.locators = loadLocatorsFromUsd(stage, character.skeleton);

  // Load momentum-specific metadata from SkelRoot prim
  for (const auto& prim : stage->Traverse()) {
    if (prim.IsA<UsdSkelRoot>()) {
      loadMomentumMetadata(character, prim);
      break;
    }
  }

  return character;
}

/// RAII guard that removes a temporary file on destruction.
struct TempFileGuard {
  filesystem::path path;
  explicit TempFileGuard(filesystem::path p) : path(std::move(p)) {}
  ~TempFileGuard() {
    std::error_code ec;
    filesystem::remove(path, ec);
  }
  TempFileGuard(const TempFileGuard&) = delete;
  TempFileGuard& operator=(const TempFileGuard&) = delete;
  TempFileGuard(TempFileGuard&&) = delete;
  TempFileGuard& operator=(TempFileGuard&&) = delete;
};

std::tuple<Character, MotionParameters, IdentityParameters, float>
loadUsdCharacterWithMotionFromStage(const UsdStageRefPtr& stage) {
  auto character = loadUsdCharacterFromStage(stage);

  MotionParameters motion;
  IdentityParameters identity;
  float fps = 120.0f;

  for (const auto& prim : stage->Traverse()) {
    if (prim.IsA<UsdSkelRoot>()) {
      std::tie(motion, identity, fps) = loadMotionFromUsd(prim);
      break;
    }
  }

  return {std::move(character), std::move(motion), std::move(identity), fps};
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

  TempFileGuard tempGuard(tempPath);

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

  if (character.collision) {
    saveCollisionGeometryToUsd(
        *character.collision, character.skeleton, stage, SdfPath("/SkelRoot"));
  }

  if (!character.locators.empty()) {
    saveLocatorsToUsd(character.locators, character.skeleton, stage, SdfPath("/SkelRoot"));
  }

  saveMomentumMetadata(character, skelRoot.GetPrim());

  stage->GetRootLayer()->Save();
}

std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadUsdCharacterWithSkeletonStates(const filesystem::path& inputPath) {
  initializeUsdWithSuppressedWarnings();
  std::lock_guard<std::mutex> lock(g_usdOperationMutex);

  auto stage = UsdStage::Open(inputPath.string());
  MT_THROW_IF(!stage, "Failed to open USD stage: {}", inputPath.string());

  auto character = loadUsdCharacterFromStage(stage);
  auto [states, frameTimes] = loadSkeletonStatesFromUsd(stage, character.skeleton);

  return {std::move(character), std::move(states), std::move(frameTimes)};
}

std::tuple<Character, std::vector<SkeletonState>, std::vector<float>>
loadUsdCharacterWithSkeletonStates(std::span<const std::byte> inputSpan) {
  initializeUsdWithSuppressedWarnings();
  std::lock_guard<std::mutex> lock(g_usdOperationMutex);

  auto tempDir = filesystem::temp_directory_path();
  auto tempPath = tempDir /
      ("momentum_usd_" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) +
       "_" + std::to_string(std::time(nullptr)) + ".usd");

  TempFileGuard tempGuard(tempPath);

  {
    std::ofstream tempFile(tempPath, std::ios::binary);
    MT_THROW_IF(!tempFile.is_open(), "Failed to create temporary file: {}", tempPath.string());
    tempFile.write(reinterpret_cast<const char*>(inputSpan.data()), inputSpan.size());
  }

  auto stage = UsdStage::Open(tempPath.string());
  MT_THROW_IF(!stage, "Failed to open USD stage from buffer");

  auto character = loadUsdCharacterFromStage(stage);
  auto [states, frameTimes] = loadSkeletonStatesFromUsd(stage, character.skeleton);

  return {std::move(character), std::move(states), std::move(frameTimes)};
}

void saveUsdCharacter(
    const filesystem::path& filename,
    const Character& character,
    float fps,
    std::span<const SkeletonState> skeletonStates) {
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

  if (!skeletonStates.empty()) {
    saveSkeletonStatesToUsd(stage, skeleton, character.skeleton, skeletonStates, fps);
  }

  if (character.collision) {
    saveCollisionGeometryToUsd(
        *character.collision, character.skeleton, stage, SdfPath("/SkelRoot"));
  }

  if (!character.locators.empty()) {
    saveLocatorsToUsd(character.locators, character.skeleton, stage, SdfPath("/SkelRoot"));
  }

  saveMomentumMetadata(character, skelRoot.GetPrim());

  stage->GetRootLayer()->Save();
}

void saveUsdCharacterWithMotion(
    const filesystem::path& filename,
    const Character& character,
    float fps,
    const MotionParameters& motion,
    const IdentityParameters& offsets) {
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

  if (character.collision) {
    saveCollisionGeometryToUsd(
        *character.collision, character.skeleton, stage, SdfPath("/SkelRoot"));
  }

  if (!character.locators.empty()) {
    saveLocatorsToUsd(character.locators, character.skeleton, stage, SdfPath("/SkelRoot"));
  }

  // Save motion data as custom attributes
  const auto& [paramNames, poses] = motion;
  if (!paramNames.empty() && poses.cols() > 0) {
    saveMotionToUsd(skelRoot.GetPrim(), fps, motion, offsets);

    // Also bake as skeleton states for interop with standard USD viewers
    const auto numFrames = poses.cols();
    std::vector<SkeletonState> skeletonStates;
    skeletonStates.reserve(numFrames);

    for (Eigen::Index f = 0; f < numFrames; ++f) {
      // Apply the model parameters through the parameter transform
      JointParameters jointParams =
          character.parameterTransform.apply(ModelParameters(poses.col(f)));
      skeletonStates.emplace_back(jointParams, character.skeleton, false);
    }

    saveSkeletonStatesToUsd(stage, skeleton, character.skeleton, skeletonStates, fps);
  }

  saveMomentumMetadata(character, skelRoot.GetPrim());

  stage->GetRootLayer()->Save();
}

std::tuple<Character, MotionParameters, IdentityParameters, float> loadUsdCharacterWithMotion(
    const filesystem::path& inputPath) {
  initializeUsdWithSuppressedWarnings();
  std::lock_guard<std::mutex> lock(g_usdOperationMutex);

  auto stage = UsdStage::Open(inputPath.string());
  MT_THROW_IF(!stage, "Failed to open USD stage: {}", inputPath.string());

  return loadUsdCharacterWithMotionFromStage(stage);
}

std::tuple<Character, MotionParameters, IdentityParameters, float> loadUsdCharacterWithMotion(
    std::span<const std::byte> inputSpan) {
  initializeUsdWithSuppressedWarnings();
  std::lock_guard<std::mutex> lock(g_usdOperationMutex);

  auto tempDir = filesystem::temp_directory_path();
  auto tempPath = tempDir /
      ("momentum_usd_" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) +
       "_" + std::to_string(std::time(nullptr)) + ".usd");

  TempFileGuard tempGuard(tempPath);

  {
    std::ofstream tempFile(tempPath, std::ios::binary);
    MT_THROW_IF(!tempFile.is_open(), "Failed to create temporary file: {}", tempPath.string());
    tempFile.write(reinterpret_cast<const char*>(inputSpan.data()), inputSpan.size());
  }

  auto stage = UsdStage::Open(tempPath.string());
  MT_THROW_IF(!stage, "Failed to open USD stage from buffer");

  return loadUsdCharacterWithMotionFromStage(stage);
}

} // namespace momentum
