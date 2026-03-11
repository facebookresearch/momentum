/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/common/filesystem.h>
#include <momentum/test/io/io_helpers.h>

#include <gtest/gtest.h>

#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define WEXITSTATUS(x) (x)
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

/// Integration test fixture that invokes the convert_model CLI binary
/// with various argument combinations to verify the refactoring
/// preserves all behavior.
class ConvertModelIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* binaryPath = getenv("CONVERT_MODEL_BINARY");
    ASSERT_NE(binaryPath, nullptr) << "CONVERT_MODEL_BINARY environment variable not set";
    // Make binary path absolute so it works from any CWD
    binaryPath_ = filesystem::absolute(binaryPath).string();

    const char* testDataPath = getenv("TEST_MOMENTUM_MODELS_PATH");
    ASSERT_NE(testDataPath, nullptr) << "TEST_MOMENTUM_MODELS_PATH environment variable not set";
    testDataPath_ = filesystem::absolute(testDataPath).string();

    // Use Momentum's temporaryDirectory() which handles RE sandbox env vars
    tempDir_ = momentum::temporaryDirectory("convert_model_cli");
  }

  /// Run the convert_model binary with the given arguments.
  /// Returns the process exit code (0 = success).
  int runConvertModel(const std::vector<std::string>& args) {
#ifdef _WIN32
    // On Windows, use _spawnv which doesn't spawn a shell
    std::vector<const char*> argv;
    argv.push_back(binaryPath_.c_str());
    for (const auto& arg : args) {
      argv.push_back(arg.c_str());
    }
    argv.push_back(nullptr);
    const int status = _spawnv(_P_WAIT, binaryPath_.c_str(), argv.data());
    return status;
#else
    // Build argv for execvp (no shell spawned)
    std::vector<const char*> argv;
    argv.push_back(binaryPath_.c_str());
    for (const auto& arg : args) {
      argv.push_back(arg.c_str());
    }
    argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid == -1) {
      return -1; // fork failed
    }
    if (pid == 0) {
      // Child process
      execvp(binaryPath_.c_str(), const_cast<char* const*>(argv.data()));
      _exit(127); // execvp failed
    }
    // Parent process
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
      return WEXITSTATUS(status);
    }
    return -1; // abnormal termination
#endif
  }

  /// Helper to construct a path to a test data file.
  [[nodiscard]] std::string testFile(const std::string& filename) const {
    return (std::filesystem::path(testDataPath_) / filename).string();
  }

  /// Helper to construct a path to a temporary output file.
  [[nodiscard]] std::string outputFile(const std::string& filename) const {
    return (tempDir_.path() / filename).string();
  }

  std::string binaryPath_;
  std::string testDataPath_;
  momentum::TemporaryDirectory tempDir_;
};

// ============================================================================
// PAIRWISE INTEGRATION TESTS
// Test all combinations of input parameters through the actual CLI
// ============================================================================

// Test 1: GLB model + GLB motion -> FBX output (with params, locators, mesh, namespace)
TEST_F(ConvertModelIntegrationTest, Pairwise01_Glb_Glb_Fbx_Full) {
  const auto out = outputFile("output_01.fbx");
  const auto outLocLocal = outputFile("output_01_local.locators");
  const auto outLocGlobal = outputFile("output_01_global.locators");

  int rc = runConvertModel(
      {"--model",
       testFile("character_with_motion.glb"),
       "--motion",
       testFile("character_with_motion.glb"),
       "--parameters",
       testFile("character.model"),
       "--locator",
       testFile("character.locators"),
       "--out",
       out,
       "--out-locator-local",
       outLocLocal,
       "--out-locator-global",
       outLocGlobal,
       "--save-markers",
       "--character-mesh",
       "--fbx-namespace",
       "test_ns"});

  EXPECT_EQ(rc, 0) << "convert_model should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test 2: FBX model + FBX motion -> GLB output (with params for correct parameter transform)
TEST_F(ConvertModelIntegrationTest, Pairwise02_Fbx_Fbx_Glb_Minimal) {
  const auto out = outputFile("output_02.glb");

  int rc = runConvertModel(
      {"--model",
       testFile("character.fbx"),
       "--motion",
       testFile("motion.fbx"),
       "--parameters",
       testFile("character.model"),
       "--out",
       out});

  EXPECT_EQ(rc, 0) << "convert_model should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test 3: GLB model + no motion -> GLTF output (with params)
TEST_F(ConvertModelIntegrationTest, Pairwise03_Glb_NoMotion_Gltf) {
  const auto out = outputFile("output_03.gltf");

  int rc = runConvertModel(
      {"--model",
       testFile("character_with_motion.glb"),
       "--parameters",
       testFile("character.model"),
       "--out",
       out});

  EXPECT_EQ(rc, 0) << "convert_model should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test 4: No model + GLB motion (embedded character) -> FBX output
TEST_F(ConvertModelIntegrationTest, Pairwise04_NoModel_GlbMotion_Fbx) {
  const auto out = outputFile("output_04.fbx");

  int rc = runConvertModel({"--motion", testFile("character_with_motion.glb"), "--out", out});

  EXPECT_EQ(rc, 0) << "convert_model should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test 5: No model + GLB motion (embedded character) -> GLB output (with markers)
TEST_F(ConvertModelIntegrationTest, Pairwise05_NoModel_GlbMotion_Glb_Markers) {
  const auto out = outputFile("output_05.glb");

  int rc = runConvertModel(
      {"--motion", testFile("character_with_motion.glb"), "--out", out, "--save-markers"});

  EXPECT_EQ(rc, 0) << "convert_model should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test 6: No model + GLB motion (embedded character) -> GLTF output (no motion used)
TEST_F(ConvertModelIntegrationTest, Pairwise06_NoModel_GlbMotion_Gltf) {
  const auto out = outputFile("output_06.gltf");
  const auto outLocLocal = outputFile("output_06_local.locators");
  const auto outLocGlobal = outputFile("output_06_global.locators");

  int rc = runConvertModel(
      {"--motion",
       testFile("character_with_motion.glb"),
       "--out",
       out,
       "--out-locator-local",
       outLocLocal,
       "--out-locator-global",
       outLocGlobal});

  EXPECT_EQ(rc, 0) << "convert_model should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test 7: FBX motion + params + locators -> FBX output (with mesh, namespace)
TEST_F(ConvertModelIntegrationTest, Pairwise07_Fbx_Fbx_Fbx_ParamsLocators) {
  const auto out = outputFile("output_07.fbx");

  int rc = runConvertModel(
      {"--model",
       testFile("motion.fbx"),
       "--motion",
       testFile("motion.fbx"),
       "--parameters",
       testFile("character.model"),
       "--locator",
       testFile("character.locators"),
       "--out",
       out,
       "--character-mesh",
       "--fbx-namespace",
       "test_ns"});

  EXPECT_EQ(rc, 0) << "convert_model should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test 8: FBX motion + params -> GLB output
TEST_F(ConvertModelIntegrationTest, Pairwise08_Fbx_Fbx_Glb_Params) {
  const auto out = outputFile("output_08.glb");

  int rc = runConvertModel(
      {"--model",
       testFile("motion.fbx"),
       "--motion",
       testFile("motion.fbx"),
       "--parameters",
       testFile("character.model"),
       "--out",
       out});

  EXPECT_EQ(rc, 0) << "convert_model should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test 9: FBX model + no motion -> GLTF output
TEST_F(ConvertModelIntegrationTest, Pairwise09_Fbx_NoMotion_Gltf) {
  const auto out = outputFile("output_09.gltf");

  int rc = runConvertModel({"--model", testFile("character.fbx"), "--out", out});

  EXPECT_EQ(rc, 0) << "convert_model should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test 10: GLB model + GLB motion -> GLB output (with locators)
TEST_F(ConvertModelIntegrationTest, Pairwise10_Glb_Glb_Glb_Locators) {
  const auto out = outputFile("output_10.glb");
  const auto outLocLocal = outputFile("output_10_local.locators");
  const auto outLocGlobal = outputFile("output_10_global.locators");

  int rc = runConvertModel(
      {"--model",
       testFile("character_with_motion.glb"),
       "--motion",
       testFile("character_with_motion.glb"),
       "--locator",
       testFile("character.locators"),
       "--out",
       out,
       "--out-locator-local",
       outLocLocal,
       "--out-locator-global",
       outLocGlobal});

  EXPECT_EQ(rc, 0) << "convert_model should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test 11: GLB model + FBX motion -> GLTF output (with params)
TEST_F(ConvertModelIntegrationTest, Pairwise11_Glb_Fbx_Gltf_Params) {
  const auto out = outputFile("output_11.gltf");

  int rc = runConvertModel(
      {"--model",
       testFile("character_with_motion.glb"),
       "--motion",
       testFile("motion.fbx"),
       "--parameters",
       testFile("character.model"),
       "--out",
       out});

  EXPECT_EQ(rc, 0) << "convert_model should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test 12: No model + GLB motion -> GLTF output
TEST_F(ConvertModelIntegrationTest, Pairwise12_NoModel_GlbMotion_Gltf) {
  const auto out = outputFile("output_12.gltf");

  int rc = runConvertModel({"--motion", testFile("character_with_motion.glb"), "--out", out});

  EXPECT_EQ(rc, 0) << "convert_model should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// ============================================================================
// CRITICAL WORKFLOW TESTS
// ============================================================================

// Test: FBX to GLB conversion
TEST_F(ConvertModelIntegrationTest, Workflow_FbxToGlb) {
  const auto out = outputFile("fbx_to_glb.glb");

  int rc = runConvertModel(
      {"--model",
       testFile("motion.fbx"),
       "--motion",
       testFile("motion.fbx"),
       "--parameters",
       testFile("character.model"),
       "--out",
       out});

  EXPECT_EQ(rc, 0) << "FBX to GLB conversion should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test: GLB to FBX conversion
TEST_F(ConvertModelIntegrationTest, Workflow_GlbToFbx) {
  const auto out = outputFile("glb_to_fbx.fbx");

  int rc = runConvertModel(
      {"--model",
       testFile("character_with_motion.glb"),
       "--motion",
       testFile("character_with_motion.glb"),
       "--out",
       out,
       "--character-mesh"});

  EXPECT_EQ(rc, 0) << "GLB to FBX conversion should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test: Model-only export (no motion) to FBX
TEST_F(ConvertModelIntegrationTest, Workflow_ModelOnly_Fbx) {
  const auto out = outputFile("model_only.fbx");

  int rc = runConvertModel({"--model", testFile("character_with_motion.glb"), "--out", out});

  EXPECT_EQ(rc, 0) << "Model-only FBX export should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test: Model-only export (no motion) to GLB
TEST_F(ConvertModelIntegrationTest, Workflow_ModelOnly_Glb) {
  const auto out = outputFile("model_only.glb");

  int rc = runConvertModel({"--model", testFile("character_with_motion.glb"), "--out", out});

  EXPECT_EQ(rc, 0) << "Model-only GLB export should succeed";
  EXPECT_TRUE(std::filesystem::exists(out));
}

// Test: Round-trip FBX -> GLB -> FBX
TEST_F(ConvertModelIntegrationTest, Workflow_RoundTrip_FbxGlbFbx) {
  const auto glbOut = outputFile("roundtrip.glb");
  const auto fbxOut = outputFile("roundtrip.fbx");

  // Step 1: FBX -> GLB
  int rc1 = runConvertModel(
      {"--model",
       testFile("motion.fbx"),
       "--motion",
       testFile("motion.fbx"),
       "--parameters",
       testFile("character.model"),
       "--out",
       glbOut});
  EXPECT_EQ(rc1, 0) << "FBX to GLB should succeed";
  EXPECT_TRUE(std::filesystem::exists(glbOut));

  // Step 2: GLB -> FBX
  int rc2 =
      runConvertModel({"--model", glbOut, "--motion", glbOut, "--out", fbxOut, "--character-mesh"});
  EXPECT_EQ(rc2, 0) << "GLB to FBX should succeed";
  EXPECT_TRUE(std::filesystem::exists(fbxOut));
}

// ============================================================================
// ERROR PATH TESTS
// ============================================================================

// Test: Invalid output format
TEST_F(ConvertModelIntegrationTest, Error_InvalidOutputFormat) {
  int rc = runConvertModel(
      {"--model", testFile("character_with_motion.glb"), "--out", outputFile("output.obj")});

  EXPECT_NE(rc, 0) << "Invalid output format should fail";
}

// Test: Unknown motion format
TEST_F(ConvertModelIntegrationTest, Error_UnknownMotionFormat) {
  int rc = runConvertModel(
      {"--model",
       testFile("character_with_motion.glb"),
       "--motion",
       testFile("character.locators"),
       "--out",
       outputFile("output.fbx")});

  // Should succeed (unknown motion format is warned but not fatal)
  // The app logs a warning and exports without motion
  EXPECT_EQ(rc, 0);
}

// Test: Locator output (local and global)
TEST_F(ConvertModelIntegrationTest, LocatorOutput_LocalAndGlobal) {
  const auto out = outputFile("locator_test.fbx");
  const auto outLocLocal = outputFile("locators_local.locators");
  const auto outLocGlobal = outputFile("locators_global.locators");

  int rc = runConvertModel(
      {"--model",
       testFile("character_with_motion.glb"),
       "--locator",
       testFile("character.locators"),
       "--out",
       out,
       "--out-locator-local",
       outLocLocal,
       "--out-locator-global",
       outLocGlobal});

  EXPECT_EQ(rc, 0);
  EXPECT_TRUE(std::filesystem::exists(out));
  EXPECT_TRUE(std::filesystem::exists(outLocLocal));
  EXPECT_TRUE(std::filesystem::exists(outLocGlobal));
}

} // namespace
