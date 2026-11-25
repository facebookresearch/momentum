/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "convert_model_test_helpers.h"

#include <momentum/common/filesystem.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/test/io/io_helpers.h>

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
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
       "--out-locators-local",
       outLocLocal,
       "--out-locators-global",
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
       "--out-locators-local",
       outLocLocal,
       "--out-locators-global",
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
       "--out-locators-local",
       outLocLocal,
       "--out-locators-global",
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

// Test: Parameter output writes a valid .model file
TEST_F(ConvertModelIntegrationTest, ParameterOutput_WritesModelFile) {
  const auto out = outputFile("params_test.glb");
  const auto outParams = outputFile("exported.model");

  int rc = runConvertModel(
      {"--model",
       testFile("character_with_motion.glb"),
       "--parameters",
       testFile("character.model"),
       "--out",
       out,
       "--out-parameters",
       outParams});

  ASSERT_EQ(rc, 0);
  ASSERT_TRUE(std::filesystem::exists(outParams));
  EXPECT_GT(std::filesystem::file_size(outParams), 0u);

  std::ifstream paramsFile(outParams);
  ASSERT_TRUE(paramsFile.is_open());
  const std::string paramsData{
      std::istreambuf_iterator<char>{paramsFile}, std::istreambuf_iterator<char>{}};
  EXPECT_NE(paramsData.find("[ParameterTransform]"), std::string::npos);
  EXPECT_NE(paramsData.find("root.tx = 1*root_tx"), std::string::npos);
  EXPECT_NE(paramsData.find("joint2.rz = 0.5*shared_rz"), std::string::npos);
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
       "--out-locators-local",
       outLocLocal,
       "--out-locators-global",
       outLocGlobal});

  EXPECT_EQ(rc, 0);
  EXPECT_TRUE(std::filesystem::exists(out));
  EXPECT_TRUE(std::filesystem::exists(outLocLocal));
  EXPECT_TRUE(std::filesystem::exists(outLocGlobal));
}

// ============================================================================
// CONVERSION RESULT TESTS
// Load the output back and compare it with the input
// ============================================================================

// Test: a single-character GLB motion comes back from the FBX output
TEST_F(ConvertModelIntegrationTest, Result_SingleCharacterGlbToFbx_KeepsTheMotion) {
  const auto input = testFile("character_with_motion.glb");
  const auto out = outputFile("single.fbx");

  ASSERT_EQ(runConvertModel({"--motion", input, "--out", out}), 0);

  const auto [character, motion, identity, fps] = momentum::loadCharacterWithMotion(input);
  const auto [loaded, takes, loadedFps] = momentum::loadFbxCharacterWithMotion(out);
  ASSERT_EQ(takes.size(), 1);
  const momentum::MatrixXf expected = momentum::toJointParameters(character, motion);
  ASSERT_EQ(takes[0].rows(), expected.rows());
  ASSERT_EQ(takes[0].cols(), expected.cols());
  EXPECT_LT((takes[0] - expected).cwiseAbs().maxCoeff(), 1e-4f);
  EXPECT_FLOAT_EQ(loadedFps, fps);
}

// Test: a motion file is read whatever the letter case of its extension
TEST_F(ConvertModelIntegrationTest, Result_UpperCaseMotionExtension_KeepsTheMotion) {
  const auto input = outputFile("MOTION.GLB");
  std::filesystem::copy_file(testFile("character_with_motion.glb"), input);
  const auto out = outputFile("upper_case.fbx");

  ASSERT_EQ(runConvertModel({"--motion", input, "--out", out}), 0);

  const auto [character, motion, identity, fps] = momentum::loadCharacterWithMotion(input);
  const auto [loaded, takes, loadedFps] = momentum::loadFbxCharacterWithMotion(out);
  ASSERT_EQ(takes.size(), 1);
  EXPECT_EQ(takes[0].cols(), motion.cols());
}

// Test: the output format is read from the output extension in any letter case
TEST_F(ConvertModelIntegrationTest, Result_UpperCaseOutputExtension_KeepsTheMotion) {
  const auto input = testFile("character_with_motion.glb");
  const auto glb = outputFile("upper_case.GLB");
  const auto fbx = outputFile("upper_case.FBX");

  ASSERT_EQ(runConvertModel({"--motion", input, "--out", glb}), 0);
  ASSERT_EQ(runConvertModel({"--motion", input, "--out", fbx}), 0);

  const auto [character, motion, identity, fps] = momentum::loadCharacterWithMotion(input);
  const auto [glbCharacter, glbMotion, glbIdentity, glbFps] =
      momentum::loadCharacterWithMotion(glb);
  EXPECT_EQ(glbMotion.cols(), motion.cols());
  const auto [fbxCharacter, takes, fbxFps] = momentum::loadFbxCharacterWithMotion(fbx);
  ASSERT_EQ(takes.size(), 1);
  EXPECT_EQ(takes[0].cols(), motion.cols());
}

// Test: every character of a GLB file comes back from the FBX output with its motion
TEST_F(ConvertModelIntegrationTest, Result_MultiCharacterGlbToFbx_KeepsEveryCharacterAndMotion) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto input = outputFile("two.glb");
  momentum::saveTestSceneAsGltf(input, saved, 30.0f);
  const auto out = outputFile("two.fbx");

  ASSERT_EQ(runConvertModel({"--motion", input, "--out", out}), 0);

  const auto [loaded, fps] = momentum::loadFbxCharactersWithMotion(out);
  ASSERT_EQ(loaded.size(), saved.size());
  EXPECT_FLOAT_EQ(fps, 30.0f);
  for (size_t i = 0; i < saved.size(); ++i) {
    const auto& [character, takes] = loaded[i];
    EXPECT_EQ(character.name, saved[i].character.name);
    ASSERT_EQ(takes.size(), 1);
    const momentum::MatrixXf expected =
        momentum::toJointParameters(saved[i].character, saved[i].motion);
    ASSERT_EQ(takes[0].rows(), expected.rows()) << i;
    ASSERT_EQ(takes[0].cols(), expected.cols()) << i;
    EXPECT_LT((takes[0] - expected).cwiseAbs().maxCoeff(), 1e-4f) << i;
  }
}

// Test: every character of an FBX file comes back from the GLB output with its motion
TEST_F(ConvertModelIntegrationTest, Result_MultiCharacterFbxToGlb_KeepsEveryCharacterAndMotion) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto input = outputFile("two.fbx");
  momentum::saveTestSceneAsFbx(input, saved, 30.0f);
  const auto out = outputFile("two.glb");

  ASSERT_EQ(runConvertModel({"--motion", input, "--out", out}), 0);

  // The FBX characters have an identity parameter transform, so the motion is joint parameters.
  const auto [loaded, fps] = momentum::loadGltfCharactersWithMotion(out);
  ASSERT_EQ(loaded.size(), saved.size());
  EXPECT_FLOAT_EQ(fps, 30.0f);
  for (size_t i = 0; i < saved.size(); ++i) {
    EXPECT_EQ(loaded[i].character.name, saved[i].character.name);
    const momentum::MatrixXf expected =
        momentum::toJointParameters(saved[i].character, saved[i].motion);
    ASSERT_EQ(loaded[i].motion.rows(), expected.rows()) << i;
    ASSERT_EQ(loaded[i].motion.cols(), expected.cols()) << i;
    EXPECT_LT((loaded[i].motion - expected).cwiseAbs().maxCoeff(), 1e-4f) << i;
  }
}

// Test: a model file with several characters keeps all of them
TEST_F(ConvertModelIntegrationTest, Result_MultiCharacterModel_KeepsEveryCharacter) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto input = outputFile("two_model.glb");
  momentum::saveTestSceneAsGltf(input, saved, 30.0f);
  const auto out = outputFile("two_model.fbx");

  ASSERT_EQ(runConvertModel({"--model", input, "--out", out}), 0);

  const auto loaded = momentum::loadFbxCharacters(out);
  ASSERT_EQ(loaded.size(), saved.size());
  for (size_t i = 0; i < saved.size(); ++i) {
    EXPECT_EQ(loaded[i].name, saved[i].character.name);
    EXPECT_EQ(loaded[i].skeleton.getJointNames(), saved[i].character.skeleton.getJointNames());
  }
}

// Test: a model character that no motion-file character matches gets no motion
TEST_F(ConvertModelIntegrationTest, Result_ModelCharacterWithoutMatchingMotion_HasNoMotion) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto model = outputFile("model_ab.glb");
  momentum::GltfBuilder modelBuilder;
  for (const auto& entry : saved) {
    modelBuilder.addCharacter(entry.character);
  }
  modelBuilder.save(model);
  const auto motion = outputFile("motion_b.glb");
  momentum::saveTestSceneAsGltf(motion, {saved[1]}, 30.0f);
  const auto out = outputFile("model_ab_out.glb");

  ASSERT_EQ(runConvertModel({"--model", model, "--motion", motion, "--out", out}), 0);

  const auto [loaded, fps] = momentum::loadGltfCharactersWithMotion(out);
  ASSERT_EQ(loaded.size(), 2);
  EXPECT_EQ(loaded[0].motion.cols(), 0);
  ASSERT_EQ(loaded[1].motion.cols(), saved[1].motion.cols());
  ASSERT_EQ(loaded[1].motion.rows(), saved[1].motion.rows());
  EXPECT_TRUE(loaded[1].motion.isApprox(saved[1].motion, 1e-5f));
}

// Test: --character converts one character of a multi-character file, with the locator output
TEST_F(ConvertModelIntegrationTest, Result_SelectedCharacter_IsTheOnlyOneConverted) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto input = outputFile("two_select.glb");
  momentum::saveTestSceneAsGltf(input, saved, 30.0f);
  const auto out = outputFile("b.glb");
  const auto outLocators = outputFile("b.locators");

  ASSERT_EQ(
      runConvertModel(
          {"--motion",
           input,
           "--character",
           "b",
           "--out",
           out,
           "--out-locators-local",
           outLocators}),
      0);

  const auto [loaded, fps] = momentum::loadGltfCharactersWithMotion(out);
  ASSERT_EQ(loaded.size(), 1);
  EXPECT_EQ(loaded[0].character.name, "b");
  ASSERT_EQ(loaded[0].motion.cols(), saved[1].motion.cols());
  ASSERT_EQ(loaded[0].motion.rows(), saved[1].motion.rows());
  EXPECT_TRUE(loaded[0].motion.isApprox(saved[1].motion, 1e-5f));
  EXPECT_TRUE(std::filesystem::exists(outLocators));
}

// Test: an option for one character fails for a multi-character file without --character
TEST_F(ConvertModelIntegrationTest, Error_OneCharacterOptionForSeveralCharacters) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto input = outputFile("two_error.glb");
  momentum::saveTestSceneAsGltf(input, saved, 30.0f);
  const auto out = outputFile("two_error_out.glb");

  EXPECT_NE(
      runConvertModel(
          {"--motion", input, "--out", out, "--out-locators-local", outputFile("x.locators")}),
      0);
  EXPECT_FALSE(std::filesystem::exists(out));
}

} // namespace
