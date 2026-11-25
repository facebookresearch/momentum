/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "convert_model_test_helpers.h"

#include <momentum/common/filesystem.h>
#include <momentum/io/bvh/bvh_io.h>
#include <momentum/io/character_io.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/marker/marker_io.h>
#include <momentum/io/skeleton/locator_io.h>
#include <momentum/io/skeleton/parameter_transform_io.h>
#include <momentum/io/urdf/urdf_io.h>
#include <momentum/math/random.h>
#include <momentum/test/character/character_helpers_gtest.h>
#include <momentum/test/io/io_helpers.h>

#include <gtest/gtest.h>

#include <algorithm>
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

#ifdef _WIN32
// `arg` quoted for the Windows command-line parser: in quotes, with the backslashes before a quote
// doubled and the quotes escaped.
std::string quoteArgument(const std::string& arg) {
  if (!arg.empty() && arg.find_first_of(" \t\"") == std::string::npos) {
    return arg;
  }
  std::string result = "\"";
  size_t backslashes = 0;
  for (const char c : arg) {
    if (c == '\\') {
      ++backslashes;
      continue;
    }
    result.append(c == '"' ? 2 * backslashes + 1 : backslashes, '\\');
    backslashes = 0;
    result += c;
  }
  result.append(2 * backslashes, '\\');
  result += '"';
  return result;
}
#endif

std::string fileContents(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

/// Integration test fixture that runs the convert_model, export_rig, apply_motion, split_characters
/// and combine_characters binaries with various arguments and checks their output files.
class ConvertModelIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* convertModelPath = getenv("CONVERT_MODEL_BINARY");
    ASSERT_NE(convertModelPath, nullptr) << "CONVERT_MODEL_BINARY environment variable not set";
    // Make binary path absolute so it works from any CWD
    convertModelPath_ = filesystem::absolute(convertModelPath).string();

    const char* exportRigPath = getenv("EXPORT_RIG_BINARY");
    ASSERT_NE(exportRigPath, nullptr) << "EXPORT_RIG_BINARY environment variable not set";
    exportRigPath_ = filesystem::absolute(exportRigPath).string();

    const char* applyMotionPath = getenv("APPLY_MOTION_BINARY");
    ASSERT_NE(applyMotionPath, nullptr) << "APPLY_MOTION_BINARY environment variable not set";
    applyMotionPath_ = filesystem::absolute(applyMotionPath).string();

    const char* splitCharactersPath = getenv("SPLIT_CHARACTERS_BINARY");
    ASSERT_NE(splitCharactersPath, nullptr)
        << "SPLIT_CHARACTERS_BINARY environment variable not set";
    splitCharactersPath_ = filesystem::absolute(splitCharactersPath).string();

    const char* combineCharactersPath = getenv("COMBINE_CHARACTERS_BINARY");
    ASSERT_NE(combineCharactersPath, nullptr)
        << "COMBINE_CHARACTERS_BINARY environment variable not set";
    combineCharactersPath_ = filesystem::absolute(combineCharactersPath).string();

    const char* testDataPath = getenv("TEST_MOMENTUM_MODELS_PATH");
    ASSERT_NE(testDataPath, nullptr) << "TEST_MOMENTUM_MODELS_PATH environment variable not set";
    testDataPath_ = filesystem::absolute(testDataPath).string();

    // Use Momentum's temporaryDirectory() which handles RE sandbox env vars
    tempDir_ = momentum::temporaryDirectory("convert_model_cli");
  }

  /// Run the convert_model binary with the given arguments.
  /// Returns the process exit code (0 = success).
  int runConvertModel(const std::vector<std::string>& args) {
    return runBinary(convertModelPath_, args);
  }

  /// Run the export_rig binary with the given arguments.
  /// Returns the process exit code (0 = success).
  int runExportRig(const std::vector<std::string>& args) {
    return runBinary(exportRigPath_, args);
  }

  /// Run the apply_motion binary with the given arguments.
  /// Returns the process exit code (0 = success).
  int runApplyMotion(const std::vector<std::string>& args) {
    return runBinary(applyMotionPath_, args);
  }

  /// Run the split_characters binary with the given arguments.
  /// Returns the process exit code (0 = success).
  int runSplitCharacters(const std::vector<std::string>& args) {
    return runBinary(splitCharactersPath_, args);
  }

  /// Run the combine_characters binary with the given arguments.
  /// Returns the process exit code (0 = success).
  int runCombineCharacters(const std::vector<std::string>& args) {
    return runBinary(combineCharactersPath_, args);
  }

  /// Run a binary with the given arguments.
  /// Returns the process exit code (0 = success).
  static int runBinary(const std::string& binary, const std::vector<std::string>& args) {
#ifdef _WIN32
    // On Windows, use _spawnv which doesn't spawn a shell. It joins the arguments with spaces, so
    // each argument is quoted as the Windows command-line parser expects.
    std::vector<std::string> quoted;
    quoted.reserve(args.size() + 1);
    quoted.push_back(quoteArgument(binary));
    for (const auto& arg : args) {
      quoted.push_back(quoteArgument(arg));
    }
    std::vector<const char*> argv;
    argv.reserve(quoted.size() + 1);
    for (const auto& arg : quoted) {
      argv.push_back(arg.c_str());
    }
    argv.push_back(nullptr);
    const int status = _spawnv(_P_WAIT, binary.c_str(), argv.data());
    return status;
#else
    // Build argv for execvp (no shell spawned)
    std::vector<const char*> argv;
    argv.push_back(binary.c_str());
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
      execvp(binary.c_str(), const_cast<char* const*>(argv.data()));
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

  /// Converts `input` (glTF with motion) to BVH and back, and checks that every joint has the same
  /// global transform in every frame. Also converts the BVH file as a model.
  void expectBvhRoundTripKeepsTheJointTransforms(const std::string& input);

  /// Helper to construct a path to a temporary output file.
  [[nodiscard]] std::string outputFile(const std::string& filename) const {
    return (tempDir_.path() / filename).string();
  }

  std::string convertModelPath_;
  std::string exportRigPath_;
  std::string applyMotionPath_;
  std::string splitCharactersPath_;
  std::string combineCharactersPath_;
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

// Test: no input file fails and writes no output
TEST_F(ConvertModelIntegrationTest, Error_NoInputFile) {
  const auto out = outputFile("no_input.glb");
  EXPECT_NE(runConvertModel({"--out", out}), 0);
  EXPECT_FALSE(std::filesystem::exists(out));
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

  const std::string paramsData = fileContents(outParams);
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

// Every character of a glTF or FBX file, as the multi-character loader of its format reads it.
std::vector<momentum::Character> loadEveryCharacter(const std::string& path) {
  if (momentum::parseCharacterFormat(path) == momentum::CharacterFormat::Fbx) {
    return momentum::loadFbxCharacters(
        path, momentum::KeepLocators::Yes, momentum::Permissive::No, momentum::LoadBlendShapes::No);
  }
  return momentum::loadGltfCharacters(filesystem::path(path));
}

void expectSamePhysicalPropertiesAndMetadata(
    const std::vector<momentum::Character>& expected,
    const std::vector<momentum::Character>& loaded) {
  ASSERT_EQ(loaded.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE(expected[i].name);
    EXPECT_EQ(loaded[i].name, expected[i].name);
    EXPECT_EQ(loaded[i].metadata, expected[i].metadata);
    momentum::comparePhysicalProperties(
        expected[i].physicalProperties, loaded[i].physicalProperties);
  }
}

// Test: the physical properties and metadata of every character are kept, for one and for two
// characters, read from the model file or from the motion file, between glTF and FBX
TEST_F(ConvertModelIntegrationTest, Result_PhysicalPropertiesAndMetadata_AreKept) {
  for (const size_t numCharacters : {1, 2}) {
    auto saved = momentum::createTwoCharacterScene();
    saved.resize(numCharacters);
    std::vector<momentum::Character> expected;
    expected.reserve(saved.size());
    for (const auto& entry : saved) {
      expected.push_back(entry.character);
    }
    for (const std::string inputExtension : {".glb", ".fbx"}) {
      const auto input = outputFile("properties_" + std::to_string(numCharacters) + inputExtension);
      if (inputExtension == ".glb") {
        momentum::saveTestSceneAsGltf(input, saved, 30.0f);
      } else {
        momentum::saveTestSceneAsFbx(input, saved, 30.0f);
      }
      for (const std::string inputOption : {"--model", "--motion"}) {
        for (const std::string outputExtension : {".glb", ".fbx"}) {
          SCOPED_TRACE(
              ::testing::Message() << input << " " << inputOption << " -> " << outputExtension);
          std::string outName = "properties_out_";
          outName += std::to_string(numCharacters);
          outName += inputExtension.substr(1);
          outName += inputOption.substr(2);
          outName += outputExtension;
          const auto out = outputFile(outName);
          ASSERT_EQ(runConvertModel({inputOption, input, "--out", out}), 0);
          expectSamePhysicalPropertiesAndMetadata(expected, loadEveryCharacter(out));
        }
      }
    }
  }
}

// Test: a selected character keeps its physical properties and metadata
TEST_F(ConvertModelIntegrationTest, Result_SelectedCharacter_KeepsItsPhysicalProperties) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto input = outputFile("properties_two.glb");
  momentum::saveTestSceneAsGltf(input, saved, 30.0f);
  const auto out = outputFile("properties_b.fbx");

  ASSERT_EQ(runConvertModel({"--motion", input, "--character", "b", "--out", out}), 0);

  expectSamePhysicalPropertiesAndMetadata({saved[1].character}, loadEveryCharacter(out));
}

// Test: the physical properties of a URDF model are kept in glTF and FBX outputs
TEST_F(ConvertModelIntegrationTest, Result_UrdfModel_KeepsItsPhysicalProperties) {
  const auto urdf = outputFile("inertial.urdf");
  std::ofstream(urdf) << R"(<?xml version="1.0"?>
<robot name="inertial_robot">
  <link name="base_link">
    <inertial>
      <origin xyz="0.01 0.02 0.03" rpy="0 0 0.2"/>
      <mass value="2.0"/>
      <inertia ixx="0.10" ixy="0.01" ixz="0.00" iyy="0.20" iyz="0.02" izz="0.30"/>
    </inertial>
  </link>
  <link name="tip_link">
    <inertial>
      <origin xyz="0 0 0.01" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.00" ixz="0.00" iyy="0.02" iyz="0.00" izz="0.03"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="tip_link"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="100" velocity="1"/>
  </joint>
</robot>
)";
  const momentum::Character expected = momentum::loadUrdfCharacter(urdf);
  ASSERT_EQ(expected.physicalProperties.size(), 2);

  for (const std::string outputExtension : {".glb", ".fbx"}) {
    SCOPED_TRACE(outputExtension);
    const auto out = outputFile("inertial_out" + outputExtension);
    ASSERT_EQ(runConvertModel({"--model", urdf, "--out", out}), 0);
    const auto loaded = loadEveryCharacter(out);
    ASSERT_EQ(loaded.size(), 1);
    momentum::comparePhysicalProperties(expected.physicalProperties, loaded[0].physicalProperties);
  }

  // The extension can have any letter case. The file has another name, because inertial.urdf and
  // INERTIAL.URDF are the same file on file systems that ignore the letter case.
  const auto upperCaseUrdf = outputFile("UPPER_CASE.URDF");
  std::filesystem::copy_file(urdf, upperCaseUrdf);
  const auto out = outputFile("inertial_upper_case.glb");
  ASSERT_EQ(runConvertModel({"--model", upperCaseUrdf, "--out", out}), 0);
  momentum::comparePhysicalProperties(
      expected.physicalProperties, loadEveryCharacter(out).at(0).physicalProperties);
}

// Test: a character with its motion goes from glTF to BVH and back with the same joint transforms,
// and the BVH file converts as a model
TEST_F(ConvertModelIntegrationTest, Result_GlbToBvhToGlb_KeepsTheJointTransforms) {
  // A test file, and a file whose motion has identity offsets, which BVH cannot store separately.
  const auto withIdentity = outputFile("with_identity.glb");
  {
    auto [character, motion] = momentum::createTestSceneCharacter("a", 3, 4, 0.0f);
    momentum::removeScaleMotion(character.parameterTransform, motion);
    momentum::VectorXf identity =
        momentum::VectorXf::Zero(character.parameterTransform.numJointParameters());
    identity(momentum::kParametersPerJoint + momentum::TY) = 2.0f;
    momentum::saveGltfCharacter(
        withIdentity,
        character,
        30.0f,
        {character.parameterTransform.name, motion},
        {character.skeleton.getJointNames(), identity});
  }
  for (const auto& input : {testFile("character_with_motion.glb"), withIdentity}) {
    SCOPED_TRACE(input);
    expectBvhRoundTripKeepsTheJointTransforms(input);
  }
}

void ConvertModelIntegrationTest::expectBvhRoundTripKeepsTheJointTransforms(
    const std::string& input) {
  const auto stem = filesystem::path(input).stem().string();
  const auto bvh = outputFile(stem + ".bvh");
  const auto back = outputFile(stem + "_from_bvh.glb");
  const auto model = outputFile(stem + "_bvh_model.glb");

  ASSERT_EQ(runConvertModel({"--motion", input, "--out", bvh}), 0);
  ASSERT_EQ(runConvertModel({"--motion", bvh, "--out", back}), 0);
  ASSERT_EQ(runConvertModel({"--model", bvh, "--out", model}), 0);

  const auto [character, motion, identity, fps] = momentum::loadCharacterWithMotion(input);
  const auto [backCharacter, backMotion, backIdentity, backFps] =
      momentum::loadCharacterWithMotion(back);
  ASSERT_EQ(backMotion.cols(), motion.cols());
  EXPECT_NEAR(backFps, fps, 1e-3f);
  const auto difference = momentum::maxGlobalJointTransformDifference(
      character.skeleton,
      momentum::jointMotionOf(character.parameterTransform, motion, identity.v),
      backCharacter.skeleton,
      momentum::jointMotionOf(backCharacter.parameterTransform, backMotion, backIdentity.v));
  EXPECT_LT(difference.translation, 1e-2f);
  EXPECT_LT(difference.rotation, 1e-3f);
  EXPECT_EQ(
      momentum::loadGltfCharacter(model).skeleton.getJointNames(),
      character.skeleton.getJointNames());
}

// Test: with a parameter file whose transform has constant offsets, BVH motion on a model gives
// the same joint transforms in BVH and glTF outputs
TEST_F(ConvertModelIntegrationTest, Result_ParameterTransformOffsets_SameJointsInBvhAndGlb) {
  momentum::Character character = momentum::createTestCharacter(3);
  auto& parameterTransform = character.parameterTransform;
  // Offsets of joint parameters that model parameters drive, which .model files store.
  parameterTransform.offsets(momentum::TY) = 1.5f;
  parameterTransform.offsets(2 * momentum::kParametersPerJoint + momentum::RX) = 0.25f;
  const auto parameters = outputFile("offsets.model");
  {
    std::ofstream file(parameters);
    file << momentum::writeModelDefinition(
        character.skeleton, parameterTransform, character.parameterLimits);
  }
  const auto model = outputFile("offsets_model.glb");
  momentum::saveGltfCharacter(model, character);
  // A motion that the parameter transform can express exactly.
  momentum::Random<> rng(12345);
  auto motion =
      rng.uniform<momentum::MatrixXf>(parameterTransform.numAllModelParameters(), 3, -0.3f, 0.3f);
  momentum::removeScaleMotion(parameterTransform, motion);
  const momentum::MatrixXf jointMotion =
      momentum::jointMotionOf(parameterTransform, motion, parameterTransform.offsets);
  const auto bvh = outputFile("offsets_motion.bvh");
  momentum::saveBvhCharacterWithJointParams(bvh, character, jointMotion, 30.0f);

  const auto bvhOut = outputFile("offsets_out.bvh");
  const auto glbOut = outputFile("offsets_out.glb");
  for (const auto& out : {bvhOut, glbOut}) {
    ASSERT_EQ(
        runConvertModel(
            {"--model", model, "--parameters", parameters, "--motion", bvh, "--out", out}),
        0);
  }

  const auto fromBvh = momentum::loadBvhCharacterWithMotion(bvhOut);
  const auto bvhDifference = momentum::maxGlobalJointTransformDifference(
      character.skeleton,
      jointMotion,
      std::get<0>(fromBvh).skeleton,
      std::get<1>(std::get<1>(fromBvh)));
  EXPECT_LT(bvhDifference.translation, 1e-3f);
  EXPECT_LT(bvhDifference.rotation, 1e-3f);

  const auto [glbCharacter, glbMotion, glbIdentity, glbFps] =
      momentum::loadCharacterWithMotion(glbOut);
  const auto glbDifference = momentum::maxGlobalJointTransformDifference(
      character.skeleton,
      jointMotion,
      glbCharacter.skeleton,
      momentum::jointMotionOf(glbCharacter.parameterTransform, glbMotion, glbIdentity.v));
  EXPECT_LT(glbDifference.translation, 1e-3f);
  EXPECT_LT(glbDifference.rotation, 1e-3f);
}

// Test: a BVH file holds one character, so the input needs one or --character
TEST_F(ConvertModelIntegrationTest, Error_SeveralCharactersToBvh) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto input = outputFile("two_for_bvh.glb");
  momentum::saveTestSceneAsGltf(input, saved, 30.0f);

  EXPECT_NE(runConvertModel({"--motion", input, "--out", outputFile("two.bvh")}), 0);
  const auto out = outputFile("b.bvh");
  ASSERT_EQ(runConvertModel({"--motion", input, "--character", "b", "--out", out}), 0);
  EXPECT_EQ(
      momentum::loadBvhCharacter(out).skeleton.getJointNames(),
      saved[1].character.skeleton.getJointNames());
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

// ============================================================================
// EXPORT_RIG TESTS
// ============================================================================

// Test: export_rig writes the parameter transform, limits and locators of its input, in the files
// that the deprecated convert_model options write
TEST_F(ConvertModelIntegrationTest, ExportRig_WritesTheRigOfTheInput) {
  const auto model = testFile("character_with_motion.glb");
  const auto parameters = testFile("character.model");
  const auto locators = testFile("character.locators");
  const auto argsWithOutputs = [&](const std::string& prefix) {
    return std::vector<std::string>{
        "--model",
        model,
        "--parameters",
        parameters,
        "--locator",
        locators,
        "--out-parameters",
        outputFile(prefix + ".model"),
        "--out-locators-local",
        outputFile(prefix + "_local.locators"),
        "--out-locators-global",
        outputFile(prefix + "_global.locators")};
  };

  ASSERT_EQ(runExportRig(argsWithOutputs("rig")), 0);
  auto convertArgs = argsWithOutputs("deprecated");
  convertArgs.emplace_back("--out");
  convertArgs.emplace_back(outputFile("deprecated.glb"));
  ASSERT_EQ(runConvertModel(convertArgs), 0);

  for (const std::string suffix : {".model", "_local.locators", "_global.locators"}) {
    SCOPED_TRACE(suffix);
    ASSERT_FALSE(fileContents(outputFile("rig" + suffix)).empty());
    EXPECT_EQ(
        fileContents(outputFile("rig" + suffix)), fileContents(outputFile("deprecated" + suffix)));
  }

  const momentum::Skeleton skeleton = momentum::loadGltfCharacter(model).skeleton;
  const auto [expectedTransform, expectedLimits] =
      momentum::loadModelDefinition(parameters, skeleton);
  const auto [transform, limits] = momentum::loadModelDefinition(outputFile("rig.model"), skeleton);
  EXPECT_EQ(transform.name, expectedTransform.name);
  EXPECT_TRUE(transform.transform.isApprox(expectedTransform.transform));
  EXPECT_EQ(limits, expectedLimits);
  const auto expectedLocators = momentum::loadLocators(locators, skeleton, expectedTransform);
  ASSERT_FALSE(expectedLocators.empty());
  EXPECT_EQ(
      momentum::loadLocators(outputFile("rig_local.locators"), skeleton, expectedTransform),
      expectedLocators);
  EXPECT_EQ(
      momentum::loadLocators(outputFile("rig_global.locators"), skeleton, expectedTransform),
      expectedLocators);
}

// Test: export_rig exports one character of an input with several characters, selected by name
TEST_F(ConvertModelIntegrationTest, ExportRig_SeveralCharacters_NeedCharacter) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto input = outputFile("two_for_rig.glb");
  momentum::saveTestSceneAsGltf(input, saved, 30.0f);
  const auto out = outputFile("b.model");

  EXPECT_NE(runExportRig({"--model", input, "--out-parameters", out}), 0);
  ASSERT_EQ(runExportRig({"--model", input, "--character", "b", "--out-parameters", out}), 0);
  ASSERT_TRUE(std::filesystem::exists(out));
  const auto& b = saved[1].character;
  EXPECT_EQ(
      std::get<0>(momentum::loadModelDefinition(out, b.skeleton)).name, b.parameterTransform.name);
}

// Test: export_rig without an output fails
TEST_F(ConvertModelIntegrationTest, ExportRig_NoOutput_Fails) {
  EXPECT_NE(runExportRig({"--model", testFile("character_with_motion.glb")}), 0);
}

} // namespace

// ============================================================================
// APPLY_MOTION TESTS
// ============================================================================

// Test: apply_motion writes the file that the deprecated convert_model --model with --motion
// writes with the same options
TEST_F(ConvertModelIntegrationTest, ApplyMotion_WritesTheFileOfConvertModelWithModelAndMotion) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto models = outputFile("apply_models.glb");
  momentum::saveTestSceneAsGltf(models, saved, 30.0f);
  // The motion file has other motions than the model file, and markers.
  const auto motions = outputFile("apply_motions.glb");
  momentum::GltfBuilder builder;
  builder.setFps(30.0f);
  for (const auto& [character, motion] : saved) {
    const std::string name = builder.addCharacter(character);
    builder.addMotion(
        character,
        30.0f,
        {character.parameterTransform.name, 0.5f * motion},
        {},
        true,
        "default",
        {},
        name);
  }
  const std::vector<std::vector<momentum::Marker>> markers(
      4, {momentum::Marker{.name = "marker", .pos = {1.0, 2.0, 3.0}, .occluded = false}});
  builder.addMarkerSequence(30.0f, markers);
  builder.save(motions);

  const std::vector<std::vector<std::string>> inputs = {
      {"--model",
       testFile("character_no_motion.fbx"),
       "--parameters",
       testFile("character.model"),
       "--locator",
       testFile("character.locators"),
       "--motion",
       testFile("character_with_motion.glb")},
      {"--model", models, "--motion", motions, "--save-markers"},
      {"--model", models, "--motion", motions, "--character", "b"},
  };
  for (size_t i = 0; i < inputs.size(); ++i) {
    SCOPED_TRACE(i);
    const auto applied = outputFile("applied_" + std::to_string(i) + ".glb");
    const auto converted = outputFile("converted_" + std::to_string(i) + ".glb");
    auto args = inputs[i];
    args.insert(args.end(), {"--out", applied});
    ASSERT_EQ(runApplyMotion(args), 0);
    args.back() = converted;
    ASSERT_EQ(runConvertModel(args), 0);
    ASSERT_FALSE(fileContents(applied).empty());
    EXPECT_EQ(fileContents(applied), fileContents(converted));
  }

  // FBX files store the save time, so they are compared through their glTF conversions.
  std::vector<std::string> fbxArgs = {
      "--model",
      models,
      "--motion",
      motions,
      "--character-mesh",
      "--fbx-namespace",
      "ns42",
      "--out",
      outputFile("applied.fbx")};
  ASSERT_EQ(runApplyMotion(fbxArgs), 0);
  fbxArgs.back() = outputFile("converted.fbx");
  ASSERT_EQ(runConvertModel(fbxArgs), 0);
  EXPECT_NE(fileContents(outputFile("applied.fbx")).find("ns42:"), std::string::npos);
  for (const std::string name : {"applied", "converted"}) {
    ASSERT_EQ(
        runConvertModel(
            {"--motion", outputFile(name + ".fbx"), "--out", outputFile(name + "_fbx.glb")}),
        0);
  }
  EXPECT_EQ(
      fileContents(outputFile("applied_fbx.glb")), fileContents(outputFile("converted_fbx.glb")));
}

// Test: apply_motion fails for a motion file of an unknown format
TEST_F(ConvertModelIntegrationTest, ApplyMotion_UnknownMotionFormat_Fails) {
  const auto out = outputFile("unknown_motion.glb");
  EXPECT_NE(
      runApplyMotion(
          {"--model",
           testFile("character_with_motion.glb"),
           "--motion",
           testFile("character.locators"),
           "--out",
           out}),
      0);
  EXPECT_FALSE(std::filesystem::exists(out));
}

// Test: the parameter and locator files apply to the character of a glTF motion file, as with the
// same file as the model file and the motion file
TEST_F(ConvertModelIntegrationTest, ParametersAndLocators_ApplyToTheCharacterOfAGltfMotionFile) {
  const auto saved = momentum::createTestSceneCharacter("a", 3, 4, 0.0f);
  const auto& skeleton = saved.character.skeleton;
  const auto input = outputFile("rig_input.glb");
  momentum::saveTestSceneAsGltf(input, {saved}, 30.0f);
  // A new rig, with other parameter names than the rig in the input file.
  const auto identity = momentum::ParameterTransform::identity(skeleton.getJointNames());
  const auto parameters = outputFile("identity.model");
  {
    std::ofstream file(parameters);
    file << momentum::writeModelDefinition(skeleton, identity, {});
  }
  const auto locators = testFile("character.locators");
  const auto expectedLocators = momentum::loadLocators(locators, skeleton, identity);
  ASSERT_NE(expectedLocators, saved.character.locators);

  const auto out = outputFile("new_rig.glb");
  ASSERT_EQ(
      runConvertModel(
          {"--motion", input, "--parameters", parameters, "--locator", locators, "--out", out}),
      0);
  const auto converted = momentum::loadGltfCharacter(out);
  EXPECT_EQ(converted.parameterTransform.name, identity.name);
  EXPECT_EQ(converted.locators, expectedLocators);

  const auto sameFile = outputFile("new_rig_same_file.glb");
  ASSERT_EQ(
      runConvertModel(
          {"--model",
           input,
           "--motion",
           input,
           "--parameters",
           parameters,
           "--locator",
           locators,
           "--out",
           sameFile}),
      0);
  EXPECT_EQ(fileContents(out), fileContents(sameFile));
}

// ============================================================================
// SPLIT_CHARACTERS TESTS
// ============================================================================

// Test: split_characters writes each character to the file that convert_model --character writes
TEST_F(ConvertModelIntegrationTest, SplitCharacters_WritesTheFileOfConvertModelWithCharacter) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto input = outputFile("to_split.glb");
  momentum::saveTestSceneAsGltf(input, saved, 30.0f);

  for (const std::string format : {"glb", "bvh"}) {
    SCOPED_TRACE(format);
    const auto directory = outputFile("split_" + format);
    ASSERT_EQ(
        runSplitCharacters({"--input", input, "--out-dir", directory, "--format", format}), 0);
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
      files.push_back(entry.path().filename().string());
    }
    std::ranges::sort(files);
    EXPECT_EQ(files, (std::vector<std::string>{"a." + format, "b." + format}));
    for (const auto& [character, motion] : saved) {
      const auto expected = outputFile("expected_" + character.name + "." + format);
      ASSERT_EQ(
          runConvertModel({"--motion", input, "--character", character.name, "--out", expected}),
          0);
      const auto split = (std::filesystem::path(directory) / (character.name + "." + format));
      ASSERT_FALSE(fileContents(expected).empty());
      EXPECT_EQ(fileContents(split.string()), fileContents(expected));
    }
  }
}

// Test: the file names of split_characters are safe and unique, in any letter case, and the
// characters keep their names
TEST_F(ConvertModelIntegrationTest, SplitCharacters_FileNamesAreSafeAndUnique) {
  const std::vector<std::string> names = {
      "hand/left", "hand left", "", "hand", "Hand", "NUL", "com1.left"};
  std::vector<momentum::TestSceneCharacter> saved;
  saved.reserve(names.size());
  for (const auto& name : names) {
    saved.push_back(momentum::createTestSceneCharacter(name, 3, 2, 0.0f));
  }
  const auto input = outputFile("names_to_split.glb");
  momentum::saveTestSceneAsGltf(input, saved, 30.0f);
  const auto directory = outputFile("split_names");
  ASSERT_EQ(runSplitCharacters({"--input", input, "--out-dir", directory}), 0);

  const std::vector<std::string> files = {
      "hand_left.glb",
      "hand_left_1.glb",
      "character_2.glb",
      "hand.glb",
      "Hand_1.glb",
      "NUL_.glb",
      "com1_.left.glb"};
  size_t numFiles = 0;
  for ([[maybe_unused]] const auto& entry : std::filesystem::directory_iterator(directory)) {
    ++numFiles;
  }
  EXPECT_EQ(numFiles, files.size());
  for (size_t i = 0; i < files.size(); ++i) {
    SCOPED_TRACE(files[i]);
    const auto path = std::filesystem::path(directory) / files[i];
    ASSERT_TRUE(std::filesystem::exists(path));
    EXPECT_EQ(momentum::loadGltfCharacter(path).name, names[i]);
  }
}

// Test: split_characters takes the format with or without the dot, in any letter case
TEST_F(ConvertModelIntegrationTest, SplitCharacters_FormatWithADot) {
  const auto input = outputFile("dot_to_split.glb");
  momentum::saveTestSceneAsGltf(input, momentum::createTwoCharacterScene(), 30.0f);
  const auto directory = outputFile("split_dot");
  ASSERT_EQ(runSplitCharacters({"--input", input, "--out-dir", directory, "--format", ".GLB"}), 0);
  std::vector<std::string> files;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    files.push_back(entry.path().filename().string());
  }
  std::ranges::sort(files);
  EXPECT_EQ(files, (std::vector<std::string>{"a.glb", "b.glb"}));
}

// Test: split_characters fails for an unknown output format
TEST_F(ConvertModelIntegrationTest, SplitCharacters_UnknownFormat_Fails) {
  const auto directory = outputFile("split_obj");
  EXPECT_NE(
      runSplitCharacters(
          {"--input",
           testFile("character_with_motion.glb"),
           "--out-dir",
           directory,
           "--format",
           "obj"}),
      0);
  EXPECT_FALSE(std::filesystem::exists(directory));
}

// ============================================================================
// COMBINE_CHARACTERS TESTS
// ============================================================================

// Test: combine_characters writes the file that convert_model writes for one file with the
// characters of all inputs
TEST_F(ConvertModelIntegrationTest, CombineCharacters_WritesTheFileOfConvertModelForAllCharacters) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto both = outputFile("both.glb");
  momentum::saveTestSceneAsGltf(both, saved, 30.0f);
  const auto a = outputFile("only_a.glb");
  momentum::saveTestSceneAsGltf(a, {saved[0]}, 30.0f);
  const auto b = outputFile("only_b.glb");
  momentum::saveTestSceneAsGltf(b, {saved[1]}, 30.0f);

  const auto combined = outputFile("combined.glb");
  ASSERT_EQ(runCombineCharacters({"--input", a, "--input", b, "--out", combined}), 0);
  const auto expected = outputFile("expected_both.glb");
  ASSERT_EQ(runConvertModel({"--motion", both, "--out", expected}), 0);
  ASSERT_FALSE(fileContents(expected).empty());
  EXPECT_TRUE(fileContents(combined) == fileContents(expected));

  // An FBX input gives the characters that convert_model reads from it.
  const auto bFbx = outputFile("only_b.fbx");
  momentum::saveTestSceneAsFbx(bFbx, {saved[1]}, 30.0f);
  const auto bothFromFbx = outputFile("both_from_fbx.glb");
  ASSERT_EQ(runCombineCharacters({"--input", a, "--input", bFbx, "--out", bothFromFbx}), 0);
  const auto [entries, fps] = momentum::loadGltfCharactersWithMotion(bothFromFbx);
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[1].character.name, "b");
  EXPECT_EQ(entries[1].motion.cols(), saved[1].motion.cols());
}

// Test: inputs with different frame rates are an error without --resample; with it, the motion is
// resampled to the frame rate of the first input, or to --fps
TEST_F(ConvertModelIntegrationTest, CombineCharacters_DifferentFrameRates_NeedResample) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto a = outputFile("a_30.glb");
  momentum::saveTestSceneAsGltf(a, {saved[0]}, 30.0f);
  const auto b = outputFile("b_60.glb");
  momentum::saveTestSceneAsGltf(b, {saved[1]}, 60.0f);

  const auto out = outputFile("resampled.glb");
  EXPECT_NE(runCombineCharacters({"--input", a, "--input", b, "--out", out}), 0);
  EXPECT_FALSE(std::filesystem::exists(out));
  EXPECT_NE(runCombineCharacters({"--input", a, "--input", b, "--fps", "60", "--out", out}), 0);

  // At half the frame rate, the frames are every other input frame; at twice the frame rate, the
  // input frames with the mean of each two neighbors between them.
  const auto everyOtherFrame = [](const Eigen::MatrixXf& motion) {
    Eigen::MatrixXf result(motion.rows(), (motion.cols() + 1) / 2);
    for (Eigen::Index frame = 0; frame < result.cols(); ++frame) {
      result.col(frame) = motion.col(2 * frame);
    }
    return result;
  };
  const auto withMidpoints = [](const Eigen::MatrixXf& motion) {
    Eigen::MatrixXf result(motion.rows(), 2 * motion.cols() - 1);
    for (Eigen::Index frame = 0; frame < motion.cols(); ++frame) {
      result.col(2 * frame) = motion.col(frame);
      if (frame + 1 < motion.cols()) {
        result.col(2 * frame + 1) = 0.5f * (motion.col(frame) + motion.col(frame + 1));
      }
    }
    return result;
  };
  struct Case {
    std::vector<std::string> fpsArgs;
    float fps;
    Eigen::MatrixXf motionA;
    Eigen::MatrixXf motionB;
  };
  for (const auto& [fpsArgs, fps, motionA, motionB] :
       {Case{
            .fpsArgs = {},
            .fps = 30.0f,
            .motionA = saved[0].motion,
            .motionB = everyOtherFrame(saved[1].motion)},
        Case{
            .fpsArgs = {"--fps", "60"},
            .fps = 60.0f,
            .motionA = withMidpoints(saved[0].motion),
            .motionB = saved[1].motion}}) {
    SCOPED_TRACE(fps);
    std::vector<std::string> args = {"--input", a, "--input", b, "--resample", "--out", out};
    args.insert(args.end(), fpsArgs.begin(), fpsArgs.end());
    ASSERT_EQ(runCombineCharacters(args), 0);
    const auto [entries, loadedFps] = momentum::loadGltfCharactersWithMotion(out);
    EXPECT_EQ(loadedFps, fps);
    ASSERT_EQ(entries.size(), 2);
    EXPECT_TRUE(entries[0].motion.isApprox(motionA, 1e-5f)) << entries[0].motion;
    EXPECT_TRUE(entries[1].motion.isApprox(motionB, 1e-5f)) << entries[1].motion;
  }
}

// Test: a character with the name of an earlier character is renamed with a numeric suffix
TEST_F(ConvertModelIntegrationTest, CombineCharacters_SameNames_AreMadeUnique) {
  const auto saved = momentum::createTwoCharacterScene();
  auto renamed = saved[1];
  renamed.character.name = "a";
  const auto first = outputFile("first_a.glb");
  momentum::saveTestSceneAsGltf(first, {saved[0]}, 30.0f);
  const auto second = outputFile("second_a.glb");
  momentum::saveTestSceneAsGltf(second, {renamed}, 30.0f);

  const auto out = outputFile("unique_names.glb");
  ASSERT_EQ(runCombineCharacters({"--input", first, "--input", second, "--out", out}), 0);
  const auto characters = momentum::loadGltfCharacters(out);
  ASSERT_EQ(characters.size(), 2);
  EXPECT_EQ(characters[0].name, "a");
  EXPECT_EQ(characters[1].name, "a_1");
  EXPECT_EQ(characters[1].skeleton.joints.size(), renamed.character.skeleton.joints.size());
}

// Test: the markers of a marker file are saved when their frame rate is the frame rate of the
// output
TEST_F(ConvertModelIntegrationTest, CombineCharacters_Markers_NeedTheFrameRateOfTheOutput) {
  const auto saved = momentum::createTwoCharacterScene();
  const auto a = outputFile("a_markers.glb");
  momentum::saveTestSceneAsGltf(a, {saved[0]}, 30.0f);
  const std::vector<std::vector<momentum::Marker>> frames(
      5, {momentum::Marker{.name = "marker", .pos = {1.0, 2.0, 3.0}, .occluded = false}});
  const auto markersAt = [&](float fps) {
    const auto path = outputFile("markers_" + std::to_string(static_cast<int>(fps)) + ".glb");
    momentum::GltfBuilder builder;
    builder.addMarkerSequence(fps, frames);
    builder.save(path);
    return path;
  };

  const auto out = outputFile("with_markers.glb");
  ASSERT_EQ(runCombineCharacters({"--input", a, "--markers", markersAt(30.0f), "--out", out}), 0);
  const auto markers = momentum::loadMarkers(out);
  ASSERT_EQ(markers.size(), 1);
  EXPECT_EQ(markers[0].frames.size(), frames.size());

  EXPECT_NE(
      runCombineCharacters(
          {"--input", a, "--markers", markersAt(100.0f), "--out", outputFile("bad.glb")}),
      0);
  EXPECT_FALSE(std::filesystem::exists(outputFile("bad.glb")));
  ASSERT_EQ(
      runCombineCharacters(
          {"--input",
           a,
           "--markers",
           markersAt(100.0f),
           "--resample",
           "--fps",
           "100",
           "--out",
           outputFile("resampled_markers.glb")}),
      0);
}
