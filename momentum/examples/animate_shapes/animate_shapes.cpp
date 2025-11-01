/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/blend_shape.h>
#include <momentum/character/character.h>
#include <momentum/common/log.h>
#include <momentum/io/character_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/shape/blend_shape_io.h>

#include <CLI/CLI.hpp>

#include <numeric>

using namespace momentum;

namespace {

std::vector<float> linspace(float a, float b, size_t N) {
  float h = (b - a) / static_cast<float>(N - 1);
  std::vector<float> xs(N);
  typename std::vector<float>::iterator x;
  float val = a;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
    *x = val;
  }
  return xs;
}

struct Options {
  std::string modelFile;
  std::string blendshapeFile;
  size_t numShapes = 10;
  std::string outFile;
};

std::shared_ptr<Options> setupOptions(CLI::App& app) {
  auto opt = std::make_shared<Options>();
  app.add_option("-m,--model", opt->modelFile, "Path to the .fbx/.glb model file")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-s,--shapes", opt->blendshapeFile, "Path to the .bin blendshape file")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-n,--num", opt->numShapes, "Number of blendshapes to use")
      ->default_val(opt->numShapes)
      ->check(CLI::PositiveNumber);
  app.add_option("-o,--output", opt->outFile, "Path to the output .glb file")->required();
  return opt;
}

} // namespace

int main(int argc, char* argv[]) {
  try {
    CLI::App app("Animate Blendshapes");
    auto options = setupOptions(app);
    CLI11_PARSE(app, argc, argv);

    // load the character model and blendshapes
    Character character = loadFullCharacter(options->modelFile);
    character.addBlendShape(
        std::make_shared<BlendShape>(loadBlendShape(options->blendshapeFile)), options->numShapes);
    const ParameterTransform& pt = character.parameterTransform;

    // animate the blendshape coefficients one by one
    const size_t kFramesPerShape = 40;
    const size_t numFrames = options->numShapes * kFramesPerShape;
    std::vector<float> values;
    // go from 0 to 4, then 4 to -4, then back to 0
    values = linspace(0.0, 4.0, 10);
    auto newvec = linspace(4.0, -4.0, 20);
    values.insert(values.end(), newvec.begin(), newvec.end());
    newvec = linspace(-4.0, 0.0, 10);
    values.insert(values.end(), newvec.begin(), newvec.end());

    // create the animation
    MatrixXf motion(pt.numAllModelParameters(), numFrames);
    motion.setZero();
    for (size_t iShape = 0; iShape < options->numShapes; ++iShape) {
      motion.row(iShape).segment(iShape * kFramesPerShape, kFramesPerShape) =
          Eigen::Map<Eigen::VectorXf>(values.data(), values.size());
    }

    // save the result
    saveGltfCharacter(
        options->outFile,
        character,
        20.f,
        {pt.name, motion},
        {character.skeleton.getJointNames(), pt.zero().v});
  } catch (const std::exception& e) {
    MT_LOGE("Exception thrown. Error: {}", e.what());
    return EXIT_FAILURE;
  } catch (...) {
    MT_LOGE("Exception thrown. Unknown error.");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
