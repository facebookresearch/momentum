/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pymomentum/tensor_utility/tensor_utility.h>

#include <momentum/character_solver/skeleton_error_function.h>

#include <ATen/ATen.h>

#include <map>

namespace pymomentum {

// "Sentinel" tensors fill in for the input arguments to the
// autograd::Function::forward that aren't tensors.  We need
// to save a spot for them in the backward pass to fill in with
// an empty tensor, so we'll make sure to record them in the
// _tensorInputs list.
// TODO rename SENTINEL -> PLACEHOLDER
enum class TensorType { TYPE_FLOAT, TYPE_INT, TYPE_SENTINEL };

// Wraps up a given tensor input so it's usable to create constraints.
struct TensorInput {
  // For any differentiable input, should match the input name to pass into
  // FullyDifferentiableErrorFunction::getInput()/setInput()
  // For any non-differentiable input, should be descriptive (in the sense
  // that it would make sense in an error function).
  const char* inputName = nullptr;

  // The tensor with input values for the entire batch.
  // Can be batched or non-batched; the software will ensure
  // either gets handled correctly.
  at::Tensor tensor;

  // Tensor dimensions that are expected in the createErrorFunction()
  // function. Should not include the batch dimension.
  //
  // Negative values indicate special "shared" dimensions between tensor
  // inputs that should be validated.  Example: If you have an error
  // function that includes targets and weights, you could pass the
  // dimensions:
  //   targets: {-1, 3}
  //   weights: {-1}
  // We will then verify that the weights dimension matches the targets
  // dimension.
  std::vector<int> targetSizes;

  // Whether the tensor is a float-valued or int-valued tensor:
  TensorType targetType = TensorType::TYPE_FLOAT;

  // Whether the input is differentiable:
  enum Differentiability { DIFFERENTIABLE, NON_DIFFERENTIABLE } differentiability = DIFFERENTIABLE;

  // Whether the input is allowed to be empty.
  enum Optionality { REQUIRED, OPTIONAL } optionality = REQUIRED;

  template <typename T>
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> toEigenMap(size_t iBatch, size_t iFrame) const;

  bool isTensorEmpty() const {
    return ::pymomentum::isEmpty(tensor);
  }
};

// Base class to represent a wrapper of momentum's skeleton error function
// for PyTorch.  It is input to TensorIK.h:solveTensorIKProblem() and
// TensorIK.h:d_solveTensorIKProblem().
//
// A TensorErrorFunction has the following responsibilities:
//   1. Check the Tensor inputs and validate that they all have
//      correct dimensions.
//   2. Create the actual momentum SkeletonErrorFunction that
//      will be used in the solve.  This requires deciding which
//      tensor inputs are batched and extracting a particular
//      batch index where appropriate.
//
// A note regarding "empty" error functions: The use of
// TensorErrorFunctions is from a Python function that looks like
// this:
//    solveIK(character, ..., posError_input, orientError_input,
//    posePrior_input, ...)
// To be able to allow all kinds of error functions, we have to make
// solverIK() to accept all parameters of all error functions.  But
// the user might not need all error functions and choose to pass
// in some of the named inputs (and build only some of the error functions
// to the IK solver).  Thus, a TensorErrorFunction can be in one of
// three states:
//    1. Every required (aka non-optional) tensor input is not empty:
//       isEmpty(tensorInput.tensor) == false.
//       Then the error function is valid and created for the optimization.
//    2. Every tensor input (required and optional) is empty:
//       isEmpty(tensorInput.tensor) == true.
//       This means the user does not want to use this error function.
//       Then a dummy error function will be created for the optimization and
//       all the derivatives will be marked as empty.
//    3. All the rest cases:
//       This is an error condition and assumes the user just forgot one.
// Whether a tensor input is optional is specified in tensorInput.optional.
// Note that error functions with non-tensor "placeholder" inputs need to
// decide for themselves whether it is appropriate to create an error function
// or not.
//
// Note that it is valid to have no tensor inputs.  One example is
// TensorLimitErrorFunction that adds joint limits. In this case, the error
// function will always be created.
//
// Note that an error function should have either no tensor inputs, or at least
// one required tensor input. It should _not_ have only optional tensor inputs.
// In the case of all tensor inputs are optional, if all the tensors are empty,
// we have no way of knowing whether the user does not want to build this
// function or wants to build one with all default tensor inputs.
//
// If one of the required tensor inputs is empty, it falls into State 3 and
// is considered error. So if you have an input which is allowed to be _empty_
// without affecting the operation of the error function, mark it optional to
// avoid entering State 3 to trigger error.
template <typename T>
class TensorErrorFunction {
 public:
  // name: error function name
  // batchSize: batch size
  // nFrames: Number of frames for multi-frame/sequence problems.  Should be 0
  //     for single-frame problems.
  // tensorInputs: initializer list of TensorInputs, each describing the format
  //     and storing the value of the tensor
  // sharedSizeNames: map negative integers to dimensions whose names shared by
  //     input tensors. Those dimensions don't have the size values during
  //     when constructing the TensorErrorFunction.
  //     See more details in class TensorInput.
  // This constructor checks tensor dimensions and fixes them if possible.
  TensorErrorFunction(
      const char* name,
      const char* argument_prefix,
      size_t batchSize,
      size_t nFrames,
      std::initializer_list<TensorInput> tensorInputs,
      std::initializer_list<std::pair<int, const char*>> sharedSizeNames);
  virtual ~TensorErrorFunction() = default;

  // Return a dummy function if all required tensors are empty, otherwsie
  // call virtual function createErrorFunctionImp() to return the real
  // error function.
  // Pass in SIZE_MAX for jFrame if not a sequence/multi-frame problem.
  std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
  createErrorFunction(const momentum::Character& character, size_t iBatch, size_t jFrame) const;

  const std::vector<TensorInput>& tensorInputs() const {
    return _tensorInputs;
  }

  const TensorInput& getTensorInput(const char* name) const;

  const char* name() const {
    return _name;
  }

  const char* argumentPrefix() const {
    return _argumentPrefix;
  }

  std::string enumName() const {
    return std::string("ErrorFunctionType.") + _name;
  }

  // Whether one of the required tensor input is empty.
  // If false, this means we are in State 1 (see above): a valid function should
  // be created.
  // Otherwise, assuming we have created this TensorErrorFunction, it couldn't
  // be in State 3, so we are now in State 2: a dummy function should be
  // created.
  bool requiredTensorEmpty() const;

  size_t batchSize() const {
    return _batchSize;
  }

 protected:
  virtual std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>

  // iBatch will be the batch index (0 if unbatched, although all problems have
  // an effective batch dimension appended implicitly).
  // jFrame is the frame index, for multi-frame problems only; for single-frame
  // solves we will pass SIZE_MAX for the frame index (this gets passed along
  // to the TensorInput and allows the caller to not have to disambiguate the
  // 'first frame of a multi-frame problem' from 'single-frame problem').
  createErrorFunctionImp(const momentum::Character& character, size_t iBatch, size_t jFrame)
      const = 0;

  // sharedDimensionFlag: the negative integer representing runtime dimension
  // size as described in TensorInput. Returns the actual size of that
  // dimension.
  size_t sharedSize(int sharedDimensionFlag) const;

 private:
  // Check if there are any tensors that are non empty because
  // non empty tensors need validation (checks for dimensions, etc.).
  bool anyNonEmptyTensor() const;

  // Make sure the dimensions match the targets.
  void validateDimensions();

  // Fix up all the tensors to make sure they're properly contiguous, located on
  // the CPU, etc.
  void fixTensors();

  // Return a string representing tensor size for better-formatted error
  // messages. sizes: input container representing tensor sizes
  // includeBatchPrefix: if true, this means the tensor may have a batch
  // dimension but the input sizes does not contain this dim, so in the output
  // string a "nBatch (opt.)" is added to represent this batch dim.
  //
  // Note: sizes can contain negative integers. In this case, this integer
  // serves as a placeholder for the dimension with unspecified size shared by
  // input tensors. If this integer is a key in _sharedSizeNames, the
  // corresponding value in _sharedSizeNames will be printed as the name of this
  // dimension. Otherwise, print "Y<absolute value of the integer>".
  template <typename TVec>
  std::string formatTensorSize(const TVec& sizes, bool includeBatchPrefix = false) const;

  std::string sharedSizeName(int iShared) const;

  std::string fullArgumentName(const TensorInput& input) const;

  const char* _name;
  const char* _argumentPrefix;
  const size_t _batchSize;
  const size_t _nFrames;
  std::vector<TensorInput> _tensorInputs;
  std::vector<std::pair<int, const char*>> _sharedSizeNames;
  std::map<int, size_t> _sharedSizes;
};

} // namespace pymomentum
