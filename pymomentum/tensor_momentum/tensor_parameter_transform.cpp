/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"

#include "pymomentum/python_utility/python_utility.h"
#include "pymomentum/tensor_utility/autograd_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/character/joint.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/common/checks.h>

#include <dispenso/parallel_for.h> // @manual
#ifndef PYMOMENTUM_LIMITED_TORCH_API
#include <torch/csrc/jit/python/python_ivalue.h>
#endif
#include <Eigen/Core>

namespace pymomentum {

namespace {

#ifndef PYMOMENTUM_LIMITED_TORCH_API

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

template <typename T>
struct ApplyParameterTransformFunction
    : public torch::autograd::Function<ApplyParameterTransformFunction<T>> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      PyObject* characters,
      const momentum::ParameterTransform* paramTransform,
      at::Tensor modelParams);

  static variable_list backward(AutogradContext* ctx, variable_list grad_jointParameters);

  static std::vector<momentum::ParameterTransformT<T>> getParameterTransforms(
      const momentum::ParameterTransform* paramTransform,
      PyObject* characters,
      int nBatch);
};

template <typename T>
std::vector<momentum::ParameterTransformT<T>>
ApplyParameterTransformFunction<T>::getParameterTransforms(
    const momentum::ParameterTransform* paramTransform,
    PyObject* characters,
    int nBatch) {
  std::vector<momentum::ParameterTransformT<T>> result;
  if (paramTransform) {
    result.push_back(paramTransform->cast<T>());
  } else {
    for (const auto c : toCharacterList(characters, nBatch, "ParameterTransform.apply", false)) {
      result.push_back(c->parameterTransform.cast<T>());
    }
  }
  return result;
}

template <typename T>
variable_list ApplyParameterTransformFunction<T>::forward(
    AutogradContext* ctx,
    PyObject* characters,
    const momentum::ParameterTransform* paramTransform,
    at::Tensor modelParams) {
  const int nModelParam = (paramTransform != nullptr)
      ? (int)paramTransform->numAllModelParameters()
      : anyCharacter(characters, "ParameterTransform.apply")
            .parameterTransform.numAllModelParameters();

  TensorChecker checker("ParameterTransform.apply");
  bool squeeze;
  const auto input_device = modelParams.device();

  modelParams = checker.validateAndFixTensor(
      modelParams,
      "modelParameters",
      {nModelParam},
      {"nModelParams"},
      toScalarType<T>(),
      true,
      false,
      &squeeze);

  if (paramTransform) {
    ctx->saved_data["parameterTransform"] =
        c10::ivalue::ConcretePyObjectHolder::create(py::cast(paramTransform));
  } else {
    ctx->saved_data["character"] = c10::ivalue::ConcretePyObjectHolder::create(characters);
  }

  const auto nBatch = checker.getBatchSize();
  MT_CHECK(paramTransform != nullptr || characters != nullptr);
  const auto paramTransforms = getParameterTransforms(paramTransform, characters, nBatch);

  assert(!paramTransforms.empty());
  const auto nJointParam = paramTransforms.front().numJointParameters();
  auto result = at::zeros({nBatch, nJointParam}, toScalarType<T>());
  dispenso::parallel_for(0, nBatch, [&](int64_t iBatch) {
    toEigenMap<T>(result.select(0, iBatch)) =
        paramTransforms[iBatch % paramTransforms.size()]
            .apply(toEigenMap<T>(modelParams.select(0, iBatch)))
            .v;
  });

  if (squeeze) {
    result = result.squeeze(0);
  }

  return {result.to(input_device)};
}

template <typename T>
variable_list ApplyParameterTransformFunction<T>::backward(
    AutogradContext* ctx,
    variable_list grad_outputs) {
  MT_THROW_IF(
      grad_outputs.size() != 1,
      "Invalid grad_outputs in ApplyParameterTransformFunction::backward");

  // Restore variables:
  PyObject* characters = nullptr;
  const momentum::ParameterTransform* paramTransform = nullptr;

  {
    auto itr = ctx->saved_data.find("parameterTransform");
    if (itr == ctx->saved_data.end()) {
      itr = ctx->saved_data.find("character");
      MT_THROW_IF(itr == ctx->saved_data.end(), "Missing both paramTransform and characters.");
      characters = itr->second.toPyObject();
    } else {
      paramTransform = py::cast<const momentum::ParameterTransform*>(itr->second.toPyObject());
    }
  }

  MT_CHECK(paramTransform != nullptr || characters != nullptr);
  const int nJointParams = (paramTransform != nullptr)
      ? (int)paramTransform->numJointParameters()
      : anyCharacter(characters, "ParameterTransform.apply")
            .parameterTransform.numJointParameters();

  TensorChecker checker("ParameterTransform.apply");
  bool squeeze;
  const auto input_device = grad_outputs[0].device();

  auto dLoss_dJointParameters = checker.validateAndFixTensor(
      grad_outputs[0],
      "dLoss_dJointParameters",
      {nJointParams},
      {"nJointParameters"},
      toScalarType<T>(),
      true,
      false,
      &squeeze);

  const auto nBatch = checker.getBatchSize();

  MT_CHECK(paramTransform != nullptr || characters != nullptr);
  const auto paramTransforms = getParameterTransforms(paramTransform, characters, nBatch);
  const int nModelParams = (int)paramTransforms.front().numAllModelParameters();

  at::Tensor result = at::zeros({(int)nBatch, (int)nModelParams}, toScalarType<T>());
  dispenso::parallel_for(0, nBatch, [&](int64_t iBatch) {
    toEigenMap<T>(result.select(0, iBatch)) =
        paramTransforms[iBatch % paramTransforms.size()].transform.transpose() *
        toEigenMap<T>(dLoss_dJointParameters.select(0, iBatch));
  });

  if (squeeze) {
    result = result.sum(0);
  }

  return {at::Tensor(), at::Tensor(), result.to(input_device)};
}

#endif // PYMOMENTUM_LIMITED_TORCH_API

} // anonymous namespace

at::Tensor applyParamTransform(
    [[maybe_unused]] const momentum::ParameterTransform* paramTransform,
    [[maybe_unused]] at::Tensor modelParams) {
#ifndef PYMOMENTUM_LIMITED_TORCH_API
  MT_CHECK_NOTNULL(paramTransform);
  PyObject* characters = nullptr;
  return applyTemplatedAutogradFunction<ApplyParameterTransformFunction>(
      characters, paramTransform, modelParams)[0];
#else
  MT_THROW("applyParamTransform is not supported in limited PyTorch API mode");
#endif
}

at::Tensor applyParamTransform(
    [[maybe_unused]] pybind11::object characters,
    [[maybe_unused]] at::Tensor modelParams) {
#ifndef PYMOMENTUM_LIMITED_TORCH_API
  MT_CHECK_NOTNULL(characters.ptr());
  const momentum::ParameterTransform* paramTransform = nullptr;
  return applyTemplatedAutogradFunction<ApplyParameterTransformFunction>(
      characters.ptr(), paramTransform, modelParams)[0];
#else
  MT_THROW("applyParamTransform is not supported in limited PyTorch API mode");
#endif
}

at::Tensor parameterSetToTensor(
    const momentum::ParameterTransform& parameterTransform,
    const momentum::ParameterSet& paramSet) {
  const auto nParam = parameterTransform.numAllModelParameters();

  at::Tensor result = at::zeros({(int)nParam}, at::kBool);
  auto ptr = (uint8_t*)result.data_ptr();
  for (int k = 0; k < nParam; ++k) {
    if (paramSet.test(k)) {
      ptr[k] = 1;
    }
  }

  return result;
}

momentum::ParameterSet tensorToParameterSet(
    const momentum::ParameterTransform& parameterTransform,
    at::Tensor paramSet,
    DefaultParameterSet defaultParamSet) {
  if (isEmpty(paramSet)) {
    switch (defaultParamSet) {
      case DefaultParameterSet::ALL_ZEROS: {
        momentum::ParameterSet result;
        return result;
      }
      case DefaultParameterSet::ALL_ONES: {
        momentum::ParameterSet result;
        result.set();
        return result;
      }
      case DefaultParameterSet::NO_DEFAULT:
      default:
          // fall through to the check below:
          ;
    }
  }

  const auto nParam = parameterTransform.numAllModelParameters();

  MT_THROW_IF(
      isEmpty(paramSet) || paramSet.ndimension() != 1 || paramSet.size(0) != nParam,
      "Mismatch between active parameters size and parameter transform size.");

  paramSet = paramSet.to(at::DeviceType::CPU, at::ScalarType::Bool);
  auto ptr = (uint8_t*)paramSet.data_ptr();
  momentum::ParameterSet result;
  for (int k = 0; k < nParam; ++k) {
    if (ptr[k] != 0) {
      result.set(k);
    }
  }

  return result;
}

at::Tensor getScalingParameters(const momentum::ParameterTransform& parameterTransform) {
  return parameterSetToTensor(parameterTransform, parameterTransform.getScalingParameters());
}

at::Tensor getRigidParameters(const momentum::ParameterTransform& parameterTransform) {
  return parameterSetToTensor(parameterTransform, parameterTransform.getRigidParameters());
}

at::Tensor getAllParameters(const momentum::ParameterTransform& parameterTransform) {
  momentum::ParameterSet params;
  params.set();
  return parameterSetToTensor(parameterTransform, params);
}

at::Tensor getBlendShapeParameters(const momentum::ParameterTransform& parameterTransform) {
  return parameterSetToTensor(parameterTransform, parameterTransform.getBlendShapeParameters());
}

at::Tensor getFaceExpressionParameters(const momentum::ParameterTransform& parameterTransform) {
  return parameterSetToTensor(parameterTransform, parameterTransform.getFaceExpressionParameters());
}

at::Tensor getPoseParameters(const momentum::ParameterTransform& parameterTransform) {
  return parameterSetToTensor(parameterTransform, parameterTransform.getPoseParameters());
}

std::unordered_map<std::string, at::Tensor> getParameterSets(
    const momentum::ParameterTransform& parameterTransform) {
  std::unordered_map<std::string, at::Tensor> result;
  for (const auto& ps : parameterTransform.parameterSets) {
    result.insert({ps.first, parameterSetToTensor(parameterTransform, ps.second)});
  }
  return result;
}

void addParameterSet(
    momentum::ParameterTransform& parameterTransform,
    const std::string& paramSetName,
    const at::Tensor& paramSet) {
  const auto set = tensorToParameterSet(parameterTransform, paramSet);
  parameterTransform.parameterSets.insert({paramSetName, set});
}

at::Tensor getParametersForJoints(
    const momentum::ParameterTransform& parameterTransform,
    const std::vector<size_t>& jointIndices) {
  const auto nJoints = parameterTransform.numJointParameters() / momentum::kParametersPerJoint;
  std::vector<bool> activeJoints(nJoints);
  for (const auto& idx : jointIndices) {
    MT_THROW_IF(idx >= nJoints, "getParametersForJoints: joint index out of bounds.");
    activeJoints[idx] = true;
  }

  momentum::ParameterSet result;

  // iterate over all non-zero entries of the matrix
  for (int k = 0; k < parameterTransform.transform.outerSize(); ++k) {
    for (momentum::SparseRowMatrixf::InnerIterator it(parameterTransform.transform, k); it; ++it) {
      const auto globalParam = it.row();
      const auto kParam = it.col();
      assert(kParam < parameterTransform.numAllModelParameters());
      const auto jointIndex = globalParam / momentum::kParametersPerJoint;
      assert(jointIndex < nJoints);

      if (activeJoints[jointIndex]) {
        result.set(kParam, true);
      }
    }
  }

  return parameterSetToTensor(parameterTransform, result);
}

at::Tensor findParameters(
    const momentum::ParameterTransform& parameterTransform,
    const std::vector<std::string>& parameterNames,
    bool allowMissing) {
  momentum::ParameterSet result;

  for (const auto& name : parameterNames) {
    auto idx = parameterTransform.getParameterIdByName(name);
    if (idx == momentum::kInvalidIndex) {
      if (allowMissing) {
        continue;
      } else {
        MT_THROW("Missing parameter: {}", name);
      }
    }

    result.set(idx);
  }

  return parameterSetToTensor(parameterTransform, result);
}

namespace {

#ifndef PYMOMENTUM_LIMITED_TORCH_API

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

struct ApplyInverseParameterTransformFunction
    : public torch::autograd::Function<ApplyInverseParameterTransformFunction> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      const momentum::InverseParameterTransform* inverseParamTransform,
      at::Tensor jointParams);

  static variable_list backward(AutogradContext* ctx, variable_list grad_modelParameters);
};

variable_list ApplyInverseParameterTransformFunction::forward(
    AutogradContext* ctx,
    const momentum::InverseParameterTransform* inverseParamTransform,
    at::Tensor jointParams) {
  const auto nJointParam = (int)inverseParamTransform->numJointParameters();

  TensorChecker checker("ParameterTransform.apply");
  const auto input_device = jointParams.device();

  bool squeeze;
  jointParams = checker.validateAndFixTensor(
      jointParams,
      "jointParameters",
      {nJointParam},
      {"nJointParameters"},
      at::kFloat,
      true,
      false,
      &squeeze);

  const auto nBatch = checker.getBatchSize();

  ctx->saved_data["inverseParameterTransform"] =
      c10::ivalue::ConcretePyObjectHolder::create(py::cast(inverseParamTransform));

  auto result = at::zeros({nBatch, inverseParamTransform->numAllModelParameters()}, at::kFloat);
  for (int64_t iBatch = 0; iBatch < nBatch; ++iBatch) {
    toEigenMap<float>(result.select(0, iBatch)) =
        inverseParamTransform->apply(toEigenMap<float>(jointParams.select(0, iBatch))).pose.v;
  }

  if (squeeze) {
    result = result.squeeze(0);
  }

  return {result.to(input_device)};
}

variable_list ApplyInverseParameterTransformFunction::backward(
    AutogradContext* ctx,
    variable_list grad_outputs) {
  MT_THROW_IF(
      grad_outputs.size() != 1,
      "Invalid grad_outputs in ApplyParameterTransformFunction::backward");

  // Restore variables:
  const auto inverseParamTransform = py::cast<const momentum::InverseParameterTransform*>(
      ctx->saved_data["inverseParameterTransform"].toPyObject());

  const auto input_device = grad_outputs[0].device(); // grad_outputs size is guarded already

  bool squeeze = false;
  auto dLoss_dModelParameters =
      grad_outputs[0].contiguous().to(at::DeviceType::CPU, at::ScalarType::Float);
  if (dLoss_dModelParameters.ndimension() == 1) {
    squeeze = true;
    dLoss_dModelParameters = dLoss_dModelParameters.unsqueeze(0);
  }

  MT_THROW_IF(
      dLoss_dModelParameters.size(1) != inverseParamTransform->numAllModelParameters(),
      "Unexpected error: mismatch in parameter transform sizes.");

  const auto nBatch = dLoss_dModelParameters.size(0);

  const int nModelParams = static_cast<int>(inverseParamTransform->numAllModelParameters());
  const int nJointParams = static_cast<int>(inverseParamTransform->numJointParameters());

  auto result = at::zeros({nBatch, nJointParams}, at::kFloat);

  // To solve for the model parameters, given the joint parameters,
  // inverseParameterTransform uses the QR decomposition,
  //    modelParams = R^{-1} * Q^T * jointParams
  // When taking the backwards-mode derivative, we need to apply the transpose
  // of this,
  //    dLoss_dJointParams = (R^{-1} * Q^T)^T * dLoss_dModelParams
  //                       = (R^{-1})^T * Q * dLoss_dModelParams
  // As the Q matrix is an implicitly (nJointParam x nJointParam) matrix, so
  // to apply it we need to pad the dLoss_dModelParams vector with zeros;
  // we'll use the tmp vector to store it.
  Eigen::VectorXf tmp = Eigen::VectorXf::Zero(nJointParams);
  const auto& qrDecomposition = inverseParamTransform->inverseTransform;
  for (int64_t iBatch = 0; iBatch < nBatch; ++iBatch) {
    tmp.head(nModelParams) =
        qrDecomposition.matrixR().triangularView<Eigen::Upper>().transpose().solve(
            toEigenMap<float>(dLoss_dModelParameters.select(0, iBatch)));
    toEigenMap<float>(result.select(0, iBatch)) = (qrDecomposition.matrixQ() * tmp).eval();
  }

  if (squeeze) {
    result = result.sum(0);
  }

  return {at::Tensor(), result.to(input_device)};
}

#endif // PYMOMENTUM_LIMITED_TORCH_API

} // anonymous namespace

at::Tensor applyInverseParamTransform(
    [[maybe_unused]] const momentum::InverseParameterTransform* invParamTransform,
    [[maybe_unused]] at::Tensor jointParams) {
#ifndef PYMOMENTUM_LIMITED_TORCH_API
  return ApplyInverseParameterTransformFunction::apply(invParamTransform, jointParams)[0];
#else
  MT_THROW("applyInverseParamTransform is not supported in limited PyTorch API mode");
#endif
}

std::unique_ptr<momentum::InverseParameterTransform> createInverseParameterTransform(
    const momentum::ParameterTransform& transform) {
  return std::make_unique<momentum::InverseParameterTransform>(transform);
}

namespace {

void maybeSet(bool* var, bool value) {
  if (var != nullptr) {
    *var = value;
  }
}

} // namespace

at::Tensor unflattenJointParameters(
    const momentum::Character& character,
    at::Tensor tensor_in,
    bool* unflattened) {
  if (tensor_in.ndimension() >= 2 && tensor_in.size(-1) == momentum::kParametersPerJoint &&
      tensor_in.size(-2) == character.skeleton.joints.size()) {
    maybeSet(unflattened, false);
    return tensor_in;
  }

  MT_THROW_IF(
      tensor_in.ndimension() < 1 ||
          tensor_in.size(-1) != momentum::kParametersPerJoint * character.skeleton.joints.size(),
      "Expected [... x (nJoints*7)] joint parameters tensor (with nJoints={}); got {}",
      character.skeleton.joints.size(),
      formatTensorSizes(tensor_in));

  std::vector<int64_t> dimensions;
  for (int64_t i = 0; i < tensor_in.ndimension(); ++i) {
    dimensions.push_back(tensor_in.size(i));
  }
  assert(dimensions.size() >= 1); // Guaranteed by check above.
  dimensions.back() = character.skeleton.joints.size();
  dimensions.push_back(momentum::kParametersPerJoint);
  maybeSet(unflattened, true);
  return tensor_in.reshape(dimensions);
}

at::Tensor flattenJointParameters(
    const momentum::Character& character,
    at::Tensor tensor_in,
    bool* flattened) {
  if (tensor_in.ndimension() >= 1 &&
      tensor_in.size(-1) == (momentum::kParametersPerJoint * character.skeleton.joints.size())) {
    maybeSet(flattened, false);
    return tensor_in;
  }

  MT_THROW_IF(
      tensor_in.ndimension() < 2 ||
          (tensor_in.size(-1) != momentum::kParametersPerJoint ||
           tensor_in.size(-2) != character.skeleton.joints.size()),
      "Expected [... x nJoints x 7] joint parameters tensor (with nJoints={}); got {}",
      character.skeleton.joints.size(),
      formatTensorSizes(tensor_in));

  std::vector<int64_t> dimensions;
  for (int64_t i = 0; i < tensor_in.ndimension(); ++i) {
    dimensions.push_back(tensor_in.size(i));
  }
  assert(dimensions.size() >= 2); // Guaranteed by check above.
  dimensions.pop_back();
  dimensions.back() = momentum::kParametersPerJoint * character.skeleton.joints.size();
  maybeSet(flattened, true);
  return tensor_in.reshape(dimensions);
}

at::Tensor modelParametersToBlendShapeCoefficients(
    const momentum::Character& character,
    const at::Tensor& modelParameters) {
  return modelParameters.index_select(
      -1, to1DTensor(character.parameterTransform.blendShapeParameters));
}

at::Tensor modelParametersToFaceExpressionCoefficients(
    const momentum::Character& character,
    const at::Tensor& modelParameters) {
  return modelParameters.index_select(
      -1, to1DTensor(character.parameterTransform.faceExpressionParameters));
}

at::Tensor getParameterTransformTensor(const momentum::ParameterTransform& parameterTransform) {
  const auto& transformSparse = parameterTransform.transform;

  at::Tensor transformDense =
      at::zeros({transformSparse.rows(), transformSparse.cols()}, at::kFloat);

  auto transformAccessor = transformDense.accessor<float, 2>();

  for (int i = 0; i < transformSparse.outerSize(); ++i) {
    for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(transformSparse, i); it;
         ++it) {
      transformAccessor[static_cast<long>(it.row())][static_cast<long>(it.col())] = it.value();
    }
  }
  return transformDense;
}

namespace {

// Use tensor.index_select() and tensor.index_copy() to copy data from srcTensor
// to form a new tensor. The dimension the mapping is applied on is `dimension`.
// The mapping is created by matching `srcNames` to `tgtNames`. The target slice
// which has no corresponding source is filled with zero.
at::Tensor mapTensor(
    int64_t dimension,
    const std::vector<std::string>& srcNames,
    const std::vector<std::string>& tgtNames,
    const at::Tensor& srcTensor) {
  if (dimension < 0) {
    dimension += srcTensor.ndimension();
  }

  MT_THROW_IF(
      dimension < 0 || srcTensor.ndimension() <= dimension ||
          srcTensor.size(dimension) != srcNames.size(),
      "Unexpected error in mapTensor(): dimensions don't match.");

  std::unordered_map<std::string, size_t> tgtNameMap;
  for (size_t iTgt = 0; iTgt < tgtNames.size(); ++iTgt) {
    tgtNameMap.insert({tgtNames[iTgt], iTgt});
  }

  const size_t expectedSize = std::min(srcNames.size(), tgtNames.size());

  std::vector<int64_t> tgtIndices;
  tgtIndices.reserve(expectedSize);
  std::vector<int64_t> srcIndices;
  srcIndices.reserve(expectedSize);

  for (size_t iSrc = 0; iSrc < srcNames.size(); ++iSrc) {
    auto itr = tgtNameMap.find(srcNames[iSrc]);
    if (itr == tgtNameMap.end()) {
      continue;
    }

    tgtIndices.push_back(itr->second);
    srcIndices.push_back(iSrc);
  }

  at::Tensor srcIndicesTensor = to1DTensor(srcIndices).to(srcTensor.device());
  at::Tensor tgtIndicesTensor = to1DTensor(tgtIndices).to(srcTensor.device());

  std::vector<int64_t> tgtSizes;
  for (int64_t i = 0; i < srcTensor.ndimension(); ++i) {
    if (i == dimension) {
      tgtSizes.push_back(tgtNames.size());
    } else {
      tgtSizes.push_back(srcTensor.size(i));
    }
  }

  at::Tensor result = at::zeros(tgtSizes, srcTensor.scalar_type());
  return result.index_copy(
      dimension, tgtIndicesTensor, srcTensor.index_select(dimension, srcIndicesTensor));
}

template <typename T>
at::Tensor applyModelParameterLimitsTemplate(
    const momentum::Character& character,
    at::Tensor modelParams) {
  TensorChecker checker("applyBodyParameterLimits");
  bool squeeze = false;
  const int nModelParamsID = -1;
  modelParams = checker.validateAndFixTensor(
      modelParams,
      "model_params",
      {nModelParamsID},
      {"nModelParams"},
      toScalarType<T>(),
      true,
      false,
      &squeeze);

  const int64_t nModelParams = checker.getBoundValue(nModelParamsID);
  MT_THROW_IF(
      character.parameterTransform.numAllModelParameters() != nModelParams,
      "pymomentum::applyModelParameterLimits(): model param size in input tensor does not match character");

  // character.parameterLimits can be empty. In this case, there is no
  // limit for all the model params.
  if (!character.parameterLimits.empty()) {
    // Store model param indices that have limits.
    std::vector<int64_t> limitedIndices;
    // Store the limit values.
    std::vector<T> minLimits, maxLimits;
    for (const auto& l : character.parameterLimits) {
      if (l.type == momentum::LimitType::MinMax) {
        limitedIndices.push_back(l.data.minMax.parameterIndex);
        minLimits.push_back(l.data.minMax.limits.x());
        maxLimits.push_back(l.data.minMax.limits.y());
      }
    }

    // Build tensors for doing the differentiable computation:
    // - We need an index tensor to call tensor.index_select() to get the slices
    // from modelParams that have limits.
    // - We then need to call torch.minimum() using a 1D tensor of upper
    //   bounds to clamp those model parameters from above.
    // - We also need torch.maximum() with a 1D tensor of lower bounds to
    //   clamp those model parameters from below.
    // - Finally, call tensor.index_copy() to combine those clamped model
    //   parameters with the bounds-free model parameters to build the returned
    //   tensor. The indices tensor used in this step is the same as the one in
    //   tensor.index_select().

    at::Tensor limitedIndicesTensor = to1DTensor(limitedIndices).to(modelParams.device());
    at::Tensor minLimitsTensor =
        to1DTensor(minLimits).to(modelParams.device(), modelParams.dtype());
    at::Tensor maxLimitsTensor =
        to1DTensor(maxLimits).to(modelParams.device(), modelParams.dtype());
    modelParams = modelParams.index_copy(
        -1,
        limitedIndicesTensor,
        torch::clamp(
            modelParams.index_select(-1, limitedIndicesTensor), minLimitsTensor, maxLimitsTensor));
  }

  if (squeeze) {
    modelParams = modelParams.squeeze(0);
  }

  return modelParams;
}

} // namespace

at::Tensor mapModelParameters(
    const at::Tensor& motion_in,
    const momentum::Character& srcCharacter,
    const momentum::Character& tgtCharacter,
    bool verbose) {
  return mapModelParameters_names(
      motion_in, srcCharacter.parameterTransform.name, tgtCharacter, verbose);
}

at::Tensor mapModelParameters_names(
    const at::Tensor& motion_in,
    const std::vector<std::string>& parameterNames_in,
    const momentum::Character& character_remap,
    bool verbose) {
  MT_THROW_IF(
      motion_in.size(-1) != parameterNames_in.size(),
      "Mismatch between motion size and parameter name count.");

  std::vector<std::string> missingParams;
  for (const auto& iParamSource : parameterNames_in) {
    auto iParamRemap = character_remap.parameterTransform.getParameterIdByName(iParamSource);
    if (iParamRemap == momentum::kInvalidIndex) {
      missingParams.push_back(iParamSource);
    }
  }

  if (verbose && !missingParams.empty()) {
    // TODO better logging:
    pybind11::print(
        "WARNING: missing parameters found during map_model_parameters: ", missingParams);
  }

  // Map tensor at dimension -1.
  return mapTensor(-1, parameterNames_in, character_remap.parameterTransform.name, motion_in);
}

at::Tensor mapJointParameters(
    at::Tensor srcMotion,
    const momentum::Character& srcCharacter,
    const momentum::Character& tgtCharacter) {
  // Make tensor into shape: [... x n_joint_params x 7]
  bool unflattened = false;
  srcMotion = unflattenJointParameters(srcCharacter, srcMotion, &unflattened);

  // Map tensor at dimension -2.
  at::Tensor result = mapTensor(
      -2, srcCharacter.skeleton.getJointNames(), tgtCharacter.skeleton.getJointNames(), srcMotion);
  if (unflattened) {
    result = flattenJointParameters(tgtCharacter, result);
  }
  return result;
}

at::Tensor uniformRandomToModelParameters(
    const momentum::Character& character,
    at::Tensor unifNoise) {
  unifNoise = unifNoise.contiguous().to(at::DeviceType::CPU, at::ScalarType::Float);

  const auto& paramTransform = character.parameterTransform;

  const auto nModelParam = (int64_t)paramTransform.numAllModelParameters();
  assert(paramTransform.name.size() == nModelParam);

  bool squeeze = false;
  if (unifNoise.ndimension() == 1) {
    unifNoise = unifNoise.unsqueeze(0);
    squeeze = true;
  }
  const auto nBatch = unifNoise.size(0);

  MT_THROW_IF(
      unifNoise.size(1) != nModelParam,
      "In uniformRandomToModelParameters(), expected array with size [nBatch, nModelParameters] with nModelParameters={}; got array with size {}",
      nModelParam,
      formatTensorSizes(unifNoise));

  at::Tensor result = at::zeros({nBatch, nModelParam}, at::kFloat);

  for (pybind11::ssize_t iBatch = 0; iBatch < nBatch; ++iBatch) {
    auto res_i = toEigenMap<float>(result.select(0, iBatch));
    auto unif_i = toEigenMap<float>(unifNoise.select(0, iBatch));

    for (size_t iParam = 0; iParam < nModelParam; ++iParam) {
      // TODO apply limits
      const auto& name = paramTransform.name[iParam];
      if (name.find("scale_") != std::string::npos) {
        res_i[iParam] = 1.0f * (unif_i[iParam] - 0.5f);
      } else if (
          name.find("_tx") != std::string::npos || name.find("_ty") != std::string::npos ||
          name.find("_tz") != std::string::npos) {
        res_i[iParam] = 5.0f * (unif_i[iParam] - 0.5f);
      } else {
        // Assume it's a rotation parameter
        res_i[iParam] = (momentum::pi<float>() / 4.0f) * (unif_i[iParam] - 0.5f);
      }
    }
  }

  if (squeeze) {
    result = result.squeeze(0);
  }

  return result;
}

at::Tensor applyModelParameterLimits(
    const momentum::Character& character,
    const at::Tensor& modelParams) {
  if (hasFloat64(modelParams)) {
    return applyModelParameterLimitsTemplate<double>(character, modelParams);
  } else {
    return applyModelParameterLimitsTemplate<float>(character, modelParams);
  }
}

} // namespace pymomentum
