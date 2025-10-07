/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_ik/tensor_error_function.h"

#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/character/character.h>
#include <momentum/character_solver/skeleton_error_function.h>

namespace pymomentum {

template <typename T>
std::string TensorErrorFunction<T>::fullArgumentName(const TensorInput& input) const {
  std::ostringstream oss;
  oss << _argumentPrefix << "_" << input.inputName;
  return oss.str();
}

template <typename T>
TensorErrorFunction<T>::TensorErrorFunction(
    const char* name,
    const char* argument_prefix,
    size_t batchSize,
    size_t nFrames,
    std::initializer_list<TensorInput> tensorInputs,
    std::initializer_list<std::pair<int, const char*>> sharedSizeNames)
    : _name(name),
      _argumentPrefix(argument_prefix),
      _batchSize(batchSize),
      _nFrames(nFrames),
      _tensorInputs(tensorInputs),
      _sharedSizeNames(sharedSizeNames) {
  if (anyNonEmptyTensor()) {
    validateDimensions();
    fixTensors();
  }
}

template <typename T>
bool TensorErrorFunction<T>::anyNonEmptyTensor() const {
  for (const auto& i : _tensorInputs) {
    if (!isEmpty(i.tensor)) {
      return true;
    }
  }

  return false;
}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>> TensorErrorFunction<T>::createErrorFunction(
    const momentum::Character& character,
    size_t iBatch,
    size_t jFrame) const {
  MT_THROW_IF(jFrame == SIZE_MAX && _nFrames != 0, "Invalid frame index.");

  // createErrorFunctionImp assumes that all required tensors are present,
  // so return a dummy error function if not.
  if (requiredTensorEmpty()) {
    return std::make_unique<momentum::SkeletonErrorFunctionT<T>>(
        character.skeleton, character.parameterTransform);
  }

  return createErrorFunctionImp(character, iBatch, jFrame);
}

template <typename T>
bool TensorErrorFunction<T>::requiredTensorEmpty() const {
  for (const auto& i : _tensorInputs) {
    if (i.targetType != TensorType::TYPE_SENTINEL && i.optionality == TensorInput::REQUIRED &&
        isEmpty(i.tensor)) {
      return true;
    }
  }

  return false;
}

namespace {

class DimensionString {
 public:
  template <typename... T>
  void add(T... value) {
    if (!_first) {
      _oss << " x ";
    }
    _first = false;
    append(value...);
  }

  std::string value() {
    return "[" + _oss.str() + "]";
  }

 private:
  template <typename T>
  void append(const T& value) {
    _oss << value;
  }

  template <typename T1, typename... TRest>
  void append(const T1& v1, TRest... vRest) {
    append(v1);
    append(vRest...);
  }

  bool _first = true;
  std::ostringstream _oss;
};

} // namespace

template <typename T>
std::string TensorErrorFunction<T>::sharedSizeName(int iShared) const {
  auto itr = std::find_if(
      _sharedSizeNames.begin(), _sharedSizeNames.end(), [&](const std::pair<int, const char*>& pr) {
        return pr.first == iShared;
      });
  if (itr != _sharedSizeNames.end()) {
    return itr->second;
  }

  std::ostringstream oss;
  oss << "Y" << (-iShared);
  return oss.str();
}

template <typename T>
template <typename TVec>
std::string TensorErrorFunction<T>::formatTensorSize(const TVec& sizes, bool includeBatchPrefix)
    const {
  DimensionString result;

  if (includeBatchPrefix) {
    result.add("(nBatch)");
  }

  if (_nFrames != 0) {
    result.add("(nFrames)");
  }

  for (const auto& sz : sizes) {
    if (sz < 0) {
      result.add(sharedSizeName(sz));
    } else {
      result.add(sz);
    }
  }
  return result.value();
}

template <typename T>
void TensorErrorFunction<T>::validateDimensions() {
  // Ugly function that validates that all the dimensions are kosher; the
  // goal here is to centralize all error handling so the rest of the code
  // can assume that the tensors all have the correct dimension.

  // Negative integer as key of a dim -> <actual tensor dim size, the first
  // tensor with this dim>
  std::unordered_map<int, std::pair<size_t, const TensorInput*>> sharedSizes;

  for (const auto& i : _tensorInputs) {
    if (i.targetType == TensorType::TYPE_SENTINEL) {
      continue;
    }

    if (isEmpty(i.tensor)) {
      // If a required tensor is empty:
      MT_THROW_IF(
          i.optionality == TensorInput::REQUIRED,
          "Missing tensor for {} (required for {}). Expected tensor dimensions are: {}",
          fullArgumentName(i),
          enumName(),
          formatTensorSize(i.targetSizes, true));

      continue;
    }

    size_t dimOffset = 0;

    // Validate the batch and frame dimensions:
    if (_nFrames != 0) {
      // Multi-frame solve.  Expected tensor of
      //  [nBatch x nFrames x ...]
      if (i.tensor.ndimension() == i.targetSizes.size() + 2) {
        MT_THROW_IF(
            i.tensor.size(0) != _batchSize,
            "Mismatch in batch dimension for {} (used by {}); expected tensor dimensions are: {}. Expected batch size of {} but got dimensions {}",
            fullArgumentName(i),
            enumName(),
            formatTensorSize(i.targetSizes, true),
            _batchSize,
            formatTensorSize(i.tensor.sizes()));

        MT_THROW_IF(
            i.tensor.size(1) != _nFrames,
            "Mismatch in nFrames for {} (used by {}); expected tensor dimensions are: {}. Expected nFrames equal to {} but got dimensions {}",
            fullArgumentName(i),
            enumName(),
            formatTensorSize(i.targetSizes, true),
            _nFrames,
            formatTensorSize(i.tensor.sizes()));

        dimOffset = 2;
      } else if (i.tensor.ndimension() == i.targetSizes.size() + 1) {
        MT_THROW_IF(
            _batchSize != 1,
            "Ambiguity in batch/frame dimension for {} (used by {}); expected tensor dimensions are: {}. The tensor should either include _both_ a batch and a frame dimension or omit both (meaning it is shared across the batch and to avoid ambiguity.",
            fullArgumentName(i),
            enumName(),
            formatTensorSize(i.targetSizes, true));

        dimOffset = 1;
      }
    } else {
      // Single-frame solve.  Expected tensor of
      //    [nBatch x ...]
      if (i.tensor.ndimension() == i.targetSizes.size() + 1) {
        MT_THROW_IF(
            i.tensor.size(0) != _batchSize,
            "Mismatch in batch dimension for {} (used by {}); expected tensor dimensions are: {}. Expected batch size of {} but got dimensions {}",
            fullArgumentName(i),
            enumName(),
            formatTensorSize(i.targetSizes, true),
            _batchSize,
            formatTensorSize(i.tensor.sizes()));
        dimOffset = 1;
      }
    }

    // Validate the remaining dimensions.
    MT_THROW_IF(
        i.tensor.ndimension() != i.targetSizes.size() + dimOffset,
        "Mismatch in tensor dimensions for {} (used by {}); expected: {} but got {}",
        fullArgumentName(i),
        enumName(),
        formatTensorSize(i.targetSizes, true),
        formatTensorSize(i.tensor.sizes()));

    // Check validity on each dimension:
    for (size_t iDim = 0; iDim < i.targetSizes.size(); ++iDim) {
      auto tensorSz = i.tensor.size(iDim + dimOffset);
      // If the target size is negative, this means we don't know
      // the size during compilation, so we match this negative
      // integer to the input size.
      if (i.targetSizes[iDim] < 0) {
        const auto sharedDimIndex = i.targetSizes[iDim];

        // Match up to an existing shared size:
        auto itr = sharedSizes.find(sharedDimIndex);
        if (itr == sharedSizes.end()) {
          sharedSizes.insert(itr, std::make_pair(sharedDimIndex, std::make_pair(tensorSz, &i)));
        } else {
          // Found previous instance; validate that the sizes match:
          const auto storedSz = itr->second.first;
          const auto& other_i = *itr->second.second;
          MT_THROW_IF(
              storedSz != tensorSz,
              "Mismatch in tensor dimensions between {} and {} (used by {}). For {} expected {} and found {}. For {} expected {} and found {}. Mismatch is in the '{}' dimension.",
              fullArgumentName(other_i),
              fullArgumentName(i),
              enumName(),
              fullArgumentName(other_i),
              formatTensorSize(other_i.targetSizes, true),
              formatTensorSize(other_i.tensor.sizes()),
              fullArgumentName(i),
              formatTensorSize(i.targetSizes, true),
              formatTensorSize(i.tensor.sizes()),
              sharedSizeName(sharedDimIndex));
        }
      } else {
        // The size is fixed during compilation; we should match exactly:
        MT_THROW_IF(
            tensorSz != i.targetSizes[iDim],
            "Mismatch in tensor dimensions for {} (used by {}); expected: {} but received {}",
            fullArgumentName(i),
            enumName(),
            formatTensorSize(i.targetSizes, true),
            formatTensorSize(i.tensor.sizes()));
      }
    }
  }

  // Store the mapping {negative integer key -> actual tensor dim size} to
  // _sharedSizes so that the constructor of the derived class can easily know
  // the actual dims.
  for (const auto& pair : sharedSizes) {
    _sharedSizes.emplace(pair.first, pair.second.first);
  }
}

template <typename T>
void TensorErrorFunction<T>::fixTensors() {
  for (auto& i : _tensorInputs) {
    if (i.targetType == TensorType::TYPE_SENTINEL) {
      continue;
    }

    auto scalarType =
        (i.targetType == TensorType::TYPE_FLOAT) ? toScalarType<T>() : at::ScalarType::Int;

    // Tensors that require gradients need to be on the CPU or we will hit
    // errors in the tensor list that we return that are difficult to pull
    // apart.
    MT_THROW_IF(
        i.tensor.requires_grad() && !i.tensor.is_cpu(),
        "Tensor {} is not on the CPU.",
        fullArgumentName(i));

    // It needs to be contiguous (so we can access it using just the pointer)
    // and matching the correct scalar type.
    i.tensor = i.tensor.contiguous().to(at::DeviceType::CPU, scalarType);
  }
}

template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> TensorInput::toEigenMap(
    size_t iBatch,
    size_t iFrame) const {
  MT_THROW_IF(
      targetType == TensorType::TYPE_SENTINEL, "Attempt to use sentinel tensor as actual input.");

  if (isEmpty(tensor)) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(nullptr, 0);
  }

  auto tensor_cur = tensor;
  if (iFrame == SIZE_MAX) {
    // Single-frame problem.
    if (tensor.ndimension() != targetSizes.size()) {
      tensor_cur = tensor.select(0, iBatch);
    }
  } else {
    // multi-frame problem.
    if (tensor.ndimension() == targetSizes.size() + 2) {
      // Both batch and frame dimensions present:
      tensor_cur = tensor.select(0, iBatch).select(0, iFrame);
    } else if (tensor.ndimension() == targetSizes.size() + 1) {
      // If there's just one extra dimension, it's assumed to be the frame
      // dimension. The dimension checks should have flagged any cases where
      // it's ambiguous.
      MT_THROW_IF(iBatch != 0, "Invalid batch index!");
      tensor_cur = tensor.select(0, iFrame);
    }
  }

  return pymomentum::toEigenMap<T>(tensor_cur);
}

template Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> TensorInput::toEigenMap<double>(
    size_t iBatch,
    size_t iFrame) const;

template Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> TensorInput::toEigenMap<float>(
    size_t iBatch,
    size_t iFrame) const;

template Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 1>> TensorInput::toEigenMap<int>(
    size_t iBatch,
    size_t iFrame) const;

template <typename T>
const TensorInput& TensorErrorFunction<T>::getTensorInput(const char* name) const {
  auto itr =
      std::find_if(_tensorInputs.begin(), _tensorInputs.end(), [name](const TensorInput& input) {
        return strcmp(name, input.inputName) == 0;
      });
  MT_THROW_IF(itr == _tensorInputs.end(), "Missing input: {}", name);

  return *itr;
}

template <typename T>
size_t TensorErrorFunction<T>::sharedSize(int sharedDimensionFlag) const {
  const auto find = _sharedSizes.find(sharedDimensionFlag);
  MT_THROW_IF(
      find == _sharedSizes.end(), "Cannot find a shared dimension flag: {}", sharedDimensionFlag);
  return find->second;
}

template class TensorErrorFunction<float>;
template class TensorErrorFunction<double>;

} // namespace pymomentum
