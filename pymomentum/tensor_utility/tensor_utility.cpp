/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pymomentum/tensor_utility/tensor_utility.h>

#include <momentum/common/exception.h>

namespace py = pybind11;

namespace pymomentum {

template <typename Iter>
std::string formatTensorSizes(const Iter& begin, const Iter& end) {
  std::ostringstream oss;
  bool first = true;
  oss << "[";
  for (auto sz = begin; sz != end; ++sz) {
    if (!first) {
      oss << ", ";
    }
    oss << *sz;
    first = false;
  }
  oss << "]";
  return oss.str();
}

std::string formatTensorSizes(const at::Tensor& tensor) {
  auto sizes = tensor.sizes();
  return formatTensorSizes(sizes.begin(), sizes.end());
}

std::string formatTensorSizes(const std::vector<py::ssize_t>& sizes) {
  return formatTensorSizes(sizes.begin(), sizes.end());
}

at::Tensor denullify(std::optional<at::Tensor> tensor) {
  if (!tensor.has_value()) {
    return at::empty({0});
  }
  return *tensor;
}

template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> toEigenMap(at::Tensor t) {
  MT_THROW_IF(
      t.scalar_type() != toScalarType<T>(), "Mismatch in tensor types.");

  int64_t sz = 1;
  for (const auto& d : t.sizes()) {
    sz *= d;
  }

  return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>((T*)t.data_ptr(), sz);
}

template Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> toEigenMap(
    at::Tensor t);
template Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> toEigenMap(
    at::Tensor t);
template Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 1>> toEigenMap(
    at::Tensor t);

template <typename T, int Rows, int Cols>
std::vector<Eigen::Matrix<T, Rows, Cols>> toMatrixList(at::Tensor tensor) {
  MT_THROW_IF(
      tensor.ndimension() != 3, "toMatrixList expected tensor of dimension 3.");

  const auto nMat = tensor.size(0);
  const auto nRows = tensor.size(1);
  const auto nCols = tensor.size(2);

  MT_THROW_IF(
      Rows != Eigen::Dynamic && nRows != Rows,
      "Mismatch in row dimension for toMatrixList()");

  MT_THROW_IF(
      Cols != Eigen::Dynamic && nCols != Cols,
      "Mismatch in row dimension for toMatrixList()");

  std::vector<Eigen::Matrix<T, Rows, Cols>> result;
  result.reserve(nMat);

  for (int i = 0; i < nMat; ++i) {
    result.push_back(
        toEigenMap<T>(tensor.select(0, i)).reshaped(nRows, nCols).transpose());
  }

  return result;
}

template std::vector<Eigen::Matrix4f> toMatrixList<float, 4, 4>(
    at::Tensor tensor);
template std::vector<Eigen::Matrix4d> toMatrixList<double, 4, 4>(
    at::Tensor tensor);

std::string formatExpectedDimensions(
    const std::vector<int>& expectedSizes,
    const std::vector<const char*>& dimensionNames,
    const std::unordered_map<int, int64_t>& boundVariableSizes) {
  std::ostringstream oss;
  oss << "nBatch";
  for (size_t i = 0; i < expectedSizes.size(); ++i) {
    oss << " x ";
    if (dimensionNames[i] == nullptr) {
      MT_THROW_IF(
          expectedSizes[i] < 0,
          "Expected size must be non-negative when dimension name is null");
      oss << expectedSizes[i];
      continue;
    }
    oss << dimensionNames[i];
    if (expectedSizes[i] > 0) {
      oss << "[=" << expectedSizes[i] << "]";
    } else {
      auto itr = boundVariableSizes.find(expectedSizes[i]);
      if (itr != boundVariableSizes.end()) {
        oss << "[=" << itr->second << "]";
      }
    }
  }
  return oss.str();
}

at::Tensor TensorChecker::validateAndFixTensor(
    at::Tensor tensor_orig,
    const char* tensorName,
    const std::vector<int>& expectedSizes,
    const std::vector<const char*>& dimensionNames,
    at::ScalarType expectedScalarType,
    bool allowUnbatched,
    bool allowEmpty,
    bool* needsSqueeze_out) {
  at::Tensor result =
      tensor_orig.contiguous().to(at::DeviceType::CPU, expectedScalarType);

  MT_THROW_IF(
      expectedSizes.size() != dimensionNames.size(),
      "Unexpected error in validateAndFixTensor()");
  const auto nExpectedDim = expectedSizes.size();

  if (isEmpty(tensor_orig)) {
    MT_THROW_IF(
        !allowEmpty,
        "In {}, tensor argument {} is empty. Expected {}",
        _functionName,
        tensorName,
        formatExpectedDimensions(
            expectedSizes, dimensionNames, _boundVariableSizes));
    return result;
  }

  // Validate the batch dimension:
  bool needsSqueeze;
  int batchSize_new = _batchSize;
  if (tensor_orig.ndimension() == nExpectedDim) {
    MT_THROW_IF(
        !allowUnbatched,
        "In {}, expected {} to be a batched tensor with dimensions {} but got unbatched tensor with dimensions {}",
        _functionName,
        tensorName,
        formatExpectedDimensions(
            expectedSizes, dimensionNames, _boundVariableSizes),
        formatTensorSizes(tensor_orig));

    result = result.unsqueeze(0);
    needsSqueeze = true;

    if (_batchSize < 0) {
      // Bind the batch size:
      batchSize_new = 1;
    } else {
      // No batch size in input, so expand to match the expected
      // batch size.
      std::vector<int64_t> expandDim(result.ndimension(), -1);
      assert(!expandDim.empty());
      expandDim[0] = _batchSize;
      result = result.expand(expandDim);
    }
  } else if (tensor_orig.ndimension() == nExpectedDim + 1) {
    needsSqueeze = false;

    const auto batchSize_cur = tensor_orig.size(0);
    if (_batchSize < 0) {
      // Bind the batch size:
      batchSize_new = batchSize_cur;
    } else {
      MT_THROW_IF(
          batchSize_cur != _batchSize,
          "In {}, mismatch in tensor batch sizes for {}. Expected batch size of {} but got a tensor with dimensions {}",
          _functionName,
          tensorName,
          _batchSize,
          formatTensorSizes(tensor_orig));
    }
  } else {
    MT_THROW(
        "In {}, tensor argument {} has unexpected size {}. Expected {} but got a tensor with dimensions {}",
        _functionName,
        tensorName,
        formatTensorSizes(tensor_orig),
        formatExpectedDimensions(
            expectedSizes, dimensionNames, _boundVariableSizes),
        formatTensorSizes(tensor_orig));
  }

  std::unordered_map<int, int64_t> boundVariableSizes_new = _boundVariableSizes;

  // Validate the other dimensions:
  for (int iDim = 0; iDim < nExpectedDim; ++iDim) {
    // offset by 1 for the batch dimension:
    const auto foundSize = result.size(iDim + 1);

    if (expectedSizes[iDim] < 0) {
      const auto variableIndex = expectedSizes[iDim];
      auto itr = boundVariableSizes_new.find(variableIndex);
      if (itr == boundVariableSizes_new.end()) {
        // Currently unbound so go ahead and bind.
        boundVariableSizes_new.emplace(variableIndex, foundSize);
      } else {
        // Validate that we match the bound size.
        const auto boundSize = itr->second;
        MT_THROW_IF(
            foundSize != boundSize,
            "In {}, for tensor argument {} mismatch in tensor dimension {}; expected: {} but found {}",
            _functionName,
            tensorName,
            dimensionNames[iDim],
            formatExpectedDimensions(
                expectedSizes, dimensionNames, boundVariableSizes_new),
            formatTensorSizes(tensor_orig));
      }
    } else {
      MT_THROW_IF(
          foundSize != expectedSizes[iDim],
          "In {}, for tensor argument {}, mismatch in tensor dimension {}; expected: {} but found {}",
          _functionName,
          tensorName,
          dimensionNames[iDim],
          formatExpectedDimensions(
              expectedSizes, dimensionNames, boundVariableSizes_new),
          formatTensorSizes(tensor_orig));
    }
  }

  if (needsSqueeze_out != nullptr) {
    *needsSqueeze_out = needsSqueeze;
  }

  _boundVariableSizes = std::move(boundVariableSizes_new);
  _batchSize = batchSize_new;

  return result;
}

int64_t TensorChecker::getBatchSize() {
  MT_THROW_IF(
      _batchSize <= 0,
      "TensorChecker: Called getBatchSize with invalid batch size.");
  return _batchSize;
}

int64_t TensorChecker::getBoundValue(int idx) {
  auto itr = _boundVariableSizes.find(idx);
  MT_THROW_IF(
      itr == _boundVariableSizes.end(),
      "TensorChecker: Called getBoundValue with invalid variable index.");

  return itr->second;
}

void throwIfNaNOrINF(
    const at::Tensor& t,
    const char* context,
    const char* tensorName) {
  if (isEmpty(t)) {
    return;
  }

  MT_THROW_IF(
      at::isnan(t).any().cpu().item<bool>() ||
          at::isinf(t).any().cpu().item<bool>(),
      "In {}, {} with dimension {} has NaN/INF.",
      context,
      tensorName,
      formatTensorSizes(t));
}

} // namespace pymomentum
