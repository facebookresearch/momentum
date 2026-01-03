/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pymomentum/array_utility/array_utility.h>

#include <momentum/character/character.h>

#include <fmt/format.h>
#include <sstream>

namespace pymomentum {

// ============================================================================
// LeadingDimensions implementation
// ============================================================================

py::ssize_t LeadingDimensions::totalBatchElements() const {
  if (dims.empty()) {
    return 1; // Scalar case - no leading dimensions means 1 batch element
  }
  py::ssize_t total = 1;
  for (const auto& d : dims) {
    total *= d;
  }
  return total;
}

bool LeadingDimensions::isScalar() const {
  for (const auto& d : dims) {
    if (d != 1) {
      return false;
    }
  }
  return true;
}

bool LeadingDimensions::broadcastCompatibleWith(const LeadingDimensions& other) const {
  // Must have same number of leading dimensions
  if (dims.size() != other.dims.size()) {
    return false;
  }

  // Each dimension must be equal or one of them must be 1
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] != other.dims[i] && dims[i] != 1 && other.dims[i] != 1) {
      return false;
    }
  }
  return true;
}

LeadingDimensions LeadingDimensions::broadcastWith(const LeadingDimensions& other) const {
  MT_THROW_IF(
      dims.size() != other.dims.size(),
      "Cannot broadcast leading dimensions of different sizes: {} vs {}",
      dims.size(),
      other.dims.size());

  LeadingDimensions result;
  result.dims.resize(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] == other.dims[i]) {
      result.dims[i] = dims[i];
    } else if (dims[i] == 1) {
      result.dims[i] = other.dims[i];
    } else if (other.dims[i] == 1) {
      result.dims[i] = dims[i];
    } else {
      MT_THROW("Cannot broadcast dimension {}: {} vs {}", i, dims[i], other.dims[i]);
    }
  }
  return result;
}

// ============================================================================
// ArrayChecker implementation
// ============================================================================

ArrayChecker::ArrayChecker(const char* functionName, ArrayDtype expectedDtype)
    : functionName_(functionName), requestedDtype_(expectedDtype) {}

void ArrayChecker::detectAndValidateDtype(const py::array& array, const char* arrayName) {
  bool isFloat32 = array.dtype().is(py::dtype::of<float>());
  bool isFloat64 = array.dtype().is(py::dtype::of<double>());

  if (!isFloat32 && !isFloat64) {
    MT_THROW(
        "In {}, array argument {} has dtype {} but expected float32 or float64",
        functionName_,
        arrayName,
        py::str(array.dtype()).cast<std::string>());
  }

  ArrayDtype arrayDtype = isFloat64 ? ArrayDtype::Float64 : ArrayDtype::Float32;

  if (detectedDtype_ == ArrayDtype::Auto) {
    // First array - detect dtype
    if (requestedDtype_ != ArrayDtype::Auto) {
      // User requested specific dtype - validate
      MT_THROW_IF(
          requestedDtype_ != arrayDtype,
          "In {}, array argument {} has dtype {} but {} was requested",
          functionName_,
          arrayName,
          isFloat64 ? "float64" : "float32",
          requestedDtype_ == ArrayDtype::Float64 ? "float64" : "float32");
    }
    detectedDtype_ = arrayDtype;
  } else {
    // Subsequent arrays - must match first
    MT_THROW_IF(
        detectedDtype_ != arrayDtype,
        "In {}, dtype mismatch: first array was {}, but {} is {}",
        functionName_,
        detectedDtype_ == ArrayDtype::Float64 ? "float64" : "float32",
        arrayName,
        isFloat64 ? "float64" : "float32");
  }
}

void ArrayChecker::validateAndUpdateLeadingDims(
    const py::array& array,
    const char* arrayName,
    size_t numTrailingDims) {
  const auto totalNdim = static_cast<size_t>(array.ndim());
  const size_t leadingNdim = totalNdim - numTrailingDims;

  // Extract leading dimensions from this array
  LeadingDimensions arrayLeadingDims;
  arrayLeadingDims.dims.resize(leadingNdim);
  for (size_t i = 0; i < leadingNdim; ++i) {
    arrayLeadingDims.dims[i] = array.shape(i);
  }

  if (!leadingDimsSet_) {
    // First array - set the leading dimensions
    leadingDims_ = arrayLeadingDims;
    leadingDimsSet_ = true;
  } else {
    // Subsequent arrays - validate and broadcast
    MT_THROW_IF(
        leadingDims_.ndim() != arrayLeadingDims.ndim(),
        "In {}, array argument {} has {} leading dimensions but expected {} "
        "(to match previous arrays). Array has shape {}.",
        functionName_,
        arrayName,
        arrayLeadingDims.ndim(),
        leadingDims_.ndim(),
        formatArrayDims(array));

    MT_THROW_IF(
        !leadingDims_.broadcastCompatibleWith(arrayLeadingDims),
        "In {}, array argument {} has incompatible leading dimensions. "
        "Expected dimensions broadcastable with {} but got {}. Array has shape {}.",
        functionName_,
        arrayName,
        formatArrayDims(leadingDims_.dims),
        formatArrayDims(arrayLeadingDims.dims),
        formatArrayDims(array));

    // Update leading dims with broadcasted result
    leadingDims_ = leadingDims_.broadcastWith(arrayLeadingDims);
  }
}

int64_t ArrayChecker::getBoundValue(int idx) const {
  auto itr = boundVariableSizes_.find(idx);
  MT_THROW_IF(
      itr == boundVariableSizes_.end(),
      "ArrayChecker: Called getBoundValue with unbound variable index {}.",
      idx);
  return itr->second;
}

void ArrayChecker::validateArray(
    const py::array& array,
    const char* arrayName,
    const std::vector<int>& trailingDims,
    const std::vector<const char*>& dimensionNames,
    bool allowEmpty) {
  MT_THROW_IF(
      trailingDims.size() != dimensionNames.size(),
      "In {}, internal error: trailingDims.size() != dimensionNames.size()",
      functionName_);

  const size_t numTrailingDims = trailingDims.size();

  // Handle empty arrays
  if (array.size() == 0) {
    MT_THROW_IF(
        !allowEmpty,
        "In {}, array argument {} is empty. Expected {} trailing dimensions.",
        functionName_,
        arrayName,
        formatExpectedDims(trailingDims, dimensionNames, boundVariableSizes_));
    return;
  }

  // Validate dtype
  detectAndValidateDtype(array, arrayName);

  // Validate array has at least numTrailingDims dimensions
  MT_THROW_IF(
      static_cast<size_t>(array.ndim()) < numTrailingDims,
      "In {}, array argument {} has shape {} but expected at least {} trailing dimensions: {}",
      functionName_,
      arrayName,
      formatArrayDims(array),
      numTrailingDims,
      formatExpectedDims(trailingDims, dimensionNames, boundVariableSizes_));

  // Validate and update leading dimensions
  validateAndUpdateLeadingDims(array, arrayName, numTrailingDims);

  // Validate trailing dimensions
  const size_t leadingNdim = static_cast<size_t>(array.ndim()) - numTrailingDims;
  for (size_t i = 0; i < numTrailingDims; ++i) {
    const auto foundSize = array.shape(leadingNdim + i);
    const int expectedSize = trailingDims[i];

    if (expectedSize < 0) {
      // Variable dimension - bind or validate
      auto itr = boundVariableSizes_.find(expectedSize);
      if (itr == boundVariableSizes_.end()) {
        boundVariableSizes_.emplace(expectedSize, foundSize);
      } else {
        MT_THROW_IF(
            foundSize != itr->second,
            "In {}, for array argument {}, mismatch in dimension {}; expected {} but found {}. "
            "Array has shape {}.",
            functionName_,
            arrayName,
            dimensionNames[i],
            itr->second,
            foundSize,
            formatArrayDims(array));
      }
    } else {
      // Fixed dimension - validate
      MT_THROW_IF(
          foundSize != expectedSize,
          "In {}, for array argument {}, mismatch in dimension {}; expected {} but found {}. "
          "Array has shape {}.",
          functionName_,
          arrayName,
          dimensionNames[i],
          expectedSize,
          foundSize,
          formatArrayDims(array));
    }
  }
}

void ArrayChecker::validateSkeletonState(
    const py::array& array,
    const char* arrayName,
    const momentum::Character& character) {
  const auto nJoints = static_cast<int>(character.skeleton.joints.size());
  validateArray(array, arrayName, {nJoints, 8}, {"numJoints", "8"});
}

JointParamsShape ArrayChecker::validateJointParameters(
    const py::array& array,
    const char* arrayName,
    const momentum::Character& character) {
  const auto nJoints = static_cast<int>(character.skeleton.joints.size());
  const auto nJointParams = nJoints * 7; // kParametersPerJoint

  // Joint parameters can be passed in two formats:
  // 1. Structured: (..., nJoints, 7)
  // 2. Flat: (..., nJointParams) where nJointParams = nJoints * 7
  //
  // We determine which format by checking the last dimension

  if (array.ndim() == 0) {
    MT_THROW(
        "In {}, array argument {} is a scalar but expected shape "
        "(..., {}, 7) or (..., {})",
        functionName_,
        arrayName,
        nJoints,
        nJointParams);
  }

  const auto lastDim = array.shape(array.ndim() - 1);

  if (lastDim == 7 && array.ndim() >= 2) {
    // Structured format: (..., nJoints, 7)
    const auto secondLastDim = array.shape(array.ndim() - 2);
    if (secondLastDim == nJoints) {
      validateArray(array, arrayName, {nJoints, 7}, {"numJoints", "7"});
      return JointParamsShape::Structured;
    }
  }

  // Either flat format or invalid - validate as flat and let it throw if wrong
  validateArray(array, arrayName, {nJointParams}, {"numJointParams"});
  return JointParamsShape::Flat;
}

void ArrayChecker::validateModelParameters(
    const py::array& array,
    const char* arrayName,
    const momentum::Character& character) {
  const auto nModelParams = static_cast<int>(character.parameterTransform.numAllModelParameters());
  validateArray(array, arrayName, {nModelParams}, {"numModelParams"});
}

// ============================================================================
// Formatting functions
// ============================================================================

std::string formatArrayDims(const py::array& array) {
  std::ostringstream oss;
  oss << "[";
  for (py::ssize_t i = 0; i < array.ndim(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << array.shape(i);
  }
  oss << "]";
  return oss.str();
}

std::string formatArrayDims(const std::vector<py::ssize_t>& dims) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << dims[i];
  }
  oss << "]";
  return oss.str();
}

std::string formatExpectedDims(
    const std::vector<int>& trailingDims,
    const std::vector<const char*>& dimensionNames,
    const std::unordered_map<int, int64_t>& boundVariableSizes) {
  std::ostringstream oss;
  oss << "[..., ";
  for (size_t i = 0; i < trailingDims.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    if (dimensionNames[i] != nullptr) {
      oss << dimensionNames[i];
      if (trailingDims[i] > 0) {
        oss << "(=" << trailingDims[i] << ")";
      } else {
        auto itr = boundVariableSizes.find(trailingDims[i]);
        if (itr != boundVariableSizes.end()) {
          oss << "(=" << itr->second << ")";
        }
      }
    } else {
      oss << trailingDims[i];
    }
  }
  oss << "]";
  return oss.str();
}

} // namespace pymomentum
