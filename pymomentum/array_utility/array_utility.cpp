/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pymomentum/array_utility/array_utility.h>

#include <momentum/character/character.h>
#include <momentum/math/mesh.h>

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

void ArrayChecker::detectAndValidateDtype(const py::buffer_info& bufInfo, const char* bufferName) {
  bool isFloat32 = (bufInfo.format == py::format_descriptor<float>::format());
  bool isFloat64 = (bufInfo.format == py::format_descriptor<double>::format());

  if (!isFloat32 && !isFloat64) {
    MT_THROW(
        "In {}, buffer argument {} has dtype {} but expected float32 or float64",
        functionName_,
        bufferName,
        bufInfo.format);
  }

  ArrayDtype bufDtype = isFloat64 ? ArrayDtype::Float64 : ArrayDtype::Float32;

  if (detectedDtype_ == ArrayDtype::Auto) {
    // First buffer - detect dtype
    if (requestedDtype_ != ArrayDtype::Auto) {
      // User requested specific dtype - validate
      MT_THROW_IF(
          requestedDtype_ != bufDtype,
          "In {}, buffer argument {} has dtype {} but {} was requested",
          functionName_,
          bufferName,
          isFloat64 ? "float64" : "float32",
          requestedDtype_ == ArrayDtype::Float64 ? "float64" : "float32");
    }
    detectedDtype_ = bufDtype;
  } else {
    // Subsequent buffers - must match first
    MT_THROW_IF(
        detectedDtype_ != bufDtype,
        "In {}, dtype mismatch: first buffer was {}, but {} is {}",
        functionName_,
        detectedDtype_ == ArrayDtype::Float64 ? "float64" : "float32",
        bufferName,
        isFloat64 ? "float64" : "float32");
  }
}

void ArrayChecker::validateAndUpdateLeadingDims(
    const py::buffer_info& bufInfo,
    const char* bufferName,
    size_t numTrailingDims) {
  const auto totalNdim = static_cast<size_t>(bufInfo.ndim);
  const size_t leadingNdim = totalNdim - numTrailingDims;

  // Extract leading dimensions from this buffer
  LeadingDimensions bufLeadingDims;
  bufLeadingDims.dims.resize(leadingNdim);
  for (size_t i = 0; i < leadingNdim; ++i) {
    bufLeadingDims.dims[i] = bufInfo.shape[i];
  }

  if (!leadingDimsSet_) {
    // First buffer - set the leading dimensions
    leadingDims_ = bufLeadingDims;
    leadingDimsSet_ = true;
  } else {
    // Subsequent buffers - validate and broadcast
    MT_THROW_IF(
        leadingDims_.ndim() != bufLeadingDims.ndim(),
        "In {}, buffer argument {} has {} leading dimensions but expected {} "
        "(to match previous buffers). Buffer has shape {}.",
        functionName_,
        bufferName,
        bufLeadingDims.ndim(),
        leadingDims_.ndim(),
        formatArrayDims(std::vector<py::ssize_t>(bufInfo.shape.begin(), bufInfo.shape.end())));

    MT_THROW_IF(
        !leadingDims_.broadcastCompatibleWith(bufLeadingDims),
        "In {}, buffer argument {} has incompatible leading dimensions. "
        "Expected dimensions broadcastable with {} but got {}. Buffer has shape {}.",
        functionName_,
        bufferName,
        formatArrayDims(leadingDims_.dims),
        formatArrayDims(bufLeadingDims.dims),
        formatArrayDims(std::vector<py::ssize_t>(bufInfo.shape.begin(), bufInfo.shape.end())));

    // Update leading dims with broadcasted result
    leadingDims_ = leadingDims_.broadcastWith(bufLeadingDims);
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

void ArrayChecker::validateBuffer(
    const py::buffer& buffer,
    const char* bufferName,
    const std::vector<int>& trailingDims,
    const std::vector<const char*>& dimensionNames,
    bool allowEmpty) {
  MT_THROW_IF(
      trailingDims.size() != dimensionNames.size(),
      "In {}, internal error: trailingDims.size() != dimensionNames.size()",
      functionName_);

  const size_t numTrailingDims = trailingDims.size();

  // Get buffer info
  py::buffer_info bufInfo = buffer.request();

  // Handle empty buffers
  if (bufInfo.size == 0) {
    MT_THROW_IF(
        !allowEmpty,
        "In {}, buffer argument {} is empty. Expected {} trailing dimensions.",
        functionName_,
        bufferName,
        formatExpectedDims(trailingDims, dimensionNames, boundVariableSizes_));
    return;
  }

  // Validate dtype
  detectAndValidateDtype(bufInfo, bufferName);

  // Validate buffer has at least numTrailingDims dimensions
  MT_THROW_IF(
      static_cast<size_t>(bufInfo.ndim) < numTrailingDims,
      "In {}, buffer argument {} has shape {} but expected at least {} trailing dimensions: {}",
      functionName_,
      bufferName,
      formatArrayDims(std::vector<py::ssize_t>(bufInfo.shape.begin(), bufInfo.shape.end())),
      numTrailingDims,
      formatExpectedDims(trailingDims, dimensionNames, boundVariableSizes_));

  // Validate and update leading dimensions
  validateAndUpdateLeadingDims(bufInfo, bufferName, numTrailingDims);

  // Validate trailing dimensions
  const size_t leadingNdim = static_cast<size_t>(bufInfo.ndim) - numTrailingDims;
  for (size_t i = 0; i < numTrailingDims; ++i) {
    const auto foundSize = bufInfo.shape[leadingNdim + i];
    const int expectedSize = trailingDims[i];

    if (expectedSize < 0) {
      // Variable dimension - bind or validate
      auto itr = boundVariableSizes_.find(expectedSize);
      if (itr == boundVariableSizes_.end()) {
        boundVariableSizes_.emplace(expectedSize, foundSize);
      } else {
        MT_THROW_IF(
            foundSize != itr->second,
            "In {}, for buffer argument {}, mismatch in dimension {}; expected {} but found {}. "
            "Buffer has shape {}.",
            functionName_,
            bufferName,
            dimensionNames[i],
            itr->second,
            foundSize,
            formatArrayDims(std::vector<py::ssize_t>(bufInfo.shape.begin(), bufInfo.shape.end())));
      }
    } else {
      // Fixed dimension - validate
      MT_THROW_IF(
          foundSize != expectedSize,
          "In {}, for buffer argument {}, mismatch in dimension {}; expected {} but found {}. "
          "Buffer has shape {}.",
          functionName_,
          bufferName,
          dimensionNames[i],
          expectedSize,
          foundSize,
          formatArrayDims(std::vector<py::ssize_t>(bufInfo.shape.begin(), bufInfo.shape.end())));
    }
  }
}

void ArrayChecker::validateSkeletonState(
    const py::buffer& buffer,
    const char* bufferName,
    const momentum::Character& character) {
  const auto nJoints = static_cast<int>(character.skeleton.joints.size());
  validateBuffer(buffer, bufferName, {nJoints, 8}, {"numJoints", "8"});
}

JointParamsShape ArrayChecker::validateJointParameters(
    const py::buffer& buffer,
    const char* bufferName,
    const momentum::Character& character) {
  const auto nJoints = static_cast<int>(character.skeleton.joints.size());
  const auto nJointParams = nJoints * 7; // kParametersPerJoint

  // Get buffer info
  py::buffer_info bufInfo = buffer.request();

  // Joint parameters can be passed in two formats:
  // 1. Structured: (..., nJoints, 7)
  // 2. Flat: (..., nJointParams) where nJointParams = nJoints * 7
  //
  // We determine which format by checking the last dimension

  if (bufInfo.ndim == 0) {
    MT_THROW(
        "In {}, buffer argument {} is a scalar but expected shape "
        "(..., {}, 7) or (..., {})",
        functionName_,
        bufferName,
        nJoints,
        nJointParams);
  }

  const auto lastDim = bufInfo.shape[bufInfo.ndim - 1];

  if (lastDim == 7 && bufInfo.ndim >= 2) {
    // Structured format: (..., nJoints, 7)
    const auto secondLastDim = bufInfo.shape[bufInfo.ndim - 2];
    if (secondLastDim == nJoints) {
      validateBuffer(buffer, bufferName, {nJoints, 7}, {"numJoints", "7"});
      return JointParamsShape::Structured;
    }
  }

  // Either flat format or invalid - validate as flat and let it throw if wrong
  validateBuffer(buffer, bufferName, {nJointParams}, {"numJointParams"});
  return JointParamsShape::Flat;
}

void ArrayChecker::validateModelParameters(
    const py::buffer& buffer,
    const char* bufferName,
    const momentum::Character& character) {
  const auto nModelParams = static_cast<int>(character.parameterTransform.numAllModelParameters());
  validateBuffer(buffer, bufferName, {nModelParams}, {"numModelParams"});
}

TransformInputFormat ArrayChecker::validateTransforms(
    const py::buffer& buffer,
    const char* bufferName,
    const momentum::Character& character) {
  const auto nJoints = static_cast<int>(character.skeleton.joints.size());

  // Get buffer info
  py::buffer_info bufInfo = buffer.request();

  // Transforms can be passed in two formats:
  // 1. Skeleton state: (..., nJoints, 8)
  // 2. Transform matrix: (..., nJoints, 4, 4)
  //
  // We determine which format by checking the trailing dimensions

  if (bufInfo.ndim < 2) {
    MT_THROW(
        "In {}, buffer argument {} has shape {} but expected either "
        "(..., {}, 8) for skeleton state or (..., {}, 4, 4) for transform matrices",
        functionName_,
        bufferName,
        formatArrayDims(std::vector<py::ssize_t>(bufInfo.shape.begin(), bufInfo.shape.end())),
        nJoints,
        nJoints);
  }

  const auto lastDim = bufInfo.shape[bufInfo.ndim - 1];

  // Check for skeleton state format: (..., nJoints, 8)
  if (lastDim == 8 && bufInfo.ndim >= 2) {
    const auto secondLastDim = bufInfo.shape[bufInfo.ndim - 2];
    if (secondLastDim == nJoints) {
      validateBuffer(buffer, bufferName, {nJoints, 8}, {"numJoints", "8"});
      return TransformInputFormat::SkeletonState;
    }
  }

  // Check for transform matrix format: (..., nJoints, 4, 4)
  if (lastDim == 4 && bufInfo.ndim >= 3) {
    const auto secondLastDim = bufInfo.shape[bufInfo.ndim - 2];
    const auto thirdLastDim = bufInfo.shape[bufInfo.ndim - 3];
    if (secondLastDim == 4 && thirdLastDim == nJoints) {
      validateBuffer(buffer, bufferName, {nJoints, 4, 4}, {"numJoints", "4", "4"});
      return TransformInputFormat::TransformMatrix;
    }
  }

  // Neither format matched
  MT_THROW(
      "In {}, buffer argument {} has shape {} but expected either "
      "(..., {}, 8) for skeleton state or (..., {}, 4, 4) for transform matrices",
      functionName_,
      bufferName,
      formatArrayDims(std::vector<py::ssize_t>(bufInfo.shape.begin(), bufInfo.shape.end())),
      nJoints,
      nJoints);
}

void ArrayChecker::validateVertices(
    const py::buffer& buffer,
    const char* bufferName,
    const momentum::Character& character) {
  MT_THROW_IF(!character.mesh, "In {}, character has no mesh", functionName_);
  const auto nVertices = static_cast<int>(character.mesh->vertices.size());
  validateBuffer(buffer, bufferName, {nVertices, 3}, {"numVertices", "3"});
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
