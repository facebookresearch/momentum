/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <Eigen/Core>

#include <optional>
#include <unordered_map>

namespace pymomentum {

// Convert C++ type T(float, double, int) to tensor data type at::ScalarType.
template <typename T>
constexpr at::ScalarType toScalarType();

#define PYMOMENTUM_TO_SCALAR_TYPE(type, val)                    \
  template <>                                                   \
  [[nodiscard]] constexpr at::ScalarType toScalarType<type>() { \
    return at::ScalarType::val;                                 \
  }
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(PYMOMENTUM_TO_SCALAR_TYPE)
#undef PYMOMENTUM_TO_SCALAR_TYPE

// Convert cpu tensor to Eigen::Map of a dim-1 vector.
// Will check whether tensor data type is the same as T. If not, throw error.
template <typename T>
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> toEigenMap(at::Tensor t);

std::string formatTensorSizes(const at::Tensor& tensor);
std::string formatTensorSizes(const std::vector<pybind11::ssize_t>& dims);

inline bool isEmpty(const at::Tensor& t) {
  return (t.numel() == 0);
}

// Turn a N x Rows x Cols tensor into a list of Eigen matrices:
template <typename T, int Rows, int Cols>
std::vector<Eigen::Matrix<T, Rows, Cols>> toMatrixList(at::Tensor tensor);

// If the input std::optional<at::Tensor> does not contain value, return an
// empty tensor. Otherwise return the contained tensor.
at::Tensor denullify(std::optional<at::Tensor> tensor);

// Check for NaN and INF in a tensor, throw runtime error with provided
// strings in the error message: "In <context>, <tensorName> with dimension
// [...] has nan or inf."
void throwIfNaNOrINF(const at::Tensor& t, const char* context, const char* tensorName);

// Build an 1D tensor from external buffer. Support data type
// float, double, int, int64_t.
//
// We call clone() here because from_blob() creates a tensor viewing
// external memory. If the memory is released but the tensor is still
// active, memory errors will be reported when trying to access the tensor
// content. Even if this tensor is built locally to attend some computation
// to generate a returned tensor, like in:
// at::Tensor f(at::Tensor in) { auto temp = from_blob(...); return temp + in; }
// As long as autograd is on tensor `in`, tensor `temp` will be referenced
// by the autograd graph, and later backward pass will visit `temp`, and
// subsequently the released memory used by from_blob().
// To avoid this, we call clone() to make sure the generated tensor owns
// the memory.
template <typename T>
inline at::Tensor to1DTensor(const T* data, int64_t size) {
  return torch::from_blob((void*)data, {size}, torch::TensorOptions().dtype(toScalarType<T>()))
      .clone();
}

// Build 1D tensor from Eigen dense Vector. Support data type
// float, double, int, int64_t.
template <typename T, int _Rows>
inline at::Tensor to1DTensor(const Eigen::Matrix<T, _Rows, 1>& vec) {
  return to1DTensor(vec.data(), vec.size());
}

// Build 1D tensor from std::vector. Support data type
// float, double, int, int64_t.
template <typename T>
inline at::Tensor to1DTensor(const std::vector<T>& vec) {
  return to1DTensor(vec.data(), vec.size());
}

// Build 2D tensor from Eigen row-major dense Matrix. Support data type
// float, double, int, int64_t.
template <typename T, int _Rows, int _Cols>
at::Tensor to2DTensor(const Eigen::Matrix<T, _Rows, _Cols, Eigen::RowMajor>& mat) {
  return torch::from_blob(
             (void*)mat.data(),
             {mat.rows(), mat.cols()},
             torch::TensorOptions().dtype(toScalarType<T>()))
      .clone();
}

// Build 2D tensor from Eigen col-major dense Matrix. Support data type
// float, double, int, int64_t.
template <typename T, int _Rows, int _Cols>
at::Tensor to2DTensor(const Eigen::Matrix<T, _Rows, _Cols, Eigen::ColMajor>& mat) {
  Eigen::Matrix<T, _Rows, _Cols, Eigen::RowMajor> matRowMajor = mat;
  return to2DTensor(matRowMajor);
}

// Class to validate tensors as inputs for a function
//   foo(at::Tensor v1, at::Tensor v2, ...)
//
//
//
class TensorChecker {
 public:
  explicit TensorChecker(const char* functionName) : _functionName(functionName) {}

  // Performs the following validations:
  //   1. Checks that all the dimensions are correct
  //   2. If necessary (and if allowUnbatched is true), adds a batch dimension
  //   3. Convert to the desired type and make sure it's on the CPU and
  //   contiguous.
  //
  // tensor_in: the input tensor
  // expectedSizes: the expected input sizes (excluding the batch dimension).
  //   These are allowed to have certain "variable" parameters, which are
  //   indicated by passing in a negative value for the size; these variables
  //   are "bound" the first time they are encountered and then afterward
  //   are validated just like fixed size parameters.
  //
  //   Example: validateAndFixTensor(t, {2, -1}) where t has size {2, 5}
  //   would "bind" 5 to the -1 variable parameter, and so if you then
  //   passed in validateAndFixTensor(t, {3, -1}) for a tensor of size
  //   {3, 7} it would fail.
  // expectedType: aten type for the desired type value.  You can use
  //   toScalarType<T>() to derive this if desired.
  // allowUnbatched: If true, then tensors without a batch dimension will
  //   match against any batch dimension and they will then be expanded to
  //   match the stored batch dimension. If false then tensors without a
  //   batch dimension will only match against a batch dimension of size 1.
  // unsqueezed: output parameter that is set if the input was unsqueezed
  //   (that is, a batch dimension was added); this can be used to determine
  //   if the result should be squeezed.
  //
  // Note: the batch size is determined by the first tensor validated, and
  // all subsequent tensors validate against this.  That means that the
  // first tensor is "special" in the sense that if it is unbatched then
  // all tensors must be.
  at::Tensor validateAndFixTensor(
      at::Tensor tensor_orig,
      const char* tensorName,
      const std::vector<int>& expectedSizes,
      const std::vector<const char*>& dimensionNames,
      at::ScalarType expectedScalarType,
      bool allowUnbatched = true,
      bool allowEmpty = false,
      bool* needsSqueeze_out = nullptr);

  int64_t getBoundValue(int idx);
  int64_t getBatchSize();

 private:
  const char* _functionName;

  // Set to -1 until it is set by the first viewed tensor.
  int64_t _batchSize = -1;
  std::unordered_map<int, int64_t> _boundVariableSizes;
};

} // namespace pymomentum
