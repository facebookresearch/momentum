/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <mdspan/mdspan.hpp>
#include <momentum/common/aligned.h>
#include <momentum/rasterizer/fwd.h>
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>

namespace momentum::rasterizer {

using index_t = std::ptrdiff_t;

// Helper to compute contiguous strides for row-major layout
template <size_t Rank>
inline std::array<index_t, Rank> computeContiguousStrides(
    const std::array<index_t, Rank>& extents) {
  std::array<index_t, Rank> strides{};
  index_t stride = 1;
  for (size_t i = Rank; i > 0; --i) {
    strides[i - 1] = stride;
    stride *= extents[i - 1];
  }
  return strides;
}

// Simplified Tensor class local to Rasterizer
// Uses std::vector with aligned allocator and provides conversion to std::mdspan
template <typename T, size_t Rank>
class Tensor {
 private:
  using extents_t = std::array<index_t, Rank>;
  using allocator_t = momentum::AlignedAllocator<T, kSimdAlignment>;
  using storage_t = std::vector<T, allocator_t>;
  using strides_t = std::array<index_t, Rank>;
  using span_t = Kokkos::mdspan<T, Kokkos::dextents<index_t, Rank>, Kokkos::layout_stride>;
  using const_span_t =
      Kokkos::mdspan<const T, Kokkos::dextents<index_t, Rank>, Kokkos::layout_stride>;

  extents_t _extents = {};
  strides_t _strides = {};
  storage_t _data;

 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;

  Tensor() = default;

  explicit Tensor(const extents_t& extents, const T& defaultValue = T{})
      : _extents(extents), _strides(computeContiguousStrides(extents)) {
    auto size = calculateSize(extents);
    if (size > 0) {
      _data.resize(size, defaultValue);
    }
  }

  template <typename... SizeTypes>
  explicit Tensor(SizeTypes... dims) : Tensor(extents_t{static_cast<index_t>(dims)...}) {}

  // Copy constructor, move constructor, copy assignment, and move assignment
  // are automatically generated and work correctly with std::vector

  // Conversion to mdspan (layout_stride for compatibility with Span types)
  explicit operator span_t() {
    return view();
  }

  explicit operator const_span_t() const {
    return view();
  }

  // View method for compatibility - returns layout_stride mdspan
  span_t view() {
    using ExtentsType = Kokkos::dextents<index_t, Rank>;
    using MappingType = Kokkos::layout_stride::mapping<ExtentsType>;
    return span_t(_data.data(), MappingType(ExtentsType(_extents), _strides));
  }

  [[nodiscard]] const_span_t view() const {
    using ExtentsType = Kokkos::dextents<index_t, Rank>;
    using MappingType = Kokkos::layout_stride::mapping<ExtentsType>;
    return const_span_t(_data.data(), MappingType(ExtentsType(_extents), _strides));
  }

  [[nodiscard]] const_span_t const_view() const {
    return view();
  }

  // Multi-dimensional indexing operator() inspired by ASpan
  template <typename... Args>
  std::enable_if_t<sizeof...(Args) == Rank, T&> operator()(Args... indices) {
    static_assert(sizeof...(Args) == Rank, "Number of indices must match tensor rank");
    return view()(indices...);
  }

  template <typename... Args>
  std::enable_if_t<sizeof...(Args) == Rank, const T&> operator()(Args... indices) const {
    static_assert(sizeof...(Args) == Rank, "Number of indices must match tensor rank");
    return view()(indices...);
  }

  // Basic accessors
  T* data() {
    return _data.data();
  }
  [[nodiscard]] const T* data() const {
    return _data.data();
  }

  [[nodiscard]] constexpr index_t extent(size_t dim) const {
    return _extents[dim];
  }
  [[nodiscard]] constexpr extents_t extents() const {
    return _extents;
  }
  [[nodiscard]] constexpr strides_t strides() const {
    return _strides;
  }

  [[nodiscard]] index_t size() const {
    return static_cast<index_t>(_data.size());
  }
  [[nodiscard]] bool empty() const {
    return _data.empty();
  }

  void resize(const extents_t& new_extents) {
    auto new_size = calculateSize(new_extents);
    _data.resize(new_size);
    _extents = new_extents;
    _strides = computeContiguousStrides(new_extents);
  }

  template <typename... SizeTypes>
  void resize(SizeTypes... dims) {
    resize(extents_t{static_cast<index_t>(dims)...});
  }

 private:
  static index_t calculateSize(const extents_t& extents) {
    index_t size = 1;
    for (auto ext : extents) {
      size *= ext;
    }
    return size;
  }
};

using Tensor2f = Tensor<float, 2>;
using Tensor3f = Tensor<float, 3>;
using Tensor2i = Tensor<int, 2>;

} // namespace momentum::rasterizer
