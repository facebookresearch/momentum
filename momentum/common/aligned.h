/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/exception.h>

#include <limits>
#include <memory>

namespace momentum {

#if defined(_WIN32)

[[nodiscard]] inline void* aligned_malloc(size_t size, size_t align) {
  return _aligned_malloc(size, align);
}

inline void aligned_free(void* ptr) {
  return _aligned_free(ptr);
}

#elif (defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && __IPHONE_OS_VERSION_MIN_REQUIRED < 130000) || \
    (defined(__ANDROID_API__) && __ANDROID_API__ < 28)

[[nodiscard]] inline void* aligned_malloc(size_t size, size_t align) {
  void* result = nullptr;
  posix_memalign(&result, align, size);
  return result;
}

inline void aligned_free(void* ptr) {
  return std::free(ptr);
}

#else

[[nodiscard]] inline void* aligned_malloc(size_t size, size_t align) {
  return std::aligned_alloc(align, size);
}

inline void aligned_free(void* ptr) {
  return std::free(ptr);
}

#endif

/// Rounds a value up to the nearest multiple of the specified alignment.
/// @param value The value to round up.
/// @param alignment The alignment boundary (must be non-zero).
/// @return The smallest multiple of alignment that is >= value.
inline constexpr std::size_t roundUpToAlignment(std::size_t value, std::size_t alignment) {
  MT_THROW_IF_T(alignment == 0, std::invalid_argument, "Alignment must be non-zero");
  // Formula: ((value + alignment - 1) / alignment) * alignment
  // Rounds up by adding (alignment-1) before integer division truncates, then re-multiplies
  return ((value + alignment - 1) / alignment) * alignment;
}

/// Allocates a block of memory that can hold `n` elements of type `T` with the specified alignment.
///
/// This function is intended to be used in the `AlignedAllocator::allocate()` method and throws
/// exceptions as `std::allocator<T>::allocate` does.
template <typename T, std::size_t Alignment = alignof(T)>
[[nodiscard]] T* alignedAlloc(std::size_t n) {
  MT_THROW_IF_T(std::numeric_limits<std::size_t>::max() / sizeof(T) < n, std::bad_array_new_length);

  const std::size_t size = roundUpToAlignment(n * sizeof(T), Alignment);
  void* ptr = aligned_malloc(size, Alignment);

  MT_THROW_IF_T(ptr == nullptr, std::bad_alloc);

  return static_cast<T*>(ptr);
}

/// Custom deleter for aligned memory
struct AlignedDeleter {
  void operator()(void* ptr) const {
    aligned_free(ptr);
  }
};

/// Creates a std::unique_ptr for aligned memory.
template <typename T, std::size_t Alignment = alignof(T), class Deleter = AlignedDeleter>
[[nodiscard]] std::unique_ptr<T, Deleter> makeAlignedUnique(
    std::size_t n,
    Deleter deleter = Deleter()) {
  return std::unique_ptr<T, Deleter>(alignedAlloc<T, Alignment>(n), std::move(deleter));
}

/// Creates a std::shared_ptr for aligned memory.
template <typename T, std::size_t Alignment = alignof(T), class Deleter = AlignedDeleter>
[[nodiscard]] std::shared_ptr<T> makeAlignedShared(std::size_t n, Deleter deleter = Deleter()) {
  return std::shared_ptr<T>(alignedAlloc<T, Alignment>(n), std::move(deleter));
}

/// An allocator that aligns memory blocks according to a specified alignment.
/// The allocator is compatible with `std::allocator` and can be used in
/// place of `std::allocator` in STL containers.
///
/// @tparam T The type of elements that the allocator will allocate.
/// @tparam Alignment The alignment for the allocated memory blocks.
template <class T, std::size_t Alignment>
class AlignedAllocator {
 public:
  using value_type = T;

  AlignedAllocator() noexcept = default;

  template <class U>
  explicit AlignedAllocator(const AlignedAllocator<U, Alignment>& /*other*/) noexcept {}

  /// Allocates a block of memory that can hold `n` elements of type `T`.
  ///
  /// @param n The number of elements to allocate space for.
  /// @return A pointer to the first byte of the allocated memory block.
  [[nodiscard]] T* allocate(std::size_t n) {
    return alignedAlloc<T, Alignment>(n);
  }

  /// Deallocates a block of memory that was previously allocated by `allocate`.
  ///
  /// @param ptr A pointer to the first byte of the memory block to deallocate.
  /// @param size The size of the memory block to deallocate. This parameter is not used, but it is
  /// included to maintain compatibility with `std::allocator`.
  void deallocate(T* ptr, std::size_t /*n*/) noexcept {
    aligned_free(ptr);
  }

  template <class U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};

/// Checks if storage allocated from `lhs` can be deallocated from `rhs`, and vice versa.
///
/// This always returns true for stateless allocators.
template <class T, class U, std::size_t Alignment>
[[nodiscard]] bool operator==(
    const AlignedAllocator<T, Alignment>& /*lhs*/,
    const AlignedAllocator<U, Alignment>& /*rhs*/) {
  return true;
}

/// Checks if storage allocated from `lhs` cannot be deallocated from `rhs`, and vice versa.
///
/// This always returns false for stateless allocators.
template <class T, class U, std::size_t Alignment>
[[nodiscard]] bool operator!=(
    const AlignedAllocator<T, Alignment>& /*lhs*/,
    const AlignedAllocator<U, Alignment>& /*rhs*/) {
  return false;
}

} // namespace momentum
