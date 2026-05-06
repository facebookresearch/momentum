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

// Detect platforms with std::aligned_alloc (C11/C++17).
// We explicitly check for known platforms rather than assuming availability,
// so that unknown/embedded platforms safely fall back to malloc.
#if !defined(MOMENTUM_HAS_ALIGNED_ALLOC)
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 16))
// glibc 2.16+ has aligned_alloc
#define MOMENTUM_HAS_ALIGNED_ALLOC 1
#elif defined(__APPLE__)
// macOS 10.15+, iOS 13+, and other modern Apple platforms have aligned_alloc
#if (defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500) ||   \
    (defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && __IPHONE_OS_VERSION_MIN_REQUIRED >= 130000) || \
    (defined(__TV_OS_VERSION_MIN_REQUIRED) && __TV_OS_VERSION_MIN_REQUIRED >= 130000) ||         \
    (defined(__WATCH_OS_VERSION_MIN_REQUIRED) && __WATCH_OS_VERSION_MIN_REQUIRED >= 60000)
#define MOMENTUM_HAS_ALIGNED_ALLOC 1
#endif
#elif defined(__ANDROID__) && defined(__ANDROID_API__) && __ANDROID_API__ >= 28
// Android API 28+ has aligned_alloc
#define MOMENTUM_HAS_ALIGNED_ALLOC 1
#elif defined(__EMSCRIPTEN__)
// Emscripten (WebAssembly) has aligned_alloc
#define MOMENTUM_HAS_ALIGNED_ALLOC 1
#endif
#endif

#if !defined(MOMENTUM_HAS_ALIGNED_ALLOC)
#define MOMENTUM_HAS_ALIGNED_ALLOC 0
#endif

// Detect platforms with posix_memalign.
#if !defined(MOMENTUM_HAS_POSIX_MEMALIGN)
#if (defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)) && !defined(_WIN32)
#define MOMENTUM_HAS_POSIX_MEMALIGN 1
#else
#define MOMENTUM_HAS_POSIX_MEMALIGN 0
#endif
#endif

#if defined(_WIN32)

[[nodiscard]] inline void* aligned_malloc(size_t size, size_t align) {
  return _aligned_malloc(size, align);
}

inline void aligned_free(void* ptr) {
  return _aligned_free(ptr);
}

#elif MOMENTUM_HAS_ALIGNED_ALLOC

[[nodiscard]] inline void* aligned_malloc(size_t size, size_t align) {
  return std::aligned_alloc(align, size);
}

inline void aligned_free(void* ptr) {
  return std::free(ptr);
}

#elif MOMENTUM_HAS_POSIX_MEMALIGN

[[nodiscard]] inline void* aligned_malloc(size_t size, size_t align) {
  void* result = nullptr;
  posix_memalign(&result, align, size);
  return result;
}

inline void aligned_free(void* ptr) {
  return std::free(ptr);
}

#else

// Fallback for platforms without aligned allocation support (e.g., embedded systems).
// Uses standard malloc which may not honor alignment requirements,
// but Eigen and other users will still work (just potentially slower).
[[nodiscard]] inline void* aligned_malloc(size_t size, size_t /*align*/) {
  return std::malloc(size);
}

inline void aligned_free(void* ptr) {
  return std::free(ptr);
}

#endif

/// Rounds `value` up to the nearest multiple of `alignment` (in bytes).
///
/// @pre `alignment` must be non-zero. Throws `std::invalid_argument` otherwise.
/// @return The smallest multiple of `alignment` that is >= `value`.
inline constexpr std::size_t roundUpToAlignment(std::size_t value, std::size_t alignment) {
  MT_THROW_IF_T(alignment == 0, std::invalid_argument, "Alignment must be non-zero");
  return ((value + alignment - 1) / alignment) * alignment;
}

/// Allocates a block of memory that can hold `n` elements of type `T` with the specified alignment.
///
/// This function is intended to be used in the `AlignedAllocator::allocate()` method and throws
/// exceptions as `std::allocator<T>::allocate` does. The allocated size is rounded up to a multiple
/// of `Alignment` so it satisfies the size requirements of `std::aligned_alloc` on platforms where
/// it is used. The returned pointer must be released with `aligned_free`.
///
/// @throws std::bad_array_new_length if `n * sizeof(T)` would overflow.
/// @throws std::bad_alloc if the underlying aligned allocation fails.
template <typename T, std::size_t Alignment = alignof(T)>
[[nodiscard]] T* alignedAlloc(std::size_t n) {
  MT_THROW_IF_T(std::numeric_limits<std::size_t>::max() / sizeof(T) < n, std::bad_array_new_length);

  // C/POSIX aligned allocation APIs require at least pointer-sized alignment.
  constexpr std::size_t kAllocationAlignment =
      Alignment < alignof(void*) ? alignof(void*) : Alignment;
  const std::size_t size = roundUpToAlignment(n * sizeof(T), kAllocationAlignment);
  void* ptr = aligned_malloc(size, kAllocationAlignment);

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
  /// @param[in] n The number of elements to allocate space for.
  /// @return A pointer to the first byte of the allocated memory block.
  [[nodiscard]] T* allocate(std::size_t n) {
    return alignedAlloc<T, Alignment>(n);
  }

  /// Deallocates a block of memory that was previously allocated by `allocate`.
  ///
  /// @param[in] ptr A pointer to the first byte of the memory block to deallocate.
  /// @param n The number of elements in the block. Unused; included to match the
  /// `std::allocator` interface.
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
