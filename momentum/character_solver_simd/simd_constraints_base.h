/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/skeleton.h>
#include <momentum/common/aligned.h>
#include <momentum/common/checks.h>
#include <momentum/math/types.h>
#include <momentum/simd/simd.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>

namespace momentum {

/// Storage helper shared by all SIMD constraint structs.
///
/// Allocates a single aligned float buffer striped into N equal-sized blocks (one block per
/// constraint field, e.g., offsetX, offsetY, weights). Derived types name the blocks via raw
/// pointers obtained from `block(i)`. The atomic per-joint counter and the addConstraint /
/// clearConstraints helpers are also shared, since the locking pattern is identical across all
/// SIMD constraint types.
///
/// @tparam NumBlocks Number of striped float blocks (one per scalar constraint field).
template <size_t NumBlocks>
class SimdConstraintsBase {
 public:
  /// Maximum number of constraints stored per joint. Constraints beyond this are dropped by
  /// addConstraint.
  static constexpr size_t kMaxConstraints = 4096;

  /// Maximum number of joints supported.
  static constexpr size_t kMaxJoints = 512;

  explicit SimdConstraintsBase(const Skeleton* skel) {
    MT_CHECK(skel->joints.size() <= kMaxJoints);
    numJoints_ = static_cast<int>(skel->joints.size());
    const size_t dataSize = kMaxConstraints * static_cast<size_t>(numJoints_);
    MT_CHECK(dataSize % kSimdAlignment == 0);

    data_ = makeAlignedUnique<float, kSimdAlignment>(dataSize * NumBlocks);
    std::fill_n(data_.get(), dataSize * NumBlocks, 0.0f);

    for (auto& count : constraintCount_) {
      count.store(0, std::memory_order_relaxed);
    }
  }

  ~SimdConstraintsBase() = default;

  SimdConstraintsBase(const SimdConstraintsBase&) = delete;
  SimdConstraintsBase& operator=(const SimdConstraintsBase&) = delete;
  SimdConstraintsBase(SimdConstraintsBase&&) = delete;
  SimdConstraintsBase& operator=(SimdConstraintsBase&&) = delete;

  /// Reset all per-joint counters and zero the weights block.
  ///
  /// @param weightsBlockIndex Index of the block holding constraint weights. Zeroing weights
  /// alone is sufficient because zero-weight rows contribute nothing to error/gradient/Jacobian.
  void clearImpl(size_t weightsBlockIndex) {
    std::fill_n(block(weightsBlockIndex), kMaxConstraints * numJoints_, 0.0f);
    for (int i = 0; i < numJoints_; ++i) {
      constraintCount_[i].store(0, std::memory_order_relaxed);
    }
  }

  /// Number of constraints currently stored for each joint.
  [[nodiscard]] VectorXi getNumConstraints() const {
    VectorXi res = VectorXi::Zero(numJoints_);
    for (int jointIndex = 0; jointIndex < numJoints_; ++jointIndex) {
      res[jointIndex] = constraintCount_[jointIndex].load(std::memory_order_relaxed);
    }
    return res;
  }

  /// Number of joints. Public for backward-compat field-access (`numJoints`).
  [[nodiscard]] int numJoints() const noexcept {
    return numJoints_;
  }

 protected:
  /// Atomically reserves a slot for a new constraint on `jointIndex`. Returns the slot index, or
  /// `kMaxConstraints` if the joint is full (caller must not write).
  ///
  /// Lock-free CAS loop allows concurrent producers from multiple threads.
  [[nodiscard]] uint32_t reserveSlot(size_t jointIndex) {
    MT_CHECK(jointIndex < constraintCount_.size());
    uint32_t index = 0;
    while (true) {
      index = constraintCount_[jointIndex].load(std::memory_order_relaxed);
      if (index == kMaxConstraints) {
        return kMaxConstraints;
      }
      if (constraintCount_[jointIndex].compare_exchange_weak(index, index + 1u)) {
        return index;
      }
    }
  }

  /// Pointer to block `i` of the striped data buffer.
  [[nodiscard]] float* block(size_t i) noexcept {
    return data_.get() + (kMaxConstraints * static_cast<size_t>(numJoints_)) * i;
  }
  [[nodiscard]] const float* block(size_t i) const noexcept {
    return data_.get() + (kMaxConstraints * static_cast<size_t>(numJoints_)) * i;
  }

  /// Linear offset into a striped block for a given (joint, constraintIndex) pair.
  [[nodiscard]] static size_t flatIndex(size_t jointIndex, size_t constraintIndex) noexcept {
    return jointIndex * kMaxConstraints + constraintIndex;
  }

  std::unique_ptr<float, AlignedDeleter> data_;
  std::array<std::atomic<uint32_t>, kMaxJoints> constraintCount_;
  int numJoints_ = 0;
};

} // namespace momentum
