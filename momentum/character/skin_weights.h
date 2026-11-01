/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/types.h>
#include <momentum/math/types.h>

#include <cstdint>
#include <vector>

namespace momentum {

/// Maximum number of joints that can influence a single vertex
inline constexpr uint32_t kMaxSkinJoints = 8;

/// Joint indices that influence each vertex. Row = vertex, column = influence slot
/// (fixed at kMaxSkinJoints). Row-major to keep per-vertex influences contiguous.
using IndexMatrix =
    Eigen::Matrix<uint32_t, Eigen::Dynamic, kMaxSkinJoints, Eigen::AutoAlign | Eigen::RowMajor>;

/// Skinning weights for each joint influence. Row = vertex, column = influence slot
/// (fixed at kMaxSkinJoints). Row-major to keep per-vertex influences contiguous.
using WeightMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, kMaxSkinJoints, Eigen::AutoAlign | Eigen::RowMajor>;

/// Linear blend skinning weights for a character mesh: per-vertex joint indices
/// and influence weights. `index` and `weight` always have the same row count
/// (one row per vertex). Active weights are positive, sorted in non-increasing
/// order, and packed before zero-filled unused slots so skinning can stop at the
/// first zero weight.
struct SkinWeights {
  /// Joint indices influencing each vertex. Unused slots are set to 0.
  /// Read in conjunction with `weight` — index 0 only contributes when its
  /// matching weight is non-zero.
  IndexMatrix index;

  /// Joint influence weights per vertex. Active weights are finite,
  /// non-negative, and expected to sum to 1.0; unused slots are 0.0.
  WeightMatrix weight;

  /// Populates `index` and `weight` from ragged per-vertex influence lists.
  /// Zero-weight, non-finite, and negative influences are discarded with a
  /// warning. The rest are stable-sorted by descending weight. If a vertex has
  /// more than `kMaxSkinJoints` positive influences, the largest are retained
  /// without changing their values; this function does not normalize weights.
  /// The discarded count is reported in an aggregate warning.
  ///
  /// @pre `ind.size() == wgt.size()` (checked via `MT_CHECK`).
  /// @pre `ind[i].size() == wgt[i].size()` for every vertex.
  void set(const std::vector<std::vector<size_t>>& ind, const std::vector<std::vector<float>>& wgt);

  /// Approximate equality on `index` and `weight` using Eigen's `isApprox`.
  bool operator==(const SkinWeights& skinWeights) const;
};

} // namespace momentum
