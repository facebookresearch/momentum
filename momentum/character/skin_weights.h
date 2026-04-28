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
/// (one row per vertex) and matching unused-slot conventions.
struct SkinWeights {
  /// Joint indices influencing each vertex. Unused slots are set to 0.
  /// Read in conjunction with `weight` — index 0 only contributes when its
  /// matching weight is non-zero.
  IndexMatrix index;

  /// Joint influence weights per vertex. Weights for a vertex are expected to
  /// sum to 1.0; unused slots are set to 0.0. Loaders/setters do not enforce
  /// normalization — callers are responsible for normalized input.
  WeightMatrix weight;

  /// Populates `index` and `weight` from ragged per-vertex influence lists.
  /// Each entry of `ind`/`wgt` is one vertex's joints/weights; influences beyond
  /// `kMaxSkinJoints` are silently dropped, and unused slots are zero-filled.
  ///
  /// @pre `ind.size() == wgt.size()` (checked via MT_CHECK).
  /// @pre For each vertex i, `wgt[i].size() >= min(ind[i].size(), kMaxSkinJoints)`;
  ///   shorter weight rows produce out-of-bounds reads.
  void set(const std::vector<std::vector<size_t>>& ind, const std::vector<std::vector<float>>& wgt);

  /// Approximate equality on `index` and `weight` using Eigen's `isApprox`.
  bool operator==(const SkinWeights& skinWeights) const;
};

} // namespace momentum
