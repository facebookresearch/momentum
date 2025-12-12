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

/// Matrix type for storing joint indices that influence each vertex
///
/// Each row represents a vertex, and each column represents a joint influence.
/// The matrix has a fixed number of columns (kMaxSkinJoints) and a dynamic number of rows.
using IndexMatrix =
    Eigen::Matrix<uint32_t, Eigen::Dynamic, kMaxSkinJoints, Eigen::AutoAlign | Eigen::RowMajor>;

/// Matrix type for storing weights of joint influences on each vertex
///
/// Each row represents a vertex, and each column represents the weight of a joint influence.
/// The matrix has a fixed number of columns (kMaxSkinJoints) and a dynamic number of rows.
using WeightMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, kMaxSkinJoints, Eigen::AutoAlign | Eigen::RowMajor>;

/// Stores skinning weights and joint indices for character mesh deformation
struct SkinWeights {
  /// Joint indices that influence each vertex
  ///
  /// Each row corresponds to a vertex, and each column contains the index of a joint
  /// that influences that vertex. Unused influences are set to 0.
  IndexMatrix index;

  /// Weight of each joint's influence on each vertex
  ///
  /// Each row corresponds to a vertex, and each column contains the weight of a joint's
  /// influence on that vertex. Weights for a vertex typically sum to 1.0. Unused influences
  /// are set to 0.0.
  WeightMatrix weight;

  /// Sets the skin weights from vectors of joint indices and weights
  ///
  /// @param ind Vector of vectors containing joint indices for each vertex
  /// @param wgt Vector of vectors containing weights for each vertex
  /// @throws If ind.size() != wgt.size() (via MT_CHECK)
  void set(const std::vector<std::vector<size_t>>& ind, const std::vector<std::vector<float>>& wgt);

  /// Compares two SkinWeights objects for equality
  ///
  /// Two SkinWeights objects are considered equal if their index and weight matrices
  /// are approximately equal (using Eigen's isApprox method).
  ///
  /// @param skinWeights The SkinWeights object to compare with
  /// @return True if the objects are equal, false otherwise
  bool operator==(const SkinWeights& skinWeights) const;
};

} // namespace momentum
