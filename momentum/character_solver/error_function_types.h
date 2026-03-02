/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/// @file error_function_types.h
/// Shared types used across error function hierarchies.
///
/// Contains base data structures and enums that are shared between
/// JointErrorFunctionT, VertexErrorFunctionT, MultiSourceErrorFunctionT,
/// and standalone error functions.

#include <momentum/character/types.h>

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace momentum {

// ============================================================================
// ConstraintData — base struct for joint-based constraint data
// ============================================================================

/// Base structure of constraint data for joint-based error functions.
///
/// Provides the common fields shared by all constraint types: the parent joint
/// index, constraint weight, and an optional name for debugging.
struct ConstraintData {
  /// Parent joint index this constraint is under
  size_t parent = kInvalidIndex;
  /// Weight of the constraint
  float weight = 0.0f;
  /// Name of the constraint
  std::string name = {};

  ConstraintData(size_t pIndex, float w, const std::string& n = "")
      : parent(pIndex), weight(w), name(n) {}
};

/// A list of ConstraintData
using ConstraintDataList = std::vector<ConstraintData>;

/// An optional of a reference. Because optional<T&> is invalid, we need to use
/// reference_wrapper to make the reference of T as an optional.
template <typename T>
using optional_ref = std::optional<std::reference_wrapper<T>>;

// ============================================================================
// VertexConstraintType — mode selector for vertex constraint functions
// ============================================================================

/// Type of vertex constraint mode.
///
/// Used by error functions that can operate in different vertex constraint modes
/// (Position, Plane, Normal, SymmetricNormal).
enum class VertexConstraintType {
  Position,
  Plane,
  Normal,
  SymmetricNormal,
};

/// Returns a string representation of the constraint type.
[[nodiscard]] std::string_view toString(VertexConstraintType type);

} // namespace momentum
