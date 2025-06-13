/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/types.h>
#include <momentum/common/memory.h>
#include <momentum/math/utility.h>

#include <array>
#include <cstring>

namespace momentum {

enum LimitType {
  MinMax,
  MinMaxJoint,
  MinMaxJointPassive,
  Linear,
  LinearJoint,
  Ellipsoid,
  HalfPlane,

  // Keep this as the last entry to track the number of enum values
  LimitTypeCount
};

[[nodiscard]] std::string_view toString(LimitType type);

struct LimitMinMax {
  size_t parameterIndex;
  Vector2f limits; ///< [min, max]
};

struct LimitMinMaxJoint {
  size_t jointIndex;
  size_t jointParameter; ///< One of [tx, ty, tz, rx, ry, rz, sc]
  Vector2f limits; ///< [min, max]
};

/// Linear relationship between parameters: p_0 = s * p_1 - o
struct LimitLinear {
  size_t referenceIndex; ///< p_0
  size_t targetIndex; ///< p_1
  float scale; ///< s
  float offset; ///< o

  /// Range where limit applies (in target parameter values)
  ///
  /// For piecewise linear limits without double-counting
  float rangeMin; ///< Inclusive
  float rangeMax; ///< Non-inclusive
};

struct LimitLinearJoint {
  size_t referenceJointIndex;
  size_t referenceJointParameter;
  size_t targetJointIndex;
  size_t targetJointParameter;
  float scale;
  float offset;

  /// Range where limit applies (in target parameter values)
  ///
  /// For piecewise linear limits without double-counting
  float rangeMin; ///< Inclusive
  float rangeMax; ///< Non-inclusive
};

struct LimitEllipsoid {
  alignas(32) Affine3f ellipsoid;
  alignas(32) Affine3f ellipsoidInv;
  alignas(32) Vector3f offset;
  size_t ellipsoidParent;
  size_t parent;
};

/// Constraint: (p1, p2) · (normal) - offset >= 0
struct LimitHalfPlane {
  size_t param1;
  size_t param2;
  Vector2f normal;
  float offset;
};

union LimitData {
  LimitMinMax minMax;
  LimitMinMaxJoint minMaxJoint;
  LimitLinear linear;
  LimitLinearJoint linearJoint;
  LimitEllipsoid ellipsoid;
  LimitHalfPlane halfPlane;
  std::array<unsigned char, 512> rawData; ///< For memory operations

  /// Initializes all data to zero
  LimitData();

  /// Raw memory copy (compiler can't determine which member's constructor to call)
  LimitData(const LimitData& rhs);

  LimitData& operator=(const LimitData& rhs);

  /// Compares raw memory
  bool operator==(const LimitData& limitData) const;
};

struct ParameterLimit {
  LimitData data;
  LimitType type = LimitType::MinMax;
  float weight = 1.0f;

  inline bool operator==(const ParameterLimit& parameterLimit) const {
    return (
        (data == parameterLimit.data) && (type == parameterLimit.type) &&
        isApprox(weight, parameterLimit.weight));
  }
};

using ParameterLimits = std::vector<ParameterLimit>;

/// Only processes MinMaxJointPassive limits, clamping parameters to their ranges
JointParameters applyPassiveJointParameterLimits(
    const ParameterLimits& limits,
    const JointParameters& jointParams);

/// Creates MinMax limits from a pose constraint
ParameterLimits getPoseConstraintParameterLimits(
    const std::string& name,
    const ParameterTransform& pt,
    float weight = 1.0f);

/// If rangeMin and rangeMax are both 0, the limit applies to all values
bool isInRange(const LimitLinear& limit, float value);

/// If rangeMin and rangeMax are both 0, the limit applies to all values
bool isInRange(const LimitLinearJoint& limit, float value);

MOMENTUM_DEFINE_POINTERS(ParameterLimits)
} // namespace momentum
