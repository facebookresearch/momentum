/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/parameter_limits.h>

#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton.h>
#include <momentum/common/checks.h>
#include <momentum/math/utility.h>

namespace momentum {

std::string_view toString(const LimitType type) {
  switch (type) {
    case MinMax:
      return "MinMax";
    case MinMaxJoint:
      return "MinMaxJoint";
    case MinMaxJointPassive:
      return "MinMaxJointPassive";
    case Linear:
      return "Linear";
    case Ellipsoid:
      return "Ellipsoid";
    default:
      return "Unknown";
  }
}

JointParameters applyPassiveJointParameterLimits(
    const ParameterLimits& limits,
    const JointParameters& jointParams) {
  JointParameters res = jointParams;
  for (const auto& limit : limits) {
    if (limit.type != MinMaxJointPassive) {
      continue;
    }

    const auto& data = limit.data.minMaxJoint;
    const size_t parameterIndex = data.jointIndex * kParametersPerJoint + data.jointParameter;
    MT_CHECK(
        parameterIndex <= gsl::narrow<size_t>(jointParams.size()),
        "{} vs {}",
        parameterIndex,
        jointParams.size());
    if (res(parameterIndex) < data.limits[0])
      res(parameterIndex) = data.limits[0];
    if (res(parameterIndex) > data.limits[1])
      res(parameterIndex) = data.limits[1];
  }
  return res;
}

ParameterLimits getPoseConstraintParameterLimits(
    const std::string& name,
    const ParameterTransform& pt,
    const float weight) {
  ParameterLimits res;

  if (pt.poseConstraints.count(name) > 0) {
    for (const auto& pc : pt.poseConstraints.at(name).parameterIdValue) {
      ParameterLimit p;
      p.type = MinMax;
      p.data.minMax.parameterIndex = pc.first;
      p.data.minMax.limits = Vector2f::Constant(pc.second);
      p.weight = weight;
      res.emplace_back(p);
    }
  }

  return res;
}

LimitData::LimitData() {
  std::fill_n(rawData, sizeof(rawData), 0);
}

LimitData::LimitData(const LimitData& rhs) {
  std::copy_n(rhs.rawData, sizeof(rawData), rawData);
}

LimitData& LimitData::operator=(const LimitData& rhs) {
  if (&rhs != this)
    std::copy_n(rhs.rawData, sizeof(rawData), rawData);
  return *this;
}

bool LimitData::operator==(const LimitData& limitData) const {
  return (std::memcmp(this, &limitData, sizeof(LimitData)) == 0);
}

bool isInRange(const LimitLinear& limit, float value) {
  // Default initialized limits have all zeros in their members due to the std::fill_n above
  // but we want in this case to have the limit apply across the entire range of values:
  if (limit.rangeMin == 0 && limit.rangeMax == 0) {
    return true;
  }

  return value >= limit.rangeMin && value < limit.rangeMax;
}

bool isInRange(const LimitLinearJoint& limit, float value) {
  // Default initialized limits have all zeros in their members due to the std::fill_n above
  // but we want in this case to have the limit apply across the entire range of values:
  if (limit.rangeMin == 0 && limit.rangeMax == 0) {
    return true;
  }

  return value >= limit.rangeMin && value < limit.rangeMax;
}

} // namespace momentum
