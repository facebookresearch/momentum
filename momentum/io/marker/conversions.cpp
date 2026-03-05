/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/marker/conversions.h"

#include "momentum/common/log.h"
#include "momentum/math/coordinate_system.h"

namespace momentum {

namespace {

[[maybe_unused]] [[nodiscard]] Unit toUnit(const std::string& unitStr) {
  if (unitStr == "m" || unitStr == "M") {
    return Unit::M;
  } else if (unitStr == "dm" || unitStr == "DM") {
    return Unit::DM;
  } else if (unitStr == "cm" || unitStr == "CM") {
    return Unit::CM;
  } else if (unitStr == "mm" || unitStr == "MM") {
    return Unit::MM;
  } else {
    return Unit::Unknown;
  }
}

[[maybe_unused]] [[nodiscard]] std::string_view toString(const Unit& unit) {
  switch (unit) {
    case Unit::M:
      return "m";
    case Unit::DM:
      return "dm";
    case Unit::CM:
      return "cm";
    case Unit::MM:
      return "mm";
    case Unit::Unknown:
    default:
      return "unknown";
  }
}

// Maps the legacy Unit enum to LengthUnit for use with the coordinate system utility.
[[nodiscard]] LengthUnit toLengthUnit(Unit unit) {
  switch (unit) {
    case Unit::M:
      return LengthUnit::Meter;
    case Unit::DM:
      return LengthUnit::Decimeter;
    case Unit::CM:
      return LengthUnit::Centimeter;
    case Unit::MM:
      return LengthUnit::Millimeter;
    case Unit::Unknown:
    default:
      return LengthUnit::Centimeter; // Fallback: treat as cm
  }
}

} // namespace

template <typename T>
Vector3<T> toMomentumVector3(const Vector3<T>& vec, UpVector up, Unit unit) {
  if (unit == Unit::Unknown) {
    MT_LOGE(
        "{}: Unknown unit '{}' found in the file. Use centimeters instead.",
        __func__,
        toString(unit));
  }

  // Use the coordinate system utility for unit conversion
  const LengthUnit srcUnit = toLengthUnit(unit);
  const T s = scaleFactor<T>(
      CoordinateSystem{UpAxis::Y, Handedness::Right, srcUnit},
      CoordinateSystem{UpAxis::Y, Handedness::Right, LengthUnit::Centimeter});
  Vector3<T> vec_in_cm = vec * s;

  // Preserve the existing axis remapping exactly as-is.
  // The marker I/O converts to Z-up internally (Z is identity).
  // TODO: Migrate to use changeVector() once marker convention is unified to Y-up.
  switch (up) {
    case UpVector::X:
      return {vec_in_cm.y(), vec_in_cm.z(), vec_in_cm.x()};
    case UpVector::Y:
      return {vec_in_cm.x(), vec_in_cm.z(), -vec_in_cm.y()};
    case UpVector::Z:
      return vec_in_cm;
    case UpVector::YNeg:
      return {vec_in_cm.x(), -vec_in_cm.z(), vec_in_cm.y()};
    default:
      return vec_in_cm;
  }
}

template <typename T>
Vector3<T> toMomentumVector3(const Vector3<T>& vec, UpVector up, const std::string& unitStr) {
  return toMomentumVector3(vec, up, toUnit(unitStr));
}

template Vector3<float> toMomentumVector3(const Vector3<float>& vec, UpVector up, Unit unit);
template Vector3<double> toMomentumVector3(const Vector3<double>& vec, UpVector up, Unit unit);

template Vector3<float>
toMomentumVector3(const Vector3<float>& vec, UpVector up, const std::string& unitStr);
template Vector3<double>
toMomentumVector3(const Vector3<double>& vec, UpVector up, const std::string& unitStr);

} // namespace momentum
