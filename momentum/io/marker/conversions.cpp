/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/marker/conversions.h"

#include "momentum/common/log.h"

namespace momentum {

namespace internal {

// function prototypes are needed to compile on some platforms
Unit toUnit(const std::string& unitStr);

[[nodiscard]] std::string_view toString(const Unit& unit);

Unit toUnit(const std::string& unitStr) {
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

[[nodiscard]] std::string_view toString(const Unit& unit) {
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

} // namespace internal

template <typename T>
Vector3<T> toMomentumVector3(const Vector3<T>& vec, UpVector up, Unit unit) {
  // Convert the input vector's units to centimeters
  Vector3<T> vec_in_cm;
  switch (unit) {
    case Unit::M: // Meters
      vec_in_cm = vec * (T)100;
      break;
    case Unit::DM: // Decimeter
      vec_in_cm = vec * (T)10;
      break;
    case Unit::CM: // Centimeters
      vec_in_cm = vec;
      break;
    case Unit::MM: // Millimeters
      vec_in_cm = vec * (T)0.1;
      break;
    case Unit::Unknown: // Unknown units, default to centimeters
      MT_LOGE(
          "{}: Unknown unit '{}' found in the file. Use centimeters instead.",
          __func__,
          internal::toString(unit));
      vec_in_cm = vec;
      break;
    default:
      vec_in_cm = vec;
      break;
  }

  // Converting from the given UpVector coordinate system to the target coordinate system
  Vector3<T> vec_in_momentum;
  switch (up) {
    case UpVector::X: // X-up
      vec_in_momentum = Vector3<T>(vec_in_cm.y(), vec_in_cm.z(), vec_in_cm.x());
      break;
    case UpVector::Y: // Y-up
      vec_in_momentum = Vector3<T>(vec_in_cm.x(), vec_in_cm.z(), -vec_in_cm.y());
      break;
    case UpVector::Z: // Z-up
      vec_in_momentum = vec_in_cm;
      break;
    case UpVector::YNeg: // Negative Y-up
      vec_in_momentum = Vector3<T>(vec_in_cm.x(), -vec_in_cm.z(), vec_in_cm.y());
      break;
    default:
      vec_in_momentum = vec_in_cm;
      break;
  }

  return vec_in_momentum;
}

template <typename T>
Vector3<T> toMomentumVector3(const Vector3<T>& vec, UpVector up, const std::string& unitStr) {
  return toMomentumVector3(vec, up, internal::toUnit(unitStr));
}

template Vector3<float> toMomentumVector3(const Vector3<float>& vec, UpVector up, Unit unit);
template Vector3<double> toMomentumVector3(const Vector3<double>& vec, UpVector up, Unit unit);

template Vector3<float>
toMomentumVector3(const Vector3<float>& vec, UpVector up, const std::string& unitStr);
template Vector3<double>
toMomentumVector3(const Vector3<double>& vec, UpVector up, const std::string& unitStr);

} // namespace momentum
