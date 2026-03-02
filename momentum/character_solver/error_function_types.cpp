/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/error_function_types.h"

namespace momentum {

std::string_view toString(VertexConstraintType type) {
  switch (type) {
    case VertexConstraintType::Position:
      return "Position";
    case VertexConstraintType::Plane:
      return "Plane";
    case VertexConstraintType::Normal:
      return "Normal";
    case VertexConstraintType::SymmetricNormal:
      return "SymmetricNormal";
  }
  return "Unknown";
}

} // namespace momentum
