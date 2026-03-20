/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/parameter_limits.h>
#include <momentum/character/parameter_transform.h>

#include <nlohmann/json.hpp>
#include <Eigen/Core>

namespace momentum {

// Converts Eigen matrix or vector types to nlohmann::json
//
// Note: We intentionally avoid defining nlohmann::adl_serializers for type conversion between Eigen
// types and nlohmann::json to prevent multiple definitions of the same type in different projects
// that are compiled in a single translation unit. Instead, we recommend using explicit conversion
// between the types using the toJson() and fromJson() functions provided in this header.
template <typename Derived>
void toJson(const Eigen::MatrixBase<Derived>& mat, nlohmann::json& j) {
  // always save new format that's numpy compatible
  // this works for both dynamic and statically sized matrices
  j = nlohmann::json::array();
  for (int col_i = 0; col_i < mat.cols(); ++col_i) {
    nlohmann::json jcol = nlohmann::json::array();
    for (int row_i = 0; row_i < mat.rows(); ++row_i) {
      jcol.push_back(mat(row_i, col_i));
    }
    j.push_back(jcol);
  }
}

// Converts nlohmann::json to Eigen matrix or vector types
template <typename T>
T fromJson(const nlohmann::json& j) {
  T result;

  // partially dynamic matrix
  if constexpr (T::RowsAtCompileTime == Eigen::Dynamic || T::ColsAtCompileTime == Eigen::Dynamic) {
    // dynamic size matrix
    // reads dimensions from the json
    const auto numCols = j.size();
    if (numCols > 0) {
      const auto numRows = j.at(0).size();
      result.resize(numRows, numCols);

      std::vector<typename T::Scalar> coef;
      for (int col_i = 0; col_i < numCols; ++col_i) {
        coef = j[col_i];
        std::copy(coef.begin(), coef.end(), result.col(col_i).begin());
      }
    }
  }
  // fixed size matrix or vector
  else {
    const int Rows = T::RowsAtCompileTime;
    const int Cols = T::ColsAtCompileTime;

    // vector stored as array
    if (Rows == 1 && Cols != -1 && j.size() == Cols &&
        j[0].type() != nlohmann::json::value_t::array) {
      std::vector<typename T::Scalar> coef = j;
      std::copy(coef.begin(), coef.end(), result.row(0).begin());
    } else if (
        Cols == 1 && Rows != -1 && j.size() == Rows &&
        j[0].type() != nlohmann::json::value_t::array) {
      std::vector<typename T::Scalar> coef = j;
      std::copy(coef.begin(), coef.end(), result.col(0).begin());
    }
    // fixed size matrix
    else {
      const auto numCols = j.size();
      if (numCols > 0) {
        for (int col_i = 0; col_i < Cols; ++col_i) {
          std::vector<typename T::Scalar> coef = j[col_i];
          std::copy(coef.begin(), coef.end(), result.col(col_i).begin());
        }
      }
    }
  }

  return result;
}

// to json conversion functions
void parameterLimitsToJson(const Character& character, nlohmann::json& j);
void parameterSetsToJson(const Character& character, nlohmann::json& j);
void poseConstraintsToJson(const Character& character, nlohmann::json& j);
void parameterTransformToJson(const Character& character, nlohmann::json& j);

// from json conversion functions
ParameterLimits parameterLimitsFromJson(const Character& character, const nlohmann::json& j);
ParameterSets parameterSetsFromJson(const Character& character, const nlohmann::json& j);
PoseConstraints poseConstraintsFromJson(const Character& character, const nlohmann::json& j);
ParameterTransform parameterTransformFromJson(const Character& character, const nlohmann::json& j);

} // namespace momentum
