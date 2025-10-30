/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/fwd.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/math/mppca.h>

#include <span>

namespace momentum {

template <typename T>
class PosePriorErrorFunctionT : public SkeletonErrorFunctionT<T> {
 public:
  PosePriorErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      std::shared_ptr<const MppcaT<T>> pp);
  PosePriorErrorFunctionT(const Character& character, std::shared_ptr<const MppcaT<T>> pp);

  void setPosePrior(std::shared_ptr<const MppcaT<T>> pp);

  [[nodiscard]] double getError(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState) final;

  double getGradient(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::VectorX<T>> gradient) final;

  [[nodiscard]] double logProbability(const ModelParametersT<T>& params) const;

  double getJacobian(
      const ModelParametersT<T>& params,
      const SkeletonStateT<T>& state,
      const MeshStateT<T>& meshState,
      Eigen::Ref<Eigen::MatrixX<T>> jacobian,
      Eigen::Ref<Eigen::VectorX<T>> residual,
      int& usedRows) final;

  [[nodiscard]] size_t getJacobianSize() const final;

  [[nodiscard]] Eigen::VectorX<T> getMeanShape(const ModelParametersT<T>& params) const;

 protected:
  void loadInternal();
  void getBestFitMode(
      const ModelParametersT<T>& params,
      size_t& bestIdx,
      Eigen::VectorX<T>& bestDiff,
      T& bestR,
      T& minDist) const;

  std::vector<size_t> ppMap_;
  std::vector<size_t> invMap_;

  // The pose prior: note that once passed in you must not modify it
  // outside this class.  The easiest way to guarantee this is to use
  // the standard loadMppca() functions which already returns a const-
  // valued shared_ptr.
  std::shared_ptr<const MppcaT<T>> posePrior_;

  // weights for the error functions
  static constexpr T kPosePriorWeight = 1e-3;

 private:
  // Is used for optimization purpose in getJacobian to allocate memory once
  Eigen::MatrixX<T> gradientL_;
};

} // namespace momentum
