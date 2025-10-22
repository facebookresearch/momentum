/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/pose_prior_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
PosePriorErrorFunctionT<T>::PosePriorErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    std::shared_ptr<const MppcaT<T>> pp)
    : SkeletonErrorFunctionT<T>(skel, pt), posePrior_(std::move(pp)) {
  if (posePrior_) {
    loadInternal();
  }
}

template <typename T>
PosePriorErrorFunctionT<T>::PosePriorErrorFunctionT(
    const Character& character,
    std::shared_ptr<const MppcaT<T>> pp)
    : PosePriorErrorFunctionT<T>(character.skeleton, character.parameterTransform, std::move(pp)) {}

template <typename T>
void PosePriorErrorFunctionT<T>::setPosePrior(std::shared_ptr<const MppcaT<T>> pp) {
  posePrior_ = std::move(pp);

  loadInternal();
}

template <typename T>
void PosePriorErrorFunctionT<T>::loadInternal() {
  // create map from posePrior to this->parameterTransform_ and the other way round
  ppMap_.resize(posePrior_->names.size(), kInvalidIndex);
  invMap_.resize(this->parameterTransform_.numAllModelParameters(), kInvalidIndex);

  for (size_t i = 0; i < posePrior_->names.size(); i++) {
    for (Eigen::Index j = 0; j < this->parameterTransform_.numAllModelParameters(); j++) {
      if (posePrior_->names[i] == this->parameterTransform_.name[j]) {
        ppMap_[i] = j;
        invMap_[j] = i;
      }
    }
  }
}

template <typename T>
Eigen::VectorX<T> mapParameters(const ModelParametersT<T>& in, const std::vector<size_t>& mp) {
  Eigen::VectorX<T> res = Eigen::VectorX<T>::Zero(mp.size());
  for (size_t i = 0; i < mp.size(); i++) {
    if (mp[i] != kInvalidIndex) {
      res[i] = in[mp[i]];
    }
  }
  return res;
}

template <typename T>
double PosePriorErrorFunctionT<T>::logProbability(const ModelParametersT<T>& params) const {
  // loop over all joints and check for smoothness
  if (invMap_.size() != gsl::narrow_cast<size_t>(params.size())) {
    return 0.0;
  }

  // NOTE: this is NOT the correct value for a full MPPCA. To simplify computation we only
  // consider the gaussian distribution which is "closest" to our data point and treat it as
  // if all other mixtures don't exist.
  size_t meanShapeIdx = 0;
  Eigen::VectorX<T> bestDiff;
  T minDist, bestR;
  getBestFitMode(params, meanShapeIdx, bestDiff, bestR, minDist);

  return -minDist;
}

template <typename T>
double PosePriorErrorFunctionT<T>::getError(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& /* meshState */) {
  // return error
  return -logProbability(params) * kPosePriorWeight * this->weight_;
}

template <typename T>
double PosePriorErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& /* meshState */,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  MT_PROFILE_FUNCTION();

  // loop over all joints and check for smoothness
  double error = 0.0;

  if (!posePrior_ || invMap_.size() != gsl::narrow_cast<size_t>(params.size())) {
    return error;
  }

  const auto& posePrior = *posePrior_;

  // NOTE: this is NOT the correct gradient for an MPPCA. To simplify computation we only
  // consider the gaussian distribution which is "closest" to our data point and treat it as
  // if all other mixtures don't exist.
  size_t meanShapeIdx = 0;
  Eigen::VectorX<T> bestDiff;
  T minDist, bestR;
  getBestFitMode(params, meanShapeIdx, bestDiff, bestR, minDist);

  if (meanShapeIdx == kInvalidIndex) {
    return 0;
  }

  error += minDist;

  // calculate the gradient for the best parameter
  for (size_t i = 0; i < posePrior.d; i++) {
    if (ppMap_[i] != kInvalidIndex) {
      gradient(ppMap_[i]) +=
          bestDiff.dot(posePrior.Cinv[meanShapeIdx].col(i)) * kPosePriorWeight * this->weight_;
    }
  }

  // return error
  return error * kPosePriorWeight * this->weight_;
}

template <typename T>
size_t PosePriorErrorFunctionT<T>::getJacobianSize() const {
  if (posePrior_) {
    return posePrior_->d;
  } else {
    return 0;
  }
}

template <typename T>
double PosePriorErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& /*state*/,
    const MeshStateT<T>& /* meshState */,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_PROFILE_FUNCTION();

  // loop over all joints and check for smoothness
  double error = 0.0;
  usedRows = 0;

  if (!posePrior_ || invMap_.size() != gsl::narrow_cast<size_t>(params.size())) {
    return error;
  }

  const auto& posePrior = *posePrior_;

  // NOTE: this is NOT the correct jacobian for an MPPCA. To simplify computation we only
  // consider the gaussian distribution which is "closest" to our data point and treat it as
  // if all other mixtures don't exist.
  size_t meanShapeIdx = 0;
  Eigen::VectorX<T> bestDiff;
  T minDist, bestR;

  getBestFitMode(params, meanShapeIdx, bestDiff, bestR, minDist);

  if (meanShapeIdx == kInvalidIndex) {
    return T(0);
  }

  error += minDist * kPosePriorWeight * this->weight_;

  const T wgt = std::sqrt(T(0.5) * kPosePriorWeight * this->weight_);

  // calculate the gradient for the best parameter
  // NOTE: The dimensions are not forced to match,
  // we thus write in the upper rows on purpose
  this->gradientL_.noalias() = posePrior.L[meanShapeIdx].topRows(posePrior.d) * wgt;
  residual.head(posePrior.d).noalias() = this->gradientL_ * bestDiff;

  // re-order the jacobian columns on the fly
  jacobian.topRows(posePrior.d).setConstant(T(0));
  for (size_t i = 0; i < invMap_.size(); i++) {
    if (invMap_[i] != kInvalidIndex) {
      jacobian.col(i).head(posePrior.d) = this->gradientL_.col(invMap_[i]);
    }
  }

  usedRows = gsl::narrow_cast<int>(posePrior.d);

  // return error
  return error;
}

template <typename T>
Eigen::VectorX<T> PosePriorErrorFunctionT<T>::getMeanShape(
    const ModelParametersT<T>& params) const {
  size_t meanShapeIdx = 0;
  Eigen::VectorX<T> bestDiff;
  T minDist, bestR;
  getBestFitMode(params, meanShapeIdx, bestDiff, bestR, minDist);
  if (!posePrior_ || meanShapeIdx == kInvalidIndex) {
    return Eigen::VectorX<T>::Zero(this->parameterTransform_.numAllModelParameters());
  }

  return mapParameters<T>(posePrior_->mu.row(meanShapeIdx).transpose(), invMap_);
}

template <typename T>
void PosePriorErrorFunctionT<T>::getBestFitMode(
    const ModelParametersT<T>& params,
    size_t& bestIdx,
    Eigen::VectorX<T>& bestDiff,
    T& bestR,
    T& minDist) const {
  minDist = std::numeric_limits<T>::max();
  bestR = -std::numeric_limits<T>::max();
  bestIdx = kInvalidIndex;

  if (!posePrior_) {
    return;
  }

  const auto& posePrior = *posePrior_;

  const Eigen::VectorX<T> subParams = mapParameters(params, ppMap_);

  for (size_t c = 0; c < posePrior.p; c++) {
    const Eigen::VectorX<T> diff = subParams - posePrior.mu.row(c).transpose();
    const T squareDist = T(0.5) * diff.transpose() * posePrior.Cinv[c] * diff;
    // Rpre is the constant part of R that does not depend on the parameters,
    // so it's precalculated
    const T R = posePrior.Rpre(c) - squareDist;
    if (R > bestR) {
      bestR = R;
      bestDiff = diff;
      minDist = squareDist;
      bestIdx = c;
    }
  }
}

template class PosePriorErrorFunctionT<float>;
template class PosePriorErrorFunctionT<double>;

} // namespace momentum
