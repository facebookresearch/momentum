/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/diff_ik/fully_differentiable_pose_prior_error_function.h"

#include "momentum/common/checks.h"
#include "momentum/common/exception.h"
#include "momentum/math/mppca.h"

namespace momentum {

namespace {

template <typename T>
Eigen::VectorX<T> mapParameters(
    Eigen::Ref<const Eigen::VectorX<T>> in,
    const std::vector<size_t>& mp) {
  Eigen::VectorX<T> res = Eigen::VectorX<T>::Zero(mp.size());
  for (size_t i = 0; i < mp.size(); i++) {
    if (mp[i] != kInvalidIndex) {
      res[i] = in[mp[i]];
    }
  }
  return res;
}

} // namespace

template <typename T>
FullyDifferentiablePosePriorErrorFunctionT<T>::FullyDifferentiablePosePriorErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt,
    std::vector<std::string> names)
    : PosePriorErrorFunctionT<T>(skel, pt, std::make_shared<MppcaT<T>>()),
      names_(std::move(names)) {}

template <typename T>
FullyDifferentiablePosePriorErrorFunctionT<T>::~FullyDifferentiablePosePriorErrorFunctionT() =
    default;

template <typename T>
void FullyDifferentiablePosePriorErrorFunctionT<T>::setPosePrior(
    const VectorX<T>& pi,
    const MatrixX<T>& mmu,
    const std::vector<MatrixX<T>>& W,
    const VectorX<T>& sigma) {
  MT_THROW_IF(W.empty(), "Empty W matrix in setPosePrior().");

  for (const auto& W_i : W) {
    MT_THROW_IF(
        W_i.rows() != W[0].rows() || W_i.cols() != W[0].cols(),
        "Expected matching dimensions for all matrices in W in setPosePrior.");
  }

  pi_ = std::move(pi);
  mmu_ = std::move(mmu);
  W_ = std::move(W);
  sigma_ = std::move(sigma);

  auto mppca = std::make_shared<MppcaT<T>>();
  mppca->set(pi_, mmu_, W_, sigma_.array().square());
  MT_THROW_IF(
      mppca->d != names_.size(),
      "Mismatch in pose prior; expected number of names to match dimension.");
  mppca->names = names_;
  PosePriorErrorFunctionT<T>::setPosePrior(mppca);
}

template <typename T>
std::vector<std::string> FullyDifferentiablePosePriorErrorFunctionT<T>::inputs() const {
  return {kPi, kMu, kW, kSigma};
}

template <typename T>
Eigen::Index FullyDifferentiablePosePriorErrorFunctionT<T>::getInputSize(
    const std::string& name) const {
  if (name == kPi) {
    return this->pi_.size();
  } else if (name == kMu) {
    return this->mmu_.size();
  } else if (name == kW) {
    Eigen::Index result = 0;
    for (const auto& W_i : W_) {
      result += W_i.size();
    }
    return result;
  } else if (name == kSigma) {
    return sigma_.size();
  } else {
    MT_THROW(
        "Unknown input to FullyDifferentiablePosePriorErrorFunctionT<T>::getInputSize: {}", name);
  }
}

template <typename T>
void FullyDifferentiablePosePriorErrorFunctionT<T>::getInputImp(
    const std::string& name,
    Eigen::Ref<Eigen::VectorX<T>> value) const {
  MT_THROW_IF(!this->posePrior_, "No pose prior loaded, can't set values.");

  const auto& posePrior = *this->posePrior_;

  const Eigen::Index p = posePrior.p;
  const Eigen::Index d = posePrior.d;

  if (name == kPi) {
    MT_CHECK(this->pi_.size() == p);
    value = this->pi_.template cast<T>();
  } else if (name == kMu) {
    // mmu stores the means in the rows of mmu:
    MT_CHECK(this->mmu_.rows() == p);
    MT_CHECK(this->mmu_.cols() == d);

    for (Eigen::Index iMix = 0; iMix < p; ++iMix) {
      value.segment(iMix * d, d) = this->mmu_.row(iMix).template cast<T>();
    }
  } else if (name == kW) {
    MT_CHECK(this->W_.size() == p);
    Eigen::Index offset = 0;
    for (Eigen::Index iMix = 0; iMix < p; ++iMix) {
      const auto& W_i = this->W_[iMix];
      MT_CHECK(W_i.rows() == d);
      for (Eigen::Index j = 0; j < W_i.cols(); ++j) {
        value.segment(offset, d) = W_i.col(j).template cast<T>();
        offset += d;
      }
    }
  } else if (name == kSigma) {
    MT_CHECK(this->sigma_.size() == p);
    value = this->sigma_.template cast<T>();
  } else {
    MT_THROW(
        "Unknown input to FullyDifferentiablePosePriorErrorFunctionT<T>::getInputSize: {}", name);
  }
}

template <typename T>
void FullyDifferentiablePosePriorErrorFunctionT<T>::setInputImp(
    const std::string& name,
    Eigen::Ref<const Eigen::VectorX<T>> value) {
  MT_THROW_IF(!this->posePrior_, "No pose prior loaded, can't set values.");

  const auto& posePrior = *this->posePrior_;

  const Eigen::Index p = posePrior.p;
  const Eigen::Index d = posePrior.d;

  if (name == kPi) {
    MT_CHECK(this->pi_.size() == p);
    this->pi_ = value;
  } else if (name == kMu) {
    // mmu stores the means in the rows of mmu:
    MT_CHECK(this->mmu_.rows() == p);
    MT_CHECK(this->mmu_.cols() == d);

    for (Eigen::Index iMix = 0; iMix < p; ++iMix) {
      this->mmu_.row(iMix) = value.segment(iMix * d, d);
    }
  } else if (name == kW) {
    MT_CHECK(this->W_.size() == p);
    Eigen::Index offset = 0;
    for (Eigen::Index iMix = 0; iMix < p; ++iMix) {
      auto& W_i = this->W_[iMix];
      MT_CHECK(W_i.rows() == d);
      for (Eigen::Index j = 0; j < W_i.cols(); ++j) {
        W_i.col(j) = value.segment(offset, d).transpose();
        offset += d;
      }
    }
  } else if (name == kSigma) {
    MT_CHECK(this->sigma_.size() == p);
    this->sigma_ = value.cwiseMax(0);
  } else {
    MT_THROW(
        "Unknown input to FullyDifferentiablePosePriorErrorFunctionT<T>::getInputSize: {}", name);
  }

  // Update the Mppca model:
  auto mppca = std::make_shared<MppcaT<T>>();
  mppca->set(pi_, mmu_, W_, sigma_.array().square());
  mppca->names = names_;
  PosePriorErrorFunctionT<T>::setPosePrior(mppca);
}

template <typename T>
Eigen::VectorX<T> FullyDifferentiablePosePriorErrorFunctionT<T>::d_gradient_d_input_dot(
    const std::string& inputName,
    const ModelParametersT<T>& params,
    const SkeletonStateT<T>& /* state */,
    Eigen::Ref<const Eigen::VectorX<T>> inputVec) {
  MT_THROW_IF(!this->posePrior_, "No pose prior loaded, can't set values.");

  const auto& posePrior = *this->posePrior_;

  const Eigen::Index d = posePrior.d;

  Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(getInputSize(inputName));

  size_t modeIdx;
  Eigen::VectorX<T> bestDiff;
  T minDist, bestR;
  PosePriorErrorFunctionT<T>::getBestFitMode(params, modeIdx, bestDiff, bestR, minDist);

  if (modeIdx == kInvalidIndex) {
    return result;
  }

  const Eigen::VectorX<T> subParams = mapParameters<T>(params.v, this->ppMap_);
  const Eigen::VectorX<T> subInput = mapParameters<T>(inputVec, this->ppMap_);

  if (inputName == kMu) {
    // Gradient is just weight * C^{-1} * (theta - mu) so the derivative is pretty simple:
    const auto weight = this->kPosePriorWeight * this->weight_;
    result.segment(modeIdx * d, d) = -weight * posePrior.Cinv[modeIdx] * subInput;
  } else if (inputName == kSigma) {
    // We need to know the derivative of the C^{-1} matrix wrt sigma.
    // Computing this requires knowing the derivative of the matrix inverse.
    // We have the following identity:  http://matrixcookbook.com
    //    d(Ainv)/dsigma = -(Ainv) dA/dSigma (Ainv)
    // We know that A = sigma^2 * I + ..., so
    //    dA/dSigma = 2 * sigma * I
    // We also have that the derivative of A_ii wrt sigma is just
    // Therefore
    //    d(Ainv)/dsigma =  -(Ainv) * (2*sigma*I) * Ainv
    //                   = -2 * sigma * Ainv * Ainv
    //
    // Now remember that the final thing we want to compute is actually v^T * dGrad/dSigma.
    // and the gradient is:
    //  grad = weight * Ainv * (theta - mu)
    // Our standard chain rule expansion of dLoss/dSigma is:
    //   v^T dGrad/dSigma
    //       = v^T (weight * dAinv/dSigma * (theta - mu))
    //       = v^T (weight * -2 * sigma * Ainv * Ainv * (theta - mu))
    //       = -2 * weight * sigma * v^T * [ Ainv * Ainv * (theta - mu) ]
    // and we can reorder the last quantity so we do matrix-vector instead of
    // matrix-matrix multiplication.
    const auto weight = this->kPosePriorWeight * this->weight_;
    const T sigma = sigma_[modeIdx];
    const Eigen::VectorX<T> diff = subParams - posePrior.mu.row(modeIdx).transpose();
    result(modeIdx) = (-T(2) * sigma * weight) *
        (posePrior.Cinv[modeIdx] * subInput).dot(posePrior.Cinv[modeIdx] * diff);
  } else if (inputName == kW) {
    if (W_.empty()) {
      return result;
    }

    const auto weight = this->kPosePriorWeight * this->weight_;

    // We need to know the derivative of the C^{-1} matrix wrt W.
    // Computing this requires knowing the derivative of the matrix inverse.
    // We have the following identity:  http://matrixcookbook.com
    //    d(Ainv)/dW_jk = -(Ainv) dA/dW_jk (Ainv)
    // Recall that
    //    A = sigma^2*I + W * W^T
    //   dA/dW_jk = W * dW/dW_ik^T + dW/dW_ik * W^T
    // where these dW/dW_ik matrices have a single 1 in a matrix of zeros
    // that plucks out the kth column from W:
    //                 [                ]   [       |       ]
    //      dA/dW_jk = [ --- W_:k^T --- ] + [      W_:k     ]
    //                 [                ]   [       |       ]
    //
    // Recall that the full quantity we want is v^T * dGrad/dW_jk
    // Since grad = weight * Ainv * (theta - mu), we can compute
    //   v^T * dGrad/dW_jk
    //      = v^T * weight * dAinv/dW_jk * (theta - mu)
    //      = weight * v^T * -(Ainv) dA/dW_jk (Ainv) * (theta - mu)
    // If we rearrange the parentheses and use the fact that A is symmetric, this becomes
    //      = -weight * (Ainv * v)^T dA/dW_jk (Ainv * (theta - mu))
    //                  ----------             -------------------
    // The quantities on the ends can be computed once and reused, and only the dA/dW_jk
    // changes.  But this matrix has the simple sparsity pattern reported above, so multiplying
    // by it produces a vector where one entry is a dot product with the appropriate column of
    // W and the other elements are zero.
    const Eigen::MatrixX<T>& W_i = W_[modeIdx];
    MT_CHECK(W_i.rows() == d);
    const Eigen::Index W_cols = W_i.cols();

    const Eigen::Index result_offset = W_cols * d * modeIdx;

    const Eigen::MatrixX<T>& Cinv = posePrior.Cinv[modeIdx];

    const Eigen::VectorX<T> dLoss_times_Cinv = Cinv * subInput;
    const Eigen::VectorX<T> diff = subParams - posePrior.mu.row(modeIdx).transpose();
    const Eigen::VectorX<T> Cinv_times_diff = Cinv * diff;

    for (Eigen::Index j = 0; j < d; ++j) {
      for (Eigen::Index k = 0; k < W_cols; ++k) {
        const T dGrad_dInput_dot_v = -weight *
            (dLoss_times_Cinv.dot(W_i.col(k)) * Cinv_times_diff[j] +
             W_i.col(k).dot(Cinv_times_diff) * dLoss_times_Cinv[j]);
        result(result_offset + k * d + j) = dGrad_dInput_dot_v;
      }
    }
  } else if (inputName == kPi) {
    // The derivatives of the mixture weights are, for all practical purposes, zero;
    // since they are constant wrt any particular mixture they only turn up in the
    // relative weighting of the gradients and since the constant terms in the
    // probability terms that are used for that weighting almost always get swamped
    // by the exponential falloff as we move away from the mean we can safely ignore
    // them.
  }

  return result;
}

template class FullyDifferentiablePosePriorErrorFunctionT<float>;
template class FullyDifferentiablePosePriorErrorFunctionT<double>;

} // namespace momentum
