/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/skin_weights.h"

#include "momentum/common/checks.h"
#include "momentum/common/log.h"

#include <gsl/util>

#include <algorithm>
#include <cmath>
#include <utility>

namespace momentum {

void SkinWeights::set(
    const std::vector<std::vector<size_t>>& ind,
    const std::vector<std::vector<float>>& wgt) {
  MT_CHECK(ind.size() == wgt.size(), "{} is not {}", ind.size(), wgt.size());
  index = IndexMatrix::Zero(ind.size(), kMaxSkinJoints);
  weight = WeightMatrix::Zero(ind.size(), kMaxSkinJoints);

  using Influence = std::pair<size_t, float>;
  std::vector<Influence> influences;
  size_t invalidWeightCount = 0;
  size_t truncatedInfluenceCount = 0;
  for (size_t i = 0; i < ind.size(); ++i) {
    MT_CHECK(
        ind[i].size() == wgt[i].size(),
        "Skin weight row {} has {} indices but {} weights",
        i,
        ind[i].size(),
        wgt[i].size());

    influences.clear();
    influences.reserve(ind[i].size());
    for (size_t j = 0; j < ind[i].size(); ++j) {
      if (!std::isfinite(wgt[i][j]) || wgt[i][j] < 0.0f) {
        ++invalidWeightCount;
        continue;
      }
      if (wgt[i][j] > 0.0f) {
        influences.emplace_back(ind[i][j], wgt[i][j]);
      }
    }

    std::ranges::stable_sort(influences, [](const Influence& lhs, const Influence& rhs) {
      return lhs.second > rhs.second;
    });

    if (influences.size() > kMaxSkinJoints) {
      truncatedInfluenceCount += influences.size() - kMaxSkinJoints;
    }
    const size_t influenceCount = std::min(influences.size(), size_t{kMaxSkinJoints});
    for (size_t j = 0; j < influenceCount; ++j) {
      const auto& influence = influences.at(j);
      index(i, j) = gsl::narrow_cast<uint32_t>(influence.first);
      weight(i, j) = influence.second;
    }
  }

  if (invalidWeightCount > 0) {
    MT_LOGW("Discarded {} non-finite or negative skin weights", invalidWeightCount);
  }
  if (truncatedInfluenceCount > 0) {
    MT_LOGW(
        "Discarded {} skin weight influences beyond the per-vertex limit of {}",
        truncatedInfluenceCount,
        kMaxSkinJoints);
  }
}

bool SkinWeights::operator==(const SkinWeights& skinWeights) const {
  return index.isApprox(skinWeights.index) && weight.isApprox(skinWeights.weight);
}

} // namespace momentum
