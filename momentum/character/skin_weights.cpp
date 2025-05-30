/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/skin_weights.h>

#include <momentum/common/checks.h>

namespace momentum {

void SkinWeights::set(
    const std::vector<std::vector<size_t>>& ind,
    const std::vector<std::vector<float>>& wgt) {
  MT_CHECK(ind.size() == wgt.size(), "{} is not {}", ind.size(), wgt.size());
  index = IndexMatrix::Zero(ind.size(), kMaxSkinJoints);
  weight = WeightMatrix::Zero(ind.size(), kMaxSkinJoints);

  for (size_t i = 0; i < ind.size(); i++) {
    for (size_t j = 0; j < std::min(ind[i].size(), size_t(kMaxSkinJoints)); j++) {
      index(i, j) = gsl::narrow_cast<uint32_t>(ind[i][j]);
      weight(i, j) = wgt[i][j];
    }
  }
}

bool SkinWeights::operator==(const SkinWeights& skinWeights) const {
  return index.isApprox(skinWeights.index) && weight.isApprox(skinWeights.weight);
}

} // namespace momentum
