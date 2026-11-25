/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/common/stable_sort_order.h"

#include <algorithm>
#include <numeric>

namespace momentum {

std::vector<size_t> stableSortOrder(std::span<const size_t> keys) {
  std::vector<size_t> order(keys.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::ranges::stable_sort(order, {}, [keys](size_t i) { return keys[i]; });
  return order;
}

} // namespace momentum
