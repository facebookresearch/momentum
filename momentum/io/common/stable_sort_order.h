/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace momentum {

/// The order that sorts items by their keys, keeping the given order of items with the same key.
///
/// The FBX and glTF loaders sort their locators and collision shapes with this function, so that
/// one copy of the sort code serves every item type.
///
/// @param keys The sort key of each item.
/// @return The indices of the items, in sorted order.
[[nodiscard]] std::vector<size_t> stableSortOrder(std::span<const size_t> keys);

/// Reorders @p items so that position i gets the item that was at index order[i].
///
/// @param items The items to reorder.
/// @param order A permutation of the indices of @p items, such as one from stableSortOrder().
template <typename Items>
void applyOrder(Items& items, std::span<const size_t> order) {
  Items reordered;
  reordered.reserve(items.size());
  for (const size_t i : order) {
    reordered.push_back(std::move(items[i]));
  }
  items = std::move(reordered);
}

} // namespace momentum
