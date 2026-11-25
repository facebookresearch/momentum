/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/io/common/stable_sort_order.h>

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace momentum;

TEST(StableSortOrder, ItemsWithTheSameKeyKeepTheirOrder) {
  // Enough items with each key that the sort does not fall back to an insertion sort, which keeps
  // the order of equal items anyway.
  constexpr size_t kNumItems = 200;
  std::vector<size_t> keys;
  keys.reserve(kNumItems);
  for (size_t i = 0; i < kNumItems; ++i) {
    keys.push_back((i * 7) % 3);
  }
  std::vector<size_t> expected;
  for (const size_t key : {0, 1, 2}) {
    for (size_t i = 0; i < kNumItems; ++i) {
      if (keys[i] == key) {
        expected.push_back(i);
      }
    }
  }

  EXPECT_EQ(stableSortOrder(keys), expected);
}

TEST(StableSortOrder, ApplyOrderPutsTheItemsInTheGivenOrder) {
  std::vector<std::string> items = {"c0", "a1", "c2", "b3", "a4"};
  applyOrder(items, std::vector<size_t>{1, 4, 3, 0, 2});

  EXPECT_EQ(items, (std::vector<std::string>{"a1", "a4", "b3", "c0", "c2"}));
}
