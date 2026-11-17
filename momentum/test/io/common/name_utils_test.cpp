/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/io/common/name_utils.h>

#include <gtest/gtest.h>

#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace momentum;

TEST(NameUtilsTest, MakeUniqueNameKeepsUnusedName) {
  const std::unordered_map<std::string, int> taken;
  EXPECT_EQ(makeUniqueName("body", taken), "body");
}

TEST(NameUtilsTest, MakeUniqueNameAppendsSequentialSuffixes) {
  std::unordered_map<std::string, int> taken{{"body", 0}};
  EXPECT_EQ(makeUniqueName("body", taken), "body_1");

  taken.emplace("body_1", 0);
  EXPECT_EQ(makeUniqueName("body", taken), "body_2");
}

TEST(NameUtilsTest, MakeUniqueNameSkipsSuffixesTakenByExplicitNames) {
  const std::set<std::string> taken{"body", "body_1"};
  EXPECT_EQ(makeUniqueName("body", taken), "body_2");
  EXPECT_EQ(makeUniqueName("body_1", taken), "body_1_1");
}

TEST(NameUtilsTest, RegistryDedupesAndKeepsEntriesSeparate) {
  ExportedNameRegistry<int> registry;
  const std::string first = registry.makeExportedName("body");
  registry.insert(first, "body", 1);
  const std::string second = registry.makeExportedName("body");
  registry.insert(second, "body", 2);

  EXPECT_EQ(first, "body");
  EXPECT_EQ(second, "body_1");
  EXPECT_EQ(registry.size(), 2);
  EXPECT_EQ(*registry.resolve(first, "body"), 1);
  EXPECT_EQ(*registry.resolve(second, "body"), 2);
}

TEST(NameUtilsTest, RegistryEmptyNameResolvesToOwnNameElseFirstEntryOfSourceName) {
  ExportedNameRegistry<int> registry;
  // An overridden requested name ("controller") resolves from its source name ("prop") ...
  registry.insert(registry.makeExportedName("controller"), "prop", 1);
  ASSERT_NE(registry.resolve("", "prop"), nullptr);
  EXPECT_EQ(*registry.resolve("", "prop"), 1);
  // ... until an entry is exported under the source name itself.
  registry.insert(registry.makeExportedName("prop"), "prop", 2);
  EXPECT_EQ(*registry.resolve("", "prop"), 2);
  EXPECT_EQ(registry.resolve("", "unknown"), nullptr);

  // So an unnamed character stays reachable with the empty name it is exported under.
  registry.insert(registry.makeExportedName("a"), "", 3);
  registry.insert(registry.makeExportedName(""), "", 4);
  EXPECT_EQ(*registry.resolve("", ""), 4);
  EXPECT_EQ(*registry.resolve("a", ""), 3);
}

TEST(NameUtilsTest, RegistryUnknownExplicitNameThrows) {
  ExportedNameRegistry<int> registry;
  registry.insert(registry.makeExportedName("body"), "body", 0);
  EXPECT_THROW((void)registry.resolve("missing", "body"), std::runtime_error);
}

TEST(NameUtilsTest, RegistryReservedNameIsTakenButSelectsNothing) {
  ExportedNameRegistry<int> registry;
  registry.reserve(registry.makeExportedName("mug"));
  EXPECT_EQ(registry.makeExportedName("mug"), "mug_1");
  EXPECT_EQ(registry.find("mug"), nullptr);
  EXPECT_EQ(registry.resolve("", "mug"), nullptr);
  EXPECT_EQ(registry.size(), 0);
}
