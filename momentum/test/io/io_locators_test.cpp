/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/character.h>
#include <momentum/io/skeleton/locator_io.h>
#include <momentum/test/character/character_helpers.h>
#include <momentum/test/io/io_helpers.h>

#include <gtest/gtest.h>

using namespace momentum;

namespace {

LocatorList createTestLocators() {
  LocatorList locators;

  // Locator 1: Basic locator with default values
  {
    Locator loc;
    loc.name = "locator_default";
    loc.parent = 0;
    loc.offset = Vector3f(1.0f, 2.0f, 3.0f);
    loc.limitOrigin = Vector3f(1.0f, 2.0f, 3.0f);
    loc.weight = 1.0f;
    loc.limitWeight = Vector3f(0.5f, 0.6f, 0.7f);
    loc.locked = Vector3i(0, 0, 0);
    loc.attachedToSkin = false;
    loc.skinOffset = 0.0f;
    locators.push_back(loc);
  }

  // Locator 2: Locator with attachedToSkin=true and non-zero skinOffset
  {
    Locator loc;
    loc.name = "locator_skin_attached";
    loc.parent = 1;
    loc.offset = Vector3f(-1.5f, 0.5f, 2.5f);
    loc.limitOrigin = Vector3f(-1.5f, 0.5f, 2.5f);
    loc.weight = 0.8f;
    loc.limitWeight = Vector3f(0.3f, 0.4f, 0.5f);
    loc.locked = Vector3i(1, 0, 1);
    loc.attachedToSkin = true;
    loc.skinOffset = 0.05f;
    locators.push_back(loc);
  }

  // Locator 3: Locator with various non-default values
  {
    Locator loc;
    loc.name = "locator_complex";
    loc.parent = 2;
    loc.offset = Vector3f(0.0f, 1.0f, 0.0f);
    loc.limitOrigin = Vector3f(0.1f, 0.9f, 0.0f);
    loc.weight = 2.5f;
    loc.limitWeight = Vector3f(1.0f, 0.0f, 0.8f);
    loc.locked = Vector3i(1, 1, 0);
    loc.attachedToSkin = false;
    loc.skinOffset = 0.0f;
    locators.push_back(loc);
  }

  // Locator 4: Locator with attachedToSkin=true but zero skinOffset
  {
    Locator loc;
    loc.name = "locator_skin_zero_offset";
    loc.parent = 1;
    loc.offset = Vector3f(3.0f, 4.0f, 5.0f);
    loc.limitOrigin = Vector3f(3.0f, 4.0f, 5.0f);
    loc.weight = 1.5f;
    loc.limitWeight = Vector3f(0.0f, 0.0f, 0.0f);
    loc.locked = Vector3i(0, 1, 0);
    loc.attachedToSkin = true;
    loc.skinOffset = 0.0f;
    locators.push_back(loc);
  }

  // Locator 5: Locator with all fields at various values
  {
    Locator loc;
    loc.name = "locator_full_test";
    loc.parent = 0;
    loc.offset = Vector3f(-2.0f, -3.0f, -4.0f);
    loc.limitOrigin = Vector3f(-2.1f, -2.9f, -4.0f);
    loc.weight = 0.5f;
    loc.limitWeight = Vector3f(0.2f, 0.3f, 0.0f);
    loc.locked = Vector3i(1, 1, 1);
    loc.attachedToSkin = true;
    loc.skinOffset = 0.15f;
    locators.push_back(loc);
  }

  return locators;
}

void validateLocatorsSame(const LocatorList& locators1, const LocatorList& locators2) {
  ASSERT_EQ(locators1.size(), locators2.size())
      << "Locator lists have different sizes: " << locators1.size() << " vs " << locators2.size();

  for (size_t i = 0; i < locators1.size(); ++i) {
    const auto& l1 = locators1[i];
    const auto& l2 = locators2[i];

    EXPECT_EQ(l1.name, l2.name) << "Locator " << i << " name mismatch";
    EXPECT_EQ(l1.parent, l2.parent) << "Locator " << i << " (" << l1.name << ") parent mismatch";

    EXPECT_NEAR(l1.offset.x(), l2.offset.x(), 1e-5f)
        << "Locator " << i << " (" << l1.name << ") offset.x mismatch";
    EXPECT_NEAR(l1.offset.y(), l2.offset.y(), 1e-5f)
        << "Locator " << i << " (" << l1.name << ") offset.y mismatch";
    EXPECT_NEAR(l1.offset.z(), l2.offset.z(), 1e-5f)
        << "Locator " << i << " (" << l1.name << ") offset.z mismatch";

    // limitOrigin is not saved/loaded, it defaults to offset after loading
    EXPECT_NEAR(l2.limitOrigin.x(), l2.offset.x(), 1e-5f)
        << "Locator " << i << " (" << l2.name << ") limitOrigin should equal offset after loading";
    EXPECT_NEAR(l2.limitOrigin.y(), l2.offset.y(), 1e-5f)
        << "Locator " << i << " (" << l2.name << ") limitOrigin should equal offset after loading";
    EXPECT_NEAR(l2.limitOrigin.z(), l2.offset.z(), 1e-5f)
        << "Locator " << i << " (" << l2.name << ") limitOrigin should equal offset after loading";

    EXPECT_NEAR(l1.weight, l2.weight, 1e-5f)
        << "Locator " << i << " (" << l1.name << ") weight mismatch";

    EXPECT_NEAR(l1.limitWeight.x(), l2.limitWeight.x(), 1e-5f)
        << "Locator " << i << " (" << l1.name << ") limitWeight.x mismatch";
    EXPECT_NEAR(l1.limitWeight.y(), l2.limitWeight.y(), 1e-5f)
        << "Locator " << i << " (" << l1.name << ") limitWeight.y mismatch";
    EXPECT_NEAR(l1.limitWeight.z(), l2.limitWeight.z(), 1e-5f)
        << "Locator " << i << " (" << l1.name << ") limitWeight.z mismatch";

    EXPECT_EQ(l1.locked.x(), l2.locked.x())
        << "Locator " << i << " (" << l1.name << ") locked.x mismatch";
    EXPECT_EQ(l1.locked.y(), l2.locked.y())
        << "Locator " << i << " (" << l1.name << ") locked.y mismatch";
    EXPECT_EQ(l1.locked.z(), l2.locked.z())
        << "Locator " << i << " (" << l1.name << ") locked.z mismatch";

    // Test the new fields
    EXPECT_EQ(l1.attachedToSkin, l2.attachedToSkin)
        << "Locator " << i << " (" << l1.name << ") attachedToSkin mismatch";
    EXPECT_NEAR(l1.skinOffset, l2.skinOffset, 1e-5f)
        << "Locator " << i << " (" << l1.name << ") skinOffset mismatch";
  }
}

} // namespace

TEST(LocatorIOTest, RoundTrip_Local) {
  // Create test character with skeleton
  const Character character = createTestCharacter();
  const LocatorList originalLocators = createTestLocators();

  // Save locators in local space
  TemporaryFile tempFile = temporaryFile("", "locators");
  const auto& path = tempFile.path();

  ASSERT_NO_THROW(saveLocators(path, originalLocators, character.skeleton, LocatorSpace::Local));
  EXPECT_TRUE(filesystem::exists(path)) << "Locator file not created at " << path;

  // Load locators back
  LocatorList loadedLocators;
  ASSERT_NO_THROW(
      loadedLocators = loadLocators(path, character.skeleton, character.parameterTransform));

  // Validate they match
  validateLocatorsSame(originalLocators, loadedLocators);
}

TEST(LocatorIOTest, RoundTrip_Global) {
  // Create test character with skeleton
  const Character character = createTestCharacter();
  const LocatorList originalLocators = createTestLocators();

  // Save locators in global space
  TemporaryFile tempFile = temporaryFile("", "locators");
  const auto& path = tempFile.path();

  ASSERT_NO_THROW(saveLocators(path, originalLocators, character.skeleton, LocatorSpace::Global));
  EXPECT_TRUE(filesystem::exists(path)) << "Locator file not created at " << path;

  // Load locators back (should be converted back to local space)
  LocatorList loadedLocators;
  ASSERT_NO_THROW(
      loadedLocators = loadLocators(path, character.skeleton, character.parameterTransform));

  // Validate they match (offsets might differ due to global->local conversion,
  // but other fields should be preserved)
  ASSERT_EQ(originalLocators.size(), loadedLocators.size());
  for (size_t i = 0; i < originalLocators.size(); ++i) {
    const auto& l1 = originalLocators[i];
    const auto& l2 = loadedLocators[i];

    EXPECT_EQ(l1.name, l2.name);
    EXPECT_EQ(l1.parent, l2.parent);
    EXPECT_NEAR(l1.weight, l2.weight, 1e-5f);
    EXPECT_EQ(l1.attachedToSkin, l2.attachedToSkin);
    EXPECT_NEAR(l1.skinOffset, l2.skinOffset, 1e-5f);
  }
}

TEST(LocatorIOTest, EmptyLocatorList) {
  const Character character = createTestCharacter();
  LocatorList emptyLocators;

  // Save empty locator list
  TemporaryFile tempFile = temporaryFile("", "locators");
  const auto& path = tempFile.path();

  ASSERT_NO_THROW(saveLocators(path, emptyLocators, character.skeleton, LocatorSpace::Local));
  EXPECT_TRUE(filesystem::exists(path)) << "Locator file not created at " << path;

  // Load empty locator list
  LocatorList loadedLocators;
  ASSERT_NO_THROW(
      loadedLocators = loadLocators(path, character.skeleton, character.parameterTransform));

  EXPECT_EQ(loadedLocators.size(), 0);
}
