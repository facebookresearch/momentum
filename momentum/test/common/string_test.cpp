/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/string.h"

#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

using namespace momentum;

// Tests for trim with std::string
TEST(StringTest, TrimString) {
  // Test trimming whitespace from both ends
  std::string s1 = "  hello  ";
  EXPECT_EQ(trim(s1), "hello");

  // Test trimming tabs
  std::string s2 = "\thello\t";
  EXPECT_EQ(trim(s2), "hello");

  // Test trimming mixed whitespace
  std::string s3 = " \t hello \t ";
  EXPECT_EQ(trim(s3), "hello");

  // Test with no whitespace to trim
  std::string s4 = "hello";
  EXPECT_EQ(trim(s4), "hello");

  // Test with empty string
  std::string s5 = "";
  EXPECT_EQ(trim(s5), "");

  // Test with only whitespace
  std::string s6 = "   ";
  EXPECT_EQ(trim(s6), "");

  std::string s7 = "\t\t\t";
  EXPECT_EQ(trim(s7), "");

  // Test with whitespace in the middle
  std::string s8 = "  hello world  ";
  EXPECT_EQ(trim(s8), "hello world");

  // Test with custom whitespace characters
  std::string s9 = "__hello__";
  EXPECT_EQ(trim(s9, "_"), "hello");

  std::string s10 = "_*hello*_";
  EXPECT_EQ(trim(s10, "_*"), "hello");
}

// Tests for trim with std::string_view
TEST(StringTest, TrimStringView) {
  // Test trimming whitespace from both ends
  std::string_view sv = "  hello  ";
  EXPECT_EQ(trim(sv), "hello");

  // Test trimming tabs
  sv = "\thello\t";
  EXPECT_EQ(trim(sv), "hello");

  // Test trimming mixed whitespace
  sv = " \t hello \t ";
  EXPECT_EQ(trim(sv), "hello");

  // Test with no whitespace to trim
  sv = "hello";
  EXPECT_EQ(trim(sv), "hello");

  // Test with empty string
  sv = "";
  EXPECT_EQ(trim(sv), "");

  // Test with only whitespace
  sv = "   ";
  EXPECT_EQ(trim(sv), "");
  sv = "\t\t\t";
  EXPECT_EQ(trim(sv), "");

  // Test with whitespace in the middle
  sv = "  hello world  ";
  EXPECT_EQ(trim(sv), "hello world");

  // Test with custom whitespace characters
  sv = "__hello__";
  EXPECT_EQ(trim(sv, "_"), "hello");
  sv = "_*hello*_";
  EXPECT_EQ(trim(sv, "_*"), "hello");
}

// Tests for trim with C-style strings
TEST(StringTest, TrimCString) {
  // Test trimming whitespace from both ends
  EXPECT_EQ(trim("  hello  "), "hello");

  // Test trimming tabs
  EXPECT_EQ(trim("\thello\t"), "hello");

  // Test trimming mixed whitespace
  EXPECT_EQ(trim(" \t hello \t "), "hello");

  // Test with no whitespace to trim
  EXPECT_EQ(trim("hello"), "hello");

  // Test with empty string
  EXPECT_EQ(trim(""), "");

  // Test with only whitespace
  EXPECT_EQ(trim("   "), "");
  EXPECT_EQ(trim("\t\t\t"), "");

  // Test with whitespace in the middle
  EXPECT_EQ(trim("  hello world  "), "hello world");

  // Test with custom whitespace characters
  EXPECT_EQ(trim("__hello__", "_"), "hello");
  EXPECT_EQ(trim("_*hello*_", "_*"), "hello");
}

// Tests for tokenize with std::string
TEST(StringTest, TokenizeString) {
  // Test basic tokenization with default delimiters
  std::string s1 = "hello world";
  std::vector<std::string> tokens = tokenize(s1);
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "world");

  // Test with multiple spaces
  std::string s2 = "hello   world";
  tokens = tokenize(s2);
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "world");

  // Test with tabs and newlines
  std::string s3 = "hello\tworld\ntest";
  tokens = tokenize(s3);
  ASSERT_EQ(tokens.size(), 3);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "world");
  EXPECT_EQ(tokens[2], "test");

  // Test with custom delimiters
  std::string s4 = "hello,world;test";
  std::string delim1 = ",;";
  tokens = tokenize(s4, delim1);
  ASSERT_EQ(tokens.size(), 3);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "world");
  EXPECT_EQ(tokens[2], "test");

  // Test with empty string
  std::string s5 = "";
  tokens = tokenize(s5);
  EXPECT_TRUE(tokens.empty());

  // Test with only delimiters
  std::string s6 = "   ";
  tokens = tokenize(s6);
  EXPECT_TRUE(tokens.empty());

  // Test with trim=false
  std::string s7 = "hello   world";
  std::string delim2 = " ";
  tokens = tokenize(s7, delim2, false);
  ASSERT_EQ(tokens.size(), 4);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "");
  EXPECT_EQ(tokens[2], "");
  EXPECT_EQ(tokens[3], "world");

  // Test with delimiters at the beginning and end
  std::string s8 = ",hello,world,";
  std::string delim3 = ",";
  tokens = tokenize(s8, delim3);
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "world");

  // Test with delimiters at the beginning and end, trim=false
  tokens = tokenize(s8, delim3, false);
  ASSERT_EQ(tokens.size(), 4);
  EXPECT_EQ(tokens[0], "");
  EXPECT_EQ(tokens[1], "hello");
  EXPECT_EQ(tokens[2], "world");
  EXPECT_EQ(tokens[3], "");
}

// Tests for tokenize with std::string_view
TEST(StringTest, TokenizeStringView) {
  // Test basic tokenization with default delimiters
  std::string_view sv = "hello world";
  std::vector<std::string_view> tokens = tokenize(sv);
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "world");

  // Test with multiple spaces
  sv = "hello   world";
  tokens = tokenize(sv);
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "world");

  // Test with tabs and newlines
  sv = "hello\tworld\ntest";
  tokens = tokenize(sv);
  ASSERT_EQ(tokens.size(), 3);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "world");
  EXPECT_EQ(tokens[2], "test");

  // Test with custom delimiters
  sv = "hello,world;test";
  tokens = tokenize(sv, ",;");
  ASSERT_EQ(tokens.size(), 3);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "world");
  EXPECT_EQ(tokens[2], "test");

  // Test with empty string
  sv = "";
  tokens = tokenize(sv);
  EXPECT_TRUE(tokens.empty());

  // Test with only delimiters
  sv = "   ";
  tokens = tokenize(sv);
  EXPECT_TRUE(tokens.empty());

  // Test with trim=false
  sv = "hello   world";
  tokens = tokenize(sv, " ", false);
  ASSERT_EQ(tokens.size(), 4);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "");
  EXPECT_EQ(tokens[2], "");
  EXPECT_EQ(tokens[3], "world");

  // Test with delimiters at the beginning and end
  sv = ",hello,world,";
  tokens = tokenize(sv, ",");
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "world");

  // Test with delimiters at the beginning and end, trim=false
  sv = ",hello,world,";
  tokens = tokenize(sv, ",", false);
  ASSERT_EQ(tokens.size(), 4);
  EXPECT_EQ(tokens[0], "");
  EXPECT_EQ(tokens[1], "hello");
  EXPECT_EQ(tokens[2], "world");
  EXPECT_EQ(tokens[3], "");
}

// Test edge cases and special scenarios
TEST(StringTest, EdgeCases) {
  // Test trim with string containing only the characters to be trimmed
  std::string s1 = "___";
  EXPECT_EQ(trim(s1, "_"), "");

  // Test trim with string containing none of the characters to be trimmed
  std::string s2 = "hello";
  EXPECT_EQ(trim(s2, "_"), "hello");

  // Test tokenize with delimiter not in the string
  std::string s3 = "hello world";
  std::string delim1 = ",";
  std::vector<std::string> tokens = tokenize(s3, delim1);
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], "hello world");

  // Test tokenize with empty delimiter string
  std::string delim2 = "";
  tokens = tokenize(s3, delim2);
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], "hello world");

  // Test tokenize with string containing only delimiters, trim=false
  std::string s4 = ",,,,";
  std::string delim3 = ",";
  tokens = tokenize(s4, delim3, false);
  ASSERT_EQ(tokens.size(), 5);
  for (const auto& token : tokens) {
    EXPECT_TRUE(token.empty());
  }

  // Test tokenize with string containing only delimiters, trim=true
  tokens = tokenize(s4, delim3, true);
  EXPECT_TRUE(tokens.empty());
}

// Test Unicode handling
TEST(StringTest, UnicodeHandling) {
  // Test trim with Unicode characters
  std::string s1 = "  hello世界  ";
  EXPECT_EQ(trim(s1), "hello世界");

  // Test tokenize with Unicode characters
  std::string s2 = "hello 世界 test";
  std::vector<std::string> tokens = tokenize(s2);
  ASSERT_EQ(tokens.size(), 3);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "世界");
  EXPECT_EQ(tokens[2], "test");

  // Test tokenize with Unicode delimiters
  std::string s3 = "hello世界test";
  std::string delim = "世界";
  tokens = tokenize(s3, delim);
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0], "hello");
  EXPECT_EQ(tokens[1], "test");
}
