/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/common/stream_utils.h"

#include <gtest/gtest.h>

using namespace momentum;

TEST(StreamUtilsTest, SpanStreamTest_ReadFromSpanStreamBuffer) {
  std::string data = "Test data";
  std::span<const std::byte> buffer(reinterpret_cast<const std::byte*>(data.data()), data.size());
  spanstreambuf sbuf(buffer);

  std::istream inputStream(&sbuf);
  std::string output;
  inputStream >> output;

  ASSERT_EQ(output, "Test");
}

TEST(StreamUtilsTest, SpanStreamTest_ReadFromISpanStream) {
  std::string data = "Test data";
  std::span<const std::byte> buffer(reinterpret_cast<const std::byte*>(data.data()), data.size());
  ispanstream inputStream(buffer);

  std::string output;
  inputStream >> output;

  ASSERT_EQ(output, "Test");
}

TEST(StreamUtilsTest, SpanStreamTest_ReadMultipleWordsFromISpanStream) {
  std::string data = "Hello World!";
  std::span<const std::byte> buffer(reinterpret_cast<const std::byte*>(data.data()), data.size());
  ispanstream inputStream(buffer);

  std::string word1, word2;
  inputStream >> word1 >> word2;

  ASSERT_EQ(word1, "Hello");
  ASSERT_EQ(word2, "World!");
}

TEST(StreamUtilsTest, SpanStreamTest_ReadFromEmptyISpanStream) {
  std::span<const std::byte> buffer;
  ispanstream inputStream(buffer);

  std::string output;
  inputStream >> output;

  ASSERT_TRUE(output.empty());
}

TEST(StreamUtilsTest, GetLineCrossPlatform_HandlesUnixLineEndings) {
  std::istringstream input("Line 1\nLine 2\nLine 3");
  std::string line;

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 1");

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 2");

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 3");
}

TEST(StreamUtilsTest, GetLineCrossPlatform_HandlesWindowsLineEndings) {
  std::istringstream input("Line 1\r\nLine 2\r\nLine 3");
  std::string line;

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 1");

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 2");

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 3");
}

TEST(StreamUtilsTest, GetLineCrossPlatform_HandlesMacOSLineEndings) {
  std::istringstream input("Line 1\rLine 2\rLine 3");
  std::string line;

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 1");

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 2");

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 3");
}

TEST(StreamUtilsTest, GetLineCrossPlatform_HandlesEmptyInput) {
  std::istringstream input("");
  std::string line;

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "");

  ASSERT_TRUE(input.eof());
}

TEST(StreamUtilsTest, GetLineCrossPlatform_HandlesMixedLineEndings) {
  std::istringstream input("Line 1\nLine 2\r\nLine 3\rLine 4");
  std::string line;

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 1");

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 2");

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 3");

  GetLineCrossPlatform(input, line);
  ASSERT_EQ(line, "Line 4");
}
