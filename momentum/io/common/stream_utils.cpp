/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/common/stream_utils.h"

#include "momentum/common/exception.h"

namespace momentum {

spanstreambuf::spanstreambuf(std::span<const std::byte> buffer) {
  if (buffer.empty()) {
    return;
  }

  // This is a bit awkward: the stream buffer won't change the data
  // but the interface still requires char* to be used
  char* data = reinterpret_cast<char*>(const_cast<std::byte*>(buffer.data()));
  this->setg(data, data, data + buffer.size());
}

spanstreambuf::~spanstreambuf() = default;

std::streamsize spanstreambuf::xsputn(
    const spanstreambuf::char_type* /* s */,
    std::streamsize /* n */) {
  MT_THROW("spanstreambuf is not writable.");
}

ispanstream::ispanstream(std::span<const std::byte> buffer) : std::istream(&sbuf_), sbuf_(buffer) {
  // Empty
}

ispanstream::~ispanstream() = default;

std::istream& GetLineCrossPlatform(std::istream& is, std::string& line) {
  line.clear();
  while (is.good()) {
    const int c = is.get();
    if (!is.good()) {
      break;
    }

    if (c == '\n') {
      break;
    }

    if (c == '\r') {
      const auto next_c = is.peek();
      if (!is.good()) {
        break;
      }

      if (next_c == '\n') {
        is.get();
      }

      break;
    }

    line.push_back(gsl::narrow_cast<char>(c));
  }

  return is;
}

} // namespace momentum
