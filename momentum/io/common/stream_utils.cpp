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

spanstreambuf::pos_type
spanstreambuf::seekoff(off_type off, std::ios_base::seekdir way, std::ios_base::openmode which) {
  if ((which & std::ios_base::in) == 0) {
    return {off_type(-1)};
  }
  char* const beg = eback();
  char* const cur = gptr();
  char* const end = egptr();
  if (beg == nullptr) {
    // Empty buffer -- only seeking to position 0 is valid.
    return (off == 0) ? pos_type(0) : pos_type(off_type(-1));
  }
  off_type base = 0;
  // libstdc++ adds the implementation-detail sentinel `_S_ios_seekdir_end` to
  // the seekdir enum; the `default:` branch handles it (and any future
  // additions) safely.
  switch (way) { // NOLINT(clang-diagnostic-switch-enum)
    case std::ios_base::beg:
      base = 0;
      break;
    case std::ios_base::cur:
      base = cur - beg;
      break;
    case std::ios_base::end:
      base = end - beg;
      break;
    default:
      return {off_type(-1)};
  }
  const off_type target = base + off;
  if (target < 0 || target > (end - beg)) {
    return {off_type(-1)};
  }
  setg(beg, beg + target, end);
  return {target};
}

spanstreambuf::pos_type spanstreambuf::seekpos(pos_type pos, std::ios_base::openmode which) {
  return seekoff(off_type(pos), std::ios_base::beg, which);
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
