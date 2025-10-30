/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/fbx_memory_stream.h"

namespace momentum {

FbxMemoryStream::FbxMemoryStream(std::span<const std::byte> buffer, int pReaderId)
    : buffer_(buffer),
      length_(buffer.size()),
      position_(0),
      state_(EState::eClosed),
      readerId_(pReaderId) {}

FbxMemoryStream::~FbxMemoryStream() {
  Close();
}

FbxStream::EState FbxMemoryStream::GetState() {
  return state_;
}

bool FbxMemoryStream::Open(void* /* pStreamData */) {
  position_ = 0;
  state_ = EState::eOpen;
  return true;
}

bool FbxMemoryStream::Close() {
  position_ = 0;
  state_ = EState::eClosed;
  return true;
}

bool FbxMemoryStream::Flush() {
  return true;
}

#if FBX_VERSION_GE(2020, 3, 2)
size_t FbxMemoryStream::Write(const void* /* buffer */, FbxUInt64 /* count */) {
#else
int FbxMemoryStream::Write(const void* /* buffer */, int /* count */) {
#endif
  errorCode_ = 1;
  return 0;
}

#if FBX_VERSION_GE(2020, 3, 2)
size_t FbxMemoryStream::Read(void* buffer, FbxUInt64 count) const {
#else
int FbxMemoryStream::Read(void* buffer, int count) const {
#endif
  long remaining = length_ - position_;
  if (count > remaining) {
    errorCode_ = 1;
    return 0;
  }

  memcpy(buffer, buffer_.data() + position_, count);
  position_ += count;
  return count;
}

int FbxMemoryStream::GetReaderID() const {
  return readerId_;
}

int FbxMemoryStream::GetWriterID() const {
  return -1;
}

void FbxMemoryStream::Seek(const FbxInt64& pOffset, const FbxFile::ESeekPos& pSeekPos) {
  long offset = (long)pOffset;
  switch (pSeekPos) {
    case FbxFile::ESeekPos::eCurrent:
      position_ += offset;
      break;
    case FbxFile::ESeekPos::eEnd:
      position_ = length_ - offset;
      break;
    default:
      position_ = offset;
      break;
  }
}

#if FBX_VERSION_GE(2020, 3, 2)
FbxInt64 FbxMemoryStream::GetPosition() const {
#else
long FbxMemoryStream::GetPosition() const {
#endif
  return position_;
}

#if FBX_VERSION_GE(2020, 3, 2)
void FbxMemoryStream::SetPosition(FbxInt64 pPosition) {
#else
void FbxMemoryStream::SetPosition(long pPosition) {
#endif
  position_ = pPosition;
}

int FbxMemoryStream::GetError() const {
  return errorCode_;
}

void FbxMemoryStream::ClearError() {
  errorCode_ = 0;
}

} // namespace momentum
