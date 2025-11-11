/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef MOMENTUM_WITH_FBX_SDK

#include <fbxsdk.h>
#include <fbxsdk/core/fbxstream.h>
#include <span>

#include <string_view>

#define FBX_VERSION_GE(major, minor, patch)                               \
  ((FBXSDK_VERSION_MAJOR > (major)) ||                                    \
   (FBXSDK_VERSION_MAJOR == (major) && FBXSDK_VERSION_MINOR > (minor)) || \
   (FBXSDK_VERSION_MAJOR == (major) && FBXSDK_VERSION_MINOR == (minor) && \
    FBXSDK_VERSION_POINT >= (patch)))

namespace momentum {

// Simplest FbxStream to read file from a string_view memory buffer
class FbxMemoryStream : public FbxStream {
 public:
  FbxMemoryStream(std::span<const std::byte> buffer, int pReaderId);
  ~FbxMemoryStream() override;
  EState GetState() override;
  bool Open(void* pStreamData) override;
  bool Close() override;
  bool Flush() override;
#if FBX_VERSION_GE(2020, 3, 2)
  size_t Write(const void* buffer, FbxUInt64 count) override;
  size_t Read(void* buffer, FbxUInt64 count) const override;
#else
  int Write(const void* buffer, int count) override;
  int Read(void* buffer, int count) const override;
#endif
  int GetReaderID() const override;
  int GetWriterID() const override;
  void Seek(const FbxInt64& pOffset, const FbxFile::ESeekPos& pSeekPos) override;
#if FBX_VERSION_GE(2020, 3, 2)
  FbxInt64 GetPosition() const override;
  void SetPosition(FbxInt64 pPosition) override;
#else
  long GetPosition() const override;
  void SetPosition(long pPosition) override;
#endif
  int GetError() const override;
  void ClearError() override;

 private:
  std::span<const std::byte> buffer_;
  long length_;
#if FBX_VERSION_GE(2020, 3, 2)
  mutable FbxInt64 position_{0};
#else
  mutable long position_;
#endif
  EState state_;
  int readerId_;
  mutable int errorCode_;
};

} // namespace momentum

#endif // MOMENTUM_WITH_FBX_SDK
