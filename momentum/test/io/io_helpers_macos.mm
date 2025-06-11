/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/test/io/io_helpers.h"

#include <unistd.h>
#include <cstdint>
#include <cstdlib>

#import <Foundation/Foundation.h> // for NSURL, NSProcessInfo, NSTemporaryDirectory, NSFileManager

namespace momentum {

std::int64_t GetPID() {
  return getpid();
}

std::optional<std::string> GetEnvVar(std::string_view name) {
  std::string envvar_name{name};
  if (const char* const result = getenv(envvar_name.c_str())) {
    return result;
  }
  return {};
}

bool SetEnvVar(std::string_view name, std::string_view value) {
  std::string envvar_name{name};
  std::string envvar_value{value};
  // returns false on success
  return !setenv(envvar_name.c_str(), envvar_value.c_str(), true);
}

filesystem::path temporaryDirectory() {
  @autoreleasepool {
    NSURL* dir = [[NSURL fileURLWithPath:NSTemporaryDirectory()]
        URLByAppendingPathComponent:[NSProcessInfo processInfo].globallyUniqueString
                        isDirectory:true];
    [[NSFileManager defaultManager] createDirectoryAtURL:dir
                             withIntermediateDirectories:YES
                                              attributes:nil
                                                   error:NULL];
    return [[NSFileManager defaultManager] fileSystemRepresentationWithPath:[dir path]];
  }
}

} // namespace momentum
