/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/test/io/io_helpers.h"

#include <Windows.h>
#include <process.h>

#include <array>

namespace momentum {

std::int64_t GetPID() {
  return _getpid();
}

std::optional<std::string> GetEnvVar(std::string_view name) {
  std::string envvar_name{name};
  std::array<char, 32767> buff; // max size on windows including null character
  buff[0] = '\0';
  ::GetEnvironmentVariableA(envvar_name.c_str(), &buff[0], static_cast<DWORD>(buff.size()));
  if (buff[0] == '\0') { // std::optional will report false
    return {};
  }
  return std::string{buff.data()}; // std::optional will report true
}

bool SetEnvVar(std::string_view name, std::string_view value) {
  std::string envvar_name{name};
  std::string envvar_value{value};
  // return true on success
  return ::SetEnvironmentVariableA(envvar_name.c_str(), envvar_value.c_str());
}

filesystem::path temporaryDirectory() {
  // Check for TMPDIR or DISK_TEMP environment variables

  if (auto temp_dir = GetEnvVar("TMPDIR")) {
    return *temp_dir;
  }

  if (auto temp_dir = GetEnvVar("DISK_TEMP")) {
    return *temp_dir;
  }

  char tmpPath[MAX_PATH + 1];
  GetTempPathA(MAX_PATH, tmpPath);
  return tmpPath;
}

} // namespace momentum
