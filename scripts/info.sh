#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

version='1.0.0'
cb='```'

if [ ! -x "$(command -v cmake)" ]; then
  echo "Please install cmake to use this script."
  exit 1
fi

info=$(cmake --system-information)

ext() {
  grep -oE '".+"$' $1 | tr -d '"'
}

print_info() {
echo "$info" | grep -e "$1" | ext
}

result="
Momentum System Info v${version}
Commit: $(git rev-parse HEAD 2> /dev/null || echo "Not in a git repo.")
CMake Version: $(cmake --version | grep -oE '[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+')
System: $(print_info 'CMAKE_SYSTEM "')
Arch: $(print_info 'CMAKE_SYSTEM_PROCESSOR')
C++ Compiler: $(print_info 'CMAKE_CXX_COMPILER ==')
C++ Compiler Version: $(print_info 'CMAKE_CXX_COMPILER_VERSION')
C Compiler: $(print_info 'CMAKE_C_COMPILER ==')
C Compiler Version: $(print_info 'CMAKE_C_COMPILER_VERSION')
CMake Prefix Path: $(print_info '_PREFIX_PATH ')
"
if [ "$CONDA_SHLVL" == 1 ]; then
  conda="
Conda Env
<details>

$cb
$(conda list)
$cb

</details>
"
fi

all="$result  $conda"
echo "$all"

if [ -x "$(command -v xclip)" ]; then
 clip="xclip -selection c"
elif [ -x "$(command -v pbcopy)" ]; then
  clip="pbcopy"
else
  echo "\nThe results will be copied to your clipboard if xclip is installed."
fi

if [ ! -z "$clip" ]; then
  echo "$all" | $clip
  echo "Result copied to clipboard!"
fi
