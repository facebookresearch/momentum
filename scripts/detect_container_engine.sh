#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Detect and export the container engine for cibuildwheel
#
# This script detects available container engines (docker/podman) and exports
# CIBW_CONTAINER_ENGINE if not already set.
#
# Priority:
# 1. Respect CIBW_CONTAINER_ENGINE if already set
# 2. Otherwise, detect docker first, then podman
#
# Usage:
#   source scripts/detect_container_engine.sh
#   # Then run cibuildwheel

if [ -z "$CIBW_CONTAINER_ENGINE" ]; then
    if command -v docker >/dev/null 2>&1; then
        export CIBW_CONTAINER_ENGINE=docker
    elif command -v podman >/dev/null 2>&1; then
        export CIBW_CONTAINER_ENGINE=podman
    else
        echo "Warning: Neither docker nor podman found. cibuildwheel may fail."
    fi
fi

echo "Container engine: ${CIBW_CONTAINER_ENGINE:-none detected}"
