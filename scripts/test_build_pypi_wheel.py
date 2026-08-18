#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the PyPI wheel build helper."""

import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.build_pypi_wheel import torch_cmake_prefix_path


class TorchCmakePrefixPathTest(unittest.TestCase):
    def test_finds_cmake_directory_without_importing_torch(self) -> None:
        with TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "torch"
            cmake_dir = package_dir / "share" / "cmake"
            cmake_dir.mkdir(parents=True)

            torch_spec = ModuleSpec("torch", loader=None, is_package=True)
            torch_spec.submodule_search_locations = [str(package_dir)]
            with patch("scripts.build_pypi_wheel.find_spec", return_value=torch_spec):
                self.assertEqual(torch_cmake_prefix_path(), str(cmake_dir))

    def test_rejects_installation_without_cmake_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "torch"
            package_dir.mkdir()

            torch_spec = ModuleSpec("torch", loader=None, is_package=True)
            torch_spec.submodule_search_locations = [str(package_dir)]
            with patch("scripts.build_pypi_wheel.find_spec", return_value=torch_spec):
                with self.assertRaisesRegex(
                    RuntimeError, "Torch CMake package directory not found"
                ):
                    torch_cmake_prefix_path()


if __name__ == "__main__":
    unittest.main()
