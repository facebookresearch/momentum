#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for variant-specific PyPI metadata generation."""

import subprocess
import sys
import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.generate_pyproject import core_version_bounds


class GeneratePyprojectTest(unittest.TestCase):
    def test_core_version_bounds(self) -> None:
        self.assertEqual(core_version_bounds("v0.1.114"), ("0.1.114", "0.1.115"))
        with self.assertRaisesRegex(ValueError, "three-part release version"):
            core_version_bounds("0.1")

    def test_generated_variants_are_layered(self) -> None:
        with TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_pyproject.py",
                    "--output-dir",
                    temp_dir,
                    "--core-version",
                    "0.1.114",
                ],
                check=True,
            )

            generated = {
                variant: tomllib.loads(
                    (
                        Path(temp_dir) / f"pyproject-pypi-{variant}-py312.toml"
                    ).read_text()
                )
                for variant in ("core", "cpu", "gpu")
            }

        core = generated["core"]
        self.assertEqual(core["project"]["dependencies"], ["numpy>=1.20.0"])
        self.assertEqual(
            core["tool"]["scikit-build"]["install"]["components"],
            ["pymomentum_core"],
        )

        for variant in ("cpu", "gpu"):
            dependencies = generated[variant]["project"]["dependencies"]
            self.assertIn("pymomentum-core>=0.1.114,<0.1.115", dependencies)
            self.assertEqual(
                generated[variant]["tool"]["scikit-build"]["install"]["components"],
                ["pymomentum_torch"],
            )
        self.assertTrue(
            any(
                value.startswith("torch>=")
                for value in generated["cpu"]["project"]["dependencies"]
            )
        )
        self.assertFalse(
            any(
                value.startswith("torch")
                for value in generated["gpu"]["project"]["dependencies"]
            )
        )


if __name__ == "__main__":
    unittest.main()
