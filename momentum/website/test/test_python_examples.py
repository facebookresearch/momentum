# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import os
import re
import shutil
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np


_PYTHON_BLOCK_PATTERN: str = r"^```python\s*\n(.*?)^```\s*$"


def _required_path(variable: str) -> Path:
    return Path(os.environ[variable])


def _find_unique(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {pattern!r} under {root}, found {len(matches)}"
        )
    return matches[0]


def _run_python_blocks(
    page: Path,
    working_directory: Path,
    environment: Mapping[str, str] | None = None,
    initial_namespace: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    blocks = re.findall(
        _PYTHON_BLOCK_PATTERN,
        page.read_text(),
        re.MULTILINE | re.DOTALL,
    )
    if not blocks:
        raise RuntimeError(f"No Python blocks found in {page}")

    namespace: dict[str, Any] = {"__name__": "__documentation_example__"}
    namespace.update(initial_namespace or {})
    source = "\n\n".join(blocks)
    with (
        contextlib.chdir(working_directory),
        mock.patch.dict(os.environ, environment or {}, clear=False),
    ):
        exec(compile(source, str(page), "exec"), namespace)
    return namespace


class TestPythonDocumentationExamples(unittest.TestCase):
    def test_character_basics_and_core_concepts(self) -> None:
        docs_root = _required_path("PYMOMENTUM_DOCS_ROOT")
        sample_assets = _required_path("PYMOMENTUM_DOCS_SAMPLE_ASSETS")
        with tempfile.TemporaryDirectory() as temp_directory:
            workdir = Path(temp_directory)
            for filename in ("character.fbx", "character.model"):
                shutil.copyfile(
                    _find_unique(sample_assets, filename), workdir / filename
                )

            namespace = _run_python_blocks(
                docs_root / "02_examples/01_python_basics.md",
                workdir,
            )
            character = namespace["character"]
            core_namespace = _run_python_blocks(
                docs_root / "01_user_guide/02_core_concepts.md",
                workdir,
                initial_namespace={"character": character},
            )

        self.assertGreater(character.parameter_transform.size, 0)
        reference_parameters = np.zeros_like(namespace["model_parameters"])
        reference_state = namespace["pym_geometry"].model_parameters_to_skeleton_state(
            character, reference_parameters
        )
        self.assertFalse(np.allclose(namespace["skeleton_state"], reference_state))
        self.assertEqual(
            namespace["skeleton_state"].shape,
            (character.skeleton.size, 8),
        )
        reference_vertices = character.skin_points(reference_state)
        self.assertFalse(np.allclose(namespace["posed_vertices"], reference_vertices))
        self.assertEqual(namespace["posed_vertices"].shape[1], 3)

        self.assertEqual(
            core_namespace["joint_parameters"].shape,
            (character.skeleton.size, 7),
        )
        self.assertEqual(
            core_namespace["skeleton_state"].shape,
            (character.skeleton.size, 8),
        )
        self.assertEqual(core_namespace["posed_vertices"].shape[1], 3)

    def test_mhr_inverse_kinematics(self) -> None:
        docs_root = _required_path("PYMOMENTUM_DOCS_ROOT")
        mhr_assets = _required_path("PYMOMENTUM_DOCS_MHR_ASSETS")
        with tempfile.TemporaryDirectory() as temp_directory:
            workdir = Path(temp_directory)
            asset_dir = workdir / "assets"
            asset_dir.mkdir()
            shutil.copyfile(
                _find_unique(mhr_assets, "*lod1*.fbx"),
                asset_dir / "lod1.fbx",
            )
            shutil.copyfile(
                _find_unique(mhr_assets, "compact_v6_1.model"),
                asset_dir / "compact_v6_1.model",
            )

            namespace = _run_python_blocks(
                docs_root / "02_examples/03_inverse_kinematics.md",
                workdir,
                {"MHR_ASSET_DIR": str(asset_dir)},
            )

        character = namespace["character"]
        self.assertGreater(character.parameter_transform.size, 0)
        wrist_joint = namespace["wrist_joint"]
        target_error = np.linalg.norm(
            namespace["solved_positions"][wrist_joint] - namespace["target"]
        )
        self.assertLess(target_error, 1e-4)

        root_joint = character.skeleton.joint_index("root")
        np.testing.assert_allclose(
            namespace["solved_state"][root_joint],
            namespace["initial_state"][root_joint],
            atol=1e-6,
        )
