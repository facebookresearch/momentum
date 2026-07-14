# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test wheel imports using uv for fast, reliable testing across platforms.

Supports:
- Linux: Tests in manylinux_2_28 container via docker or podman (auto-detected)
- macOS/Windows: Tests locally using uv virtual environments
- Multiple Python versions via uv's Python management
- Optional strict CUDA checks for GPU wheels via WHEEL_TEST_REQUIRE_CUDA=1
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


WHEEL_PREFIX_BY_TYPE = {
    "core": "pymomentum_core-",
    "cpu": "pymomentum_cpu-",
    "gpu": "pymomentum_gpu-",
}
DYNAMIC_LIBRARY_ENV_VARS = (
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_FALLBACK_FRAMEWORK_PATH",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
)
CONDA_ENV_VARS = (
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CONDA_PREFIX_1",
    "CONDA_PREFIX_2",
)


def _is_path_under(path: str, parent: str) -> bool:
    try:
        return Path(path).resolve().is_relative_to(Path(parent).resolve())
    except (OSError, RuntimeError, ValueError):
        return False


def _is_pixi_or_conda_path(path: str, conda_prefixes: list[str]) -> bool:
    normalized = path.replace("\\", "/").lower()
    if "/.pixi/" in normalized or normalized.endswith("/.pixi"):
        return True
    return any(_is_path_under(path, prefix) for prefix in conda_prefixes)


def sanitize_dynamic_library_env(env: dict[str, str]) -> None:
    """Remove pixi/conda linker paths while preserving runner Python paths."""
    conda_prefixes = [value for name in CONDA_ENV_VARS if (value := env.get(name))]
    for name in DYNAMIC_LIBRARY_ENV_VARS:
        value = env.get(name)
        if not value:
            continue
        entries = [
            entry
            for entry in value.split(os.pathsep)
            if entry and not _is_pixi_or_conda_path(entry, conda_prefixes)
        ]
        if entries:
            env[name] = os.pathsep.join(entries)
        else:
            env.pop(name, None)


def normalize_wheel_type(wheel_type: str) -> str:
    """Normalize legacy and current wheel type names."""
    if wheel_type == "base":
        return "core"
    if wheel_type in {"cpu", "torch_cpu"}:
        return "cpu"
    if wheel_type in {"gpu", "torch_gpu"}:
        return "gpu"
    return wheel_type


def has_torch_extensions(wheel_type: str) -> bool:
    return wheel_type in {"cpu", "gpu"}


def has_torch_runtime(wheel_type: str) -> bool:
    return wheel_type in {"core", "cpu", "gpu"}


def get_torch_index(wheel_type: str) -> str:
    """Get the appropriate PyTorch index URL for the wheel type."""
    if wheel_type == "gpu":
        return "https://download.pytorch.org/whl/cu129"
    return "https://download.pytorch.org/whl/cpu"


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def get_import_test_script(
    wheel_type: str,
    require_cuda: bool = False,
    require_triton: bool = False,
) -> str:
    """Return the Python smoke test script for a wheel type."""
    torch_extensions = has_torch_extensions(wheel_type)
    torch_runtime = has_torch_runtime(wheel_type)
    lines = [
        "import importlib.util",
        "import os",
        "import sys",
        "import traceback",
        "",
        'if sys.platform == "win32":',
        '    os.environ.setdefault("PYTHONIOENCODING", "utf-8")',
        "    import io",
        '    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")',
        '    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")',
        "",
        'print("Testing imports...")',
        "try:",
    ]
    if torch_runtime:
        lines.extend(
            [
                "    import torch",
                '    print(f"  PyTorch {torch.__version__}, CUDA={torch.version.cuda}")',
            ]
        )
        if torch_extensions:
            lines.append(
                '    assert torch.version.cuda is not None, "pymomentum-gpu requires CUDA PyTorch"'
                if wheel_type == "gpu"
                else '    assert torch.version.cuda is None, "pymomentum-cpu should be tested with CPU PyTorch"'
            )
    lines.extend(
        [
            "    import pymomentum.geometry as geom",
            "    import pymomentum.solver2 as solver2",
            "    import pymomentum.marker_tracking as marker_tracking",
            "    import pymomentum.axel as axel",
            "    import pymomentum.camera as camera",
            "    import pymomentum.renderer as renderer",
        ]
    )
    if torch_extensions:
        lines.extend(
            [
                "    import pymomentum.diff_geometry as diff_geometry",
                "    import pymomentum.solver as solver",
            ]
        )
        if wheel_type == "gpu" and require_cuda:
            lines.extend(
                [
                    "    import pymomentum.backend.selection as backend_selection",
                    "    import pymomentum.quaternion as quaternion",
                ]
            )
    else:
        if wheel_type == "core":
            lines.extend(
                [
                    "    import pymomentum.quaternion as quaternion",
                    "    import pymomentum.skel_state as skel_state",
                    "    import pymomentum.trs as trs",
                    "    import pymomentum.torch.character as torch_character",
                ]
            )
        lines.extend(
            [
                '    assert importlib.util.find_spec("pymomentum.diff_geometry") is None, "pymomentum-core must not expose pymomentum.diff_geometry"',
                '    assert importlib.util.find_spec("pymomentum.solver") is None, "pymomentum-core must not expose pymomentum.solver"',
            ]
        )
    lines.extend(
        [
            '    print("[PASS] All imports successful")',
            "except Exception as e:",
            '    print(f"[FAIL] Import failed ({type(e).__name__}): {e}", file=sys.stderr)',
            "    traceback.print_exc()",
            "    sys.exit(1)",
            "",
        ]
    )
    if wheel_type == "gpu" and require_cuda:
        lines.extend(
            [
                'print("Testing CUDA smoke tests...")',
                "try:",
                '    assert torch.cuda.is_available(), "pymomentum-gpu CUDA test requires a visible CUDA device"',
                '    cuda_device = torch.device("cuda")',
                '    print(f"  CUDA device: {torch.cuda.get_device_name(cuda_device)}")',
                "    cuda_tensor = torch.arange(4, device=cuda_device, dtype=torch.float32)",
                "    cuda_result = (cuda_tensor * cuda_tensor).sum()",
                "    torch.cuda.synchronize()",
                '    assert cuda_result.item() == 14.0, "CUDA tensor smoke test produced an unexpected result"',
                "    vertices_cuda = torch.tensor(",
                "        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],",
                "        device=cuda_device,",
                "        dtype=torch.float32,",
                "    )",
                "    triangles_cuda = torch.tensor([[0, 1, 2]], device=cuda_device, dtype=torch.long)",
                "    normals_cuda = diff_geometry.compute_vertex_normals(vertices_cuda, triangles_cuda)",
                "    torch.cuda.synchronize()",
                '    assert normals_cuda.is_cuda, "diff_geometry CUDA smoke test returned a CPU tensor"',
                "    expected_normals = torch.tensor(",
                "        [[1.0, 0.0, 0.0]] * 3, device=cuda_device, dtype=torch.float32",
                "    )",
                "    torch.testing.assert_close(normals_cuda, expected_normals, atol=1e-6, rtol=1e-6)",
                '    print("  PyMomentum diff_geometry CUDA smoke test passed")',
                "    if backend_selection._HAS_TRITON:",
                "        torch.manual_seed(0)",
                "        quats = torch.randn((64, 4), device=cuda_device, dtype=torch.float32)",
                '        torch_result = quaternion.to_rotation_matrix(quats, backend="torch")',
                '        triton_result = quaternion.to_rotation_matrix(quats, backend="triton")',
                "        torch.testing.assert_close(triton_result, torch_result, atol=1e-5, rtol=1e-5)",
                '        print("  Momentum Triton CUDA backend smoke test passed")',
                "    else:",
            ]
        )
        if require_triton:
            lines.append(
                '        raise RuntimeError("Triton is required for this GPU wheel test")'
            )
        else:
            lines.append(
                '        print("  Triton is unavailable; CUDA tensor smoke test passed")'
            )
        lines.extend(
            [
                '    print("[PASS] CUDA smoke tests passed")',
                "except Exception as e:",
                '    print(f"[FAIL] CUDA smoke test failed ({type(e).__name__}): {e}", file=sys.stderr)',
                "    traceback.print_exc()",
                "    sys.exit(1)",
                "",
            ]
        )
    lines.extend(
        [
            'print("Testing parallel operations (MHR issue #44 regression test)...")',
            "try:",
            "    import numpy as np",
            "    import pymomentum.geometry as geom",
            "",
            "    vertices = np.array([",
            "        [0.0, 0.0, 0.0],",
            "        [1.0, 0.0, 0.0],",
            "        [0.0, 1.0, 0.0],",
            "        [1.0, 1.0, 0.0],",
            "    ], dtype=np.float32)",
            "    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)",
            "",
            '    print("  Computing vertex normals...")',
            "    normals = geom.compute_vertex_normals(vertices, triangles)",
            '    assert normals.shape[-1] == 3, f"Expected 3D normals, got shape {normals.shape}"',
            '    assert np.all(np.isfinite(normals)), "Normals contain non-finite values"',
            '    print(f"  Normals shape: {normals.shape}")',
            "",
            "    source = np.random.rand(100, 3).astype(np.float32)",
            "    target = np.random.rand(50, 3).astype(np.float32)",
            '    print("  Finding closest points...")',
            "    result = geom.find_closest_points(source, target)",
            '    print("  Closest points computation completed")',
            "",
            '    print("[PASS] Parallel operations test passed")',
            "except Exception as e:",
            '    print(f"[FAIL] Parallel operations test failed: {e}", file=sys.stderr)',
            "    import traceback",
            "    traceback.print_exc()",
            "    sys.exit(1)",
            "",
            'print("[PASS] All tests passed!")',
        ]
    )
    return "\n".join(lines) + "\n"


def get_torch_after_pymomentum_script() -> str:
    """Return a regression test for PyMomentum's native loader state."""
    return "\n".join(
        [
            "import faulthandler",
            "faulthandler.enable()",
            'print("Testing pymomentum.geometry before torch...")',
            "import pymomentum.geometry",
            'print("  pymomentum.geometry imported")',
            "import torch",
            'print(f"  PyTorch {torch.__version__} imported")',
            'print("[PASS] pymomentum.geometry before torch")',
            "",
        ]
    )


def run_import_tests(
    python_exe: str,
    wheel_type: str,
    require_cuda: bool = False,
    require_triton: bool = False,
) -> int:
    """Run import and parallel operation tests using the specified Python."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    # Keep pixi/conda linker paths out of the import check without deleting
    # runner Python library paths that setup-python may need on Linux.
    sanitize_dynamic_library_env(env)
    for name in CONDA_ENV_VARS:
        env.pop(name, None)
    # Keep the checkout's namespace-package directories out of wheel import checks.
    with tempfile.TemporaryDirectory() as tmpdir:
        if has_torch_runtime(wheel_type):
            result = subprocess.run(
                [
                    python_exe,
                    "-I",
                    "-c",
                    get_torch_after_pymomentum_script(),
                ],
                capture_output=False,
                cwd=tmpdir,
                env=env,
            )
            if result.returncode != 0:
                return result.returncode

        result = subprocess.run(
            [
                python_exe,
                "-I",
                "-c",
                get_import_test_script(wheel_type, require_cuda, require_triton),
            ],
            capture_output=False,
            cwd=tmpdir,
            env=env,
        )
    return result.returncode


def find_container_runtime() -> str | None:
    """Find an available container runtime, preferring docker over podman."""
    for runtime in ("docker", "podman"):
        if shutil.which(runtime):
            return runtime
    return None


def test_in_container(
    wheel_file: Path,
    wheel_type: str,
    py_ver: str,
    require_cuda: bool = False,
    require_triton: bool = False,
) -> int:
    """Test wheel in a manylinux_2_28 container using docker or podman."""
    container_runtime = find_container_runtime()
    if container_runtime is None:
        print(
            "[FAIL] No container runtime found (tried docker, podman). "
            "Falling back to local uv-based testing.",
            file=sys.stderr,
        )
        return test_locally_with_uv(
            wheel_file,
            wheel_type,
            py_ver,
            require_cuda,
            require_triton,
        )
    if require_cuda and container_runtime != "docker":
        print(
            "[FAIL] CUDA container wheel tests require docker with GPU support. "
            "Set WHEEL_TEST_FORCE_LOCAL=1 to test the wheel directly on the host.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Testing wheel in manylinux_2_28 container ({container_runtime}): "
        f"{wheel_file.name}"
    )

    py_tag = py_ver[2:]  # e.g., "312" or "313"
    container_lines = [
        "set -e",
        f"PYTHON=/opt/python/cp{py_tag}-cp{py_tag}/bin/python",
        "",
        'echo "Installing uv..."',
        "$PYTHON -m pip install --quiet --root-user-action=ignore uv",
        "",
        f'echo "Creating virtual environment with Python {py_tag}..."',
        "$PYTHON -m uv venv --python $PYTHON /tmp/test_venv",
        "",
    ]
    if has_torch_runtime(wheel_type):
        torch_index = get_torch_index(wheel_type)
        container_lines.extend(
            [
                f'echo "Installing torch from {torch_index}..."',
                '$PYTHON -m uv pip install --python /tmp/test_venv/bin/python "torch>=2.8.0,<2.9" --index-url '
                + torch_index,
                "",
            ]
        )

    torch_after_pymomentum_script = get_torch_after_pymomentum_script()
    import_test_script = get_import_test_script(
        wheel_type,
        require_cuda,
        require_triton,
    )
    container_lines.extend(
        [
            'echo "Installing wheel..."',
            f"$PYTHON -m uv pip install --python /tmp/test_venv/bin/python /wheel/{wheel_file.name}",
            "",
        ]
    )
    if has_torch_runtime(wheel_type):
        container_lines.extend(
            [
                'echo "Testing pymomentum.geometry before torch..."',
                "cat >/tmp/test_torch_after_pymomentum.py <<'PYTEST'",
                *torch_after_pymomentum_script.rstrip().splitlines(),
                "PYTEST",
                "/tmp/test_venv/bin/python -I /tmp/test_torch_after_pymomentum.py",
                "",
            ]
        )
    container_lines.extend(
        [
            'echo "Testing imports..."',
            "cat >/tmp/test_wheel_imports.py <<'PYTEST'",
            *import_test_script.rstrip().splitlines(),
            "PYTEST",
            "/tmp/test_venv/bin/python -I /tmp/test_wheel_imports.py",
        ]
    )
    container_script = "\n".join(container_lines) + "\n"

    container_command = [container_runtime, "run", "--rm"]
    if require_cuda:
        container_command.extend(["--gpus", "all"])
    container_command.extend(
        [
            "-v",
            f"{wheel_file.parent.absolute()}:/wheel:ro",
            "quay.io/pypa/manylinux_2_28_x86_64:latest",
            "/bin/bash",
            "-c",
            container_script,
        ]
    )

    result = subprocess.run(container_command, capture_output=False)

    return result.returncode


def test_locally_with_uv(
    wheel_file: Path,
    wheel_type: str,
    py_ver: str,
    require_cuda: bool = False,
    require_triton: bool = False,
) -> int:
    """Test wheel locally using uv for fast virtual environment creation."""
    print(f"Testing wheel locally with uv: {wheel_file.name}")

    # Check if uv is available
    uv_path = shutil.which("uv")
    if not uv_path:
        print("Installing uv...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True
        )
        uv_path = shutil.which("uv") or "uv"

    py_version = f"{py_ver[2]}.{py_ver[3:]}"  # "cp312" -> "3.12"

    with tempfile.TemporaryDirectory() as tmpdir:
        venv_path = Path(tmpdir) / "test_venv"

        # Determine python executable path based on platform
        if sys.platform == "win32":
            python_exe = str(venv_path / "Scripts" / "python.exe")
        else:
            python_exe = str(venv_path / "bin" / "python")

        print(f"Creating virtual environment with Python {py_version}...")
        result = subprocess.run(
            [uv_path, "venv", "--python", py_version, str(venv_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            print(f"[FAIL] Failed to create venv: {result.stderr}", file=sys.stderr)
            return result.returncode

        if has_torch_runtime(wheel_type):
            torch_index = get_torch_index(wheel_type)
            print(f"Installing torch from {torch_index}...")
            result = subprocess.run(
                [
                    uv_path,
                    "pip",
                    "install",
                    "--python",
                    python_exe,
                    "torch>=2.8.0,<2.9",
                    "--index-url",
                    torch_index,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                print(
                    f"[FAIL] Failed to install torch: {result.stderr}", file=sys.stderr
                )
                return result.returncode

        print(f"Installing wheel {wheel_file.name}...")
        result = subprocess.run(
            [uv_path, "pip", "install", "--python", python_exe, str(wheel_file)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            print(f"[FAIL] Failed to install wheel: {result.stderr}", file=sys.stderr)
            return result.returncode

        return run_import_tests(python_exe, wheel_type, require_cuda, require_triton)


def main():
    # Allow override via environment variables for CI flexibility
    py_ver = os.environ.get("WHEEL_TEST_PYTHON_VERSION")
    if not py_ver:
        py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"

    env_name = os.environ.get("PIXI_ENVIRONMENT_NAME", "")
    wheel_type = os.environ.get("WHEEL_TEST_TYPE") or os.environ.get(
        "PYMOMENTUM_VARIANT"
    )
    if not wheel_type:
        wheel_type = "gpu" if "cuda" in env_name or "gpu" in env_name else "core"
    wheel_type = normalize_wheel_type(wheel_type)
    if wheel_type not in WHEEL_PREFIX_BY_TYPE:
        print(f"[FAIL] Unknown wheel type: {wheel_type}", file=sys.stderr)
        sys.exit(1)

    require_cuda = env_flag("WHEEL_TEST_REQUIRE_CUDA")
    require_triton = env_flag("WHEEL_TEST_REQUIRE_TRITON")
    force_local = env_flag("WHEEL_TEST_FORCE_LOCAL")
    if require_cuda and wheel_type != "gpu":
        print(
            "[FAIL] WHEEL_TEST_REQUIRE_CUDA is only valid for gpu wheels",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Testing {wheel_type} wheel for Python {py_ver}")

    # Find the wheel file
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("[FAIL] dist/ directory not found", file=sys.stderr)
        sys.exit(1)

    wheel_prefix = WHEEL_PREFIX_BY_TYPE[wheel_type]
    wheel_files = [
        f
        for f in dist_dir.iterdir()
        if f.name.startswith(wheel_prefix)
        and py_ver in f.name
        and f.name.endswith(".whl")
    ]

    if not wheel_files:
        print(f"[FAIL] No wheel found for {wheel_type} {py_ver}", file=sys.stderr)
        print(f"  Available wheels: {list(dist_dir.glob('*.whl'))}", file=sys.stderr)
        sys.exit(1)

    # Pick the most recently modified wheel to avoid stale wheels from prior steps
    wheel_file = max(wheel_files, key=lambda f: f.stat().st_mtime)
    print(f"Found wheel: {wheel_file.name}")

    # Choose test method based on platform and wheel type
    if sys.platform == "linux" and "manylinux" in wheel_file.name and not force_local:
        # Test in container for manylinux wheels to ensure compatibility
        returncode = test_in_container(
            wheel_file,
            wheel_type,
            py_ver,
            require_cuda,
            require_triton,
        )
    else:
        # For macOS/Windows or non-manylinux wheels, test locally with uv
        returncode = test_locally_with_uv(
            wheel_file,
            wheel_type,
            py_ver,
            require_cuda,
            require_triton,
        )

    sys.exit(returncode)


if __name__ == "__main__":
    main()
