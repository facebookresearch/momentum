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
import zipfile
from pathlib import Path

from demote_gnu_unique_symbols import count_unique_symbols
from finalize_pypi_wheel import GUARD_MARKER, INIT, INIT_PYC, INIT_PYW

WHEEL_PREFIX_BY_TYPE = {
    "core": "pymomentum_core-",
    "cpu": "pymomentum_cpu-",
    "gpu": "pymomentum_gpu-",
}
DIST_NAME_BY_TYPE = {
    "core": "pymomentum-core",
    "cpu": "pymomentum-cpu",
    "gpu": "pymomentum-gpu",
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


# The last release whose CPU and GPU wheels installed every module themselves.
PREVIOUS_FULL_PACKAGE_VERSION = "0.1.114.post0"
REPAIR_MESSAGE = "Repair with:"


def has_torch_extensions(wheel_type: str) -> bool:
    return wheel_type in {"cpu", "gpu"}


def get_torch_requirement(wheel_type: str) -> str:
    """Return the PyTorch requirement to install before the wheel."""
    if has_torch_extensions(wheel_type):
        return "torch>=2.8.0,<2.9"
    return "torch>=1.13"


def check_wheel_contents(
    wheel_file: Path, wheel_type: str, core_wheel: Path | None = None
) -> None:
    """Check that core and Torch add-on wheels own disjoint payloads."""
    with zipfile.ZipFile(wheel_file) as archive:
        names = {name for name in archive.namelist() if not name.endswith("/")}
        init = archive.read(INIT).decode("utf-8") if INIT in names else ""
        private_libstdcxx = [
            name
            for name in names
            if Path(name).name.startswith("libstdc++-pymomentum.so")
            and count_unique_symbols(archive.read(name))
        ]
    package_files = {name for name in names if name.startswith("pymomentum/")}
    if any(name.startswith(("include/", "lib/", "bin/", "share/")) for name in names):
        raise RuntimeError(f"{wheel_file.name} contains C++ SDK files")
    if any("geometry_test_utils" in name for name in names):
        raise RuntimeError(f"{wheel_file.name} contains test-only bindings")
    if private_libstdcxx:
        raise RuntimeError(
            f"{private_libstdcxx} has STB_GNU_UNIQUE symbols, so it can bind to "
            "another libstdc++ in the process"
        )

    pure_torch_paths = {
        "pymomentum/backend/selection.py",
        "pymomentum/quaternion.py",
        "pymomentum/skel_state.py",
        "pymomentum/torch/character.py",
        "pymomentum/torch/renderer.py",
        "pymomentum/trs.py",
    }
    if wheel_type == "core":
        if GUARD_MARKER not in init:
            raise RuntimeError(f"core wheel {INIT} lacks the install check")
        if "-win" in wheel_file.name and INIT_PYW not in package_files:
            raise RuntimeError(f"Windows core wheel lacks {INIT_PYW}")
        missing = sorted(pure_torch_paths - package_files)
        if missing:
            raise RuntimeError(f"core wheel is missing Torch helpers: {missing}")
        unexpected = sorted(
            name
            for name in package_files
            if name.startswith("pymomentum/diff_geometry.")
            or name.startswith("pymomentum/solver.")
        )
        if unexpected:
            raise RuntimeError(
                f"core wheel contains compiled Torch extensions: {unexpected}"
            )
        return

    if core_wheel is None:
        raise RuntimeError("CPU/GPU wheel tests require the matching core wheel")
    if INIT_PYC not in package_files:
        raise RuntimeError(f"{wheel_type} wheel lacks the {INIT_PYC} install check")
    unexpected = sorted(
        name
        for name in package_files
        if name in pure_torch_paths
        or name.startswith("pymomentum/backend/")
        or name.startswith("pymomentum/torch/")
    )
    if unexpected:
        raise RuntimeError(f"{wheel_type} wheel contains core files: {unexpected}")
    with zipfile.ZipFile(core_wheel) as archive:
        core_files = {
            name
            for name in archive.namelist()
            if name.startswith("pymomentum/") and not name.endswith("/")
        }
    overlap = sorted(package_files & core_files)
    if overlap:
        raise RuntimeError(f"core and {wheel_type} wheels overlap: {overlap}")


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
    lines.extend(
        [
            "    import pymomentum.backend as backend",
            "    import pymomentum.quaternion as quaternion",
            "    import pymomentum.skel_state as skel_state",
            "    import pymomentum.torch.character as torch_character",
            "    import pymomentum.trs as trs",
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
                    "    import pymomentum.quaternion_np as quaternion_np",
                    "    import pymomentum.skel_state_np as skel_state_np",
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
    """Return a regression test that imports pymomentum.geometry before torch."""
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


# How each runtime test sets up the process before PyMomentum's number I/O.
CXX_RUNTIME_TRIGGERS = {
    "torch_first": ["import torch", "import pymomentum.geometry as geometry"],
    "pymomentum_first": ["import pymomentum.geometry as geometry", "import torch"],
    # PyTorch is installed, but the program never imports it.
    "torch_not_imported": ["import pymomentum.geometry as geometry"],
    # Another copy of libstdc++ numbers its locale facets first, without torch.
    "system_locale_first": [
        "import ctypes",
        'ctypes.CDLL("libstdc++.so.6")._ZNSt6locale7classicEv()',
        "import pymomentum.geometry as geometry",
    ],
}


def get_cxx_runtime_triggers(linux: bool) -> list[str]:
    """Return the runtime test triggers that apply to the test platform."""
    if linux:
        return list(CXX_RUNTIME_TRIGGERS)
    return [name for name in CXX_RUNTIME_TRIGGERS if name != "system_locale_first"]


def get_cxx_runtime_script(trigger: str) -> str:
    """Return a regression test for PyMomentum's private C++ runtime.

    Parsing the URDF numbers and naming the blend-shape parameters use iostream
    number I/O, which crashed when the private libstdc++ bound its locale facet
    ids to another libstdc++ in the process.
    """
    return "\n".join(
        [
            "import faulthandler",
            "faulthandler.enable()",
            "import sys",
            "import numpy as np",
            *CXX_RUNTIME_TRIGGERS[trigger],
            f'print("Testing with_blend_shape, {trigger}...")',
            "urdf = (",
            '    b\'<robot name="r"><link name="base"/><link name="tip"/>\'',
            '    b\'<joint name="j" type="revolute"><parent link="base"/><child link="tip"/>\'',
            '    b\'<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>\'',
            "    b'</joint></robot>'",
            ")",
            "character = geometry.Character.load_urdf_from_bytes(urdf).with_mesh_and_skin_weights(",
            "    geometry.Mesh(np.eye(3, dtype=np.float32), np.array([[0, 1, 2]], dtype=np.int32)),",
            "    geometry.SkinWeights(np.zeros((3, 1), dtype=np.int32), np.ones((3, 1), dtype=np.float32)),",
            ")",
            "blend_shape = geometry.BlendShape.from_tensors(",
            "    np.zeros((3, 3), dtype=np.float32), np.ones((2, 3, 3), dtype=np.float32)",
            ")",
            "names = character.with_blend_shape(blend_shape).parameter_transform.names",
            'assert names[-2:] == ["blend_0", "blend_1"], names',
            *(
                ['assert "torch" not in sys.modules, "pymomentum imported torch"']
                if trigger in ("torch_not_imported", "system_locale_first")
                else []
            ),
            f'print("[PASS] with_blend_shape, {trigger}")',
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
        linux = sys.platform.startswith("linux")
        for script in (
            get_torch_after_pymomentum_script(),
            *(
                get_cxx_runtime_script(trigger)
                for trigger in get_cxx_runtime_triggers(linux)
            ),
            *(
                [get_torch_extension_binding_script()]
                if linux and has_torch_extensions(wheel_type)
                else []
            ),
        ):
            result = subprocess.run(
                [python_exe, "-I", "-c", script],
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


def get_previous_release_container_lines(
    wheel_file: Path, wheel_type: str, core_wheel: Path
) -> list[str]:
    """Shell steps for installs that start from the last full-package add-on."""
    old = f"{DIST_NAME_BY_TYPE[wheel_type]}=={PREVIOUS_FULL_PACKAGE_VERSION}"
    new_wheels = f"/wheel/{core_wheel.name} /wheel/{wheel_file.name}"
    torch = f'"{get_torch_requirement(wheel_type)}" --index-url ' + get_torch_index(
        wheel_type
    )
    repair = [
        f"$VENV -m pip uninstall -y pymomentum-core {DIST_NAME_BY_TYPE[wheel_type]}",
        f"$VENV -m pip install {new_wheels}",
        "$VENV -I /tmp/test_wheel_imports.py",
    ]
    lines = [
        "",
        f'echo "Testing a plain pip upgrade from {old}..."',
        "VENV=/tmp/upgrade_venv/bin/python",
        "$PYTHON -m venv /tmp/upgrade_venv",
        f"$VENV -m pip install --quiet {torch}",
        f'$VENV -m pip install --quiet --only-binary :all: "{old}"',
        f"$VENV -m pip install --upgrade {new_wheels}",
        "if ! $VENV -I /tmp/test_wheel_imports.py; then",
        f'  $VENV -I -c "import pymomentum" 2>&1 | grep "{REPAIR_MESSAGE}" || '
        '{ echo "[FAIL] broken install without the repair message"; exit 1; }',
        '  echo "Upgrade failed with the repair message; running the repair..."',
        *(f"  {line}" for line in repair),
        "fi",
    ]
    if wheel_type != "cpu":
        return lines
    return lines + [
        "",
        f'echo "Testing {old} next to the new core..."',
        "VENV=/tmp/coexist_venv/bin/python",
        "$PYTHON -m venv /tmp/coexist_venv",
        f"$VENV -m pip install --quiet {torch}",
        f'$VENV -m pip install --quiet --only-binary :all: "{old}"',
        f"$VENV -m pip install /wheel/{core_wheel.name}",
        'if $VENV -I -c "import pymomentum.geometry" >/tmp/coexist.log 2>&1; then',
        '  echo "[FAIL] the old add-on next to the new core imported silently"',
        "  exit 1",
        "fi",
        f'grep "{REPAIR_MESSAGE}" /tmp/coexist.log || '
        '{ cat /tmp/coexist.log; echo "[FAIL] no repair message"; exit 1; }',
        *repair,
    ]


def get_torch_extension_binding_script() -> str:
    """Return a Linux test that imports each PyMomentum extension module before torch.

    The test fails if torch's libraries or the system libstdc++ bind to
    PyMomentum's private libstdc++ (LD_DEBUG=bindings), which catches an
    extension module that links libtorch but is missing from the loader.
    """
    return """import importlib.machinery
import importlib.util
import os
import re
import subprocess
import sys
import tempfile

PRIVATE = r" to [^ ]*libstdc\\+\\+-pymomentum\\.so\\.6 "
PATTERNS = {
    "torch libraries": re.compile(
        r"binding file [^ ]*/torch/lib/lib(torch|torch_cpu|torch_python|c10)"
        r"\\.so \\[0\\]" + PRIVATE
    ),
    "system libstdc++": re.compile(
        r"binding file [^ ]*/libstdc\\+\\+\\.so\\.6 \\[0\\]" + PRIVATE
    ),
}
package_dirs = importlib.util.find_spec("pymomentum").submodule_search_locations
modules = sorted(
    {
        name.split(".")[0]
        for package_dir in package_dirs
        for name in os.listdir(package_dir)
        if name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
    }
)
if not modules:
    sys.exit("[FAIL] no PyMomentum extension modules found")
failed = False
for module in modules:
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, LD_DEBUG="bindings")
        env["LD_DEBUG_OUTPUT"] = os.path.join(tmp, "bindings")
        subprocess.run(
            [sys.executable, "-I", "-c", f"import pymomentum.{module}"],
            env=env,
            check=True,
        )
        counts = dict.fromkeys(PATTERNS, 0)
        for name in os.listdir(tmp):
            with open(os.path.join(tmp, name), errors="replace") as log:
                for line in log:
                    for key, pattern in PATTERNS.items():
                        counts[key] += bool(pattern.search(line))
    print(f"pymomentum.{module} before torch, bindings to the private libstdc++:", counts)
    failed = failed or any(counts.values())
if failed:
    sys.exit("[FAIL] a library bound to PyMomentum's private libstdc++")
print("[PASS] every extension module before torch")
"""


def get_cxx_runtime_container_lines() -> list[str]:
    """Return shell lines that run every runtime test in the Linux container."""
    lines = []
    for trigger in get_cxx_runtime_triggers(linux=True):
        script_path = f"/tmp/test_cxx_runtime_{trigger}.py"
        lines.extend(
            [
                f"cat >{script_path} <<'PYTEST'",
                *get_cxx_runtime_script(trigger).rstrip().splitlines(),
                "PYTEST",
                f"/tmp/test_venv/bin/python -I {script_path}",
                "",
            ]
        )
    return lines


def test_in_container(
    wheel_file: Path,
    wheel_type: str,
    py_ver: str,
    core_wheel: Path | None = None,
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
            core_wheel,
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
    torch_index = get_torch_index(wheel_type)
    container_lines.extend(
        [
            f'echo "Installing torch from {torch_index}..."',
            "$PYTHON -m uv pip install --python /tmp/test_venv/bin/python "
            f'"{get_torch_requirement(wheel_type)}" --index-url {torch_index}',
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
            "$PYTHON -m uv pip install --python /tmp/test_venv/bin/python "
            + " ".join(
                f"/wheel/{path.name}"
                for path in ([core_wheel, wheel_file] if core_wheel else [wheel_file])
            ),
            "$PYTHON -m uv pip check --python /tmp/test_venv/bin/python",
            "",
        ]
    )
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
    container_lines.extend(get_cxx_runtime_container_lines())
    if has_torch_extensions(wheel_type):
        container_lines.extend(
            [
                "cat >/tmp/test_torch_extension_bindings.py <<'PYTEST'",
                *get_torch_extension_binding_script().rstrip().splitlines(),
                "PYTEST",
                "/tmp/test_venv/bin/python -I /tmp/test_torch_extension_bindings.py",
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
    if core_wheel is not None:
        core_import_script = get_import_test_script("core")
        container_lines.extend(
            [
                "",
                'echo "Testing core after uninstalling add-on..."',
                "$PYTHON -m uv pip uninstall --python /tmp/test_venv/bin/python "
                + DIST_NAME_BY_TYPE[wheel_type],
                "cat >/tmp/test_core_imports.py <<'PYTEST'",
                *core_import_script.rstrip().splitlines(),
                "PYTEST",
                "/tmp/test_venv/bin/python -I /tmp/test_core_imports.py",
            ]
        )
    if core_wheel is not None:
        container_lines.extend(
            get_previous_release_container_lines(wheel_file, wheel_type, core_wheel)
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
    core_wheel: Path | None = None,
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

        torch_index = get_torch_index(wheel_type)
        print(f"Installing torch from {torch_index}...")
        result = subprocess.run(
            [
                uv_path,
                "pip",
                "install",
                "--python",
                python_exe,
                get_torch_requirement(wheel_type),
                "--index-url",
                torch_index,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            print(f"[FAIL] Failed to install torch: {result.stderr}", file=sys.stderr)
            return result.returncode

        print(f"Installing wheel {wheel_file.name}...")
        result = subprocess.run(
            [
                uv_path,
                "pip",
                "install",
                "--python",
                python_exe,
                *(
                    [str(core_wheel), str(wheel_file)]
                    if core_wheel
                    else [str(wheel_file)]
                ),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            print(f"[FAIL] Failed to install wheel: {result.stderr}", file=sys.stderr)
            return result.returncode

        result = subprocess.run(
            [uv_path, "pip", "check", "--python", python_exe],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            print(f"[FAIL] Dependency check failed: {result.stderr}", file=sys.stderr)
            return result.returncode

        returncode = run_import_tests(
            python_exe,
            wheel_type,
            require_cuda,
            require_triton,
        )
        if returncode != 0 or core_wheel is None:
            return returncode

        print(f"Uninstalling {DIST_NAME_BY_TYPE[wheel_type]} and retesting core...")
        result = subprocess.run(
            [
                uv_path,
                "pip",
                "uninstall",
                "--python",
                python_exe,
                DIST_NAME_BY_TYPE[wheel_type],
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            print(
                f"[FAIL] Failed to uninstall add-on: {result.stderr}", file=sys.stderr
            )
            return result.returncode
        returncode = run_import_tests(python_exe, "core")
        if returncode != 0:
            return returncode
        scenarios = ["upgrade", "coexist"] if wheel_type == "cpu" else ["upgrade"]
        for scenario in scenarios:
            returncode = test_previous_release_install(
                uv_path,
                py_version,
                Path(tmpdir) / f"{scenario}_venv",
                scenario,
                wheel_file,
                wheel_type,
                core_wheel,
                require_cuda,
                require_triton,
            )
            if returncode != 0:
                return returncode
        return 0


def test_previous_release_install(
    uv_path: str,
    py_version: str,
    venv_path: Path,
    scenario: str,
    wheel_file: Path,
    wheel_type: str,
    core_wheel: Path,
    require_cuda: bool,
    require_triton: bool,
) -> int:
    """Start from the last full-package add-on, then add the new wheels with pip.

    "upgrade" runs a plain `pip install --upgrade` of core and the add-on; it
    must work or fail with the repair message. "coexist" adds only the new core
    next to the old add-on; importing must fail with the repair message. Either
    way the printed repair must give a working install.
    """
    old = f"{DIST_NAME_BY_TYPE[wheel_type]}=={PREVIOUS_FULL_PACKAGE_VERSION}"
    print(f"Testing {scenario} from {old}...", flush=True)
    result = subprocess.run(
        [uv_path, "venv", "--seed", "--python", py_version, str(venv_path)]
    )
    if result.returncode != 0:
        return result.returncode
    if sys.platform == "win32":
        python_exe = str(venv_path / "Scripts" / "python.exe")
    else:
        python_exe = str(venv_path / "bin" / "python")
    pip = [python_exe, "-m", "pip"]
    torch = [
        get_torch_requirement(wheel_type),
        "--index-url",
        get_torch_index(wheel_type),
    ]
    new_wheels = (
        [str(core_wheel), str(wheel_file)]
        if scenario == "upgrade"
        else [str(core_wheel)]
    )
    for command in (
        [*pip, "install", "--quiet", *torch],
        [*pip, "install", "--quiet", "--only-binary", ":all:", old],
        [*pip, "install", "--upgrade", *new_wheels],
    ):
        if subprocess.run(command).returncode != 0:
            print(f"[FAIL] {' '.join(command)}", file=sys.stderr)
            return 1

    probe = subprocess.run(
        [python_exe, "-I", "-c", "import pymomentum.geometry"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if probe.returncode == 0 and scenario == "upgrade":
        return run_import_tests(python_exe, wheel_type, require_cuda, require_triton)
    if probe.returncode == 0:
        print("[FAIL] the old add-on next to the new core imported silently")
        return 1
    if REPAIR_MESSAGE not in probe.stderr:
        print(f"[FAIL] broken install without the repair message:\n{probe.stderr}")
        return 1
    print(probe.stderr.strip().splitlines()[-3:], flush=True)
    print("Running the repair from the message...", flush=True)
    for command in (
        [*pip, "uninstall", "-y", "pymomentum-core", DIST_NAME_BY_TYPE[wheel_type]],
        [*pip, "install", str(core_wheel), str(wheel_file)],
    ):
        if subprocess.run(command).returncode != 0:
            print(f"[FAIL] {' '.join(command)}", file=sys.stderr)
            return 1
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
    core_wheel = None
    if wheel_type != "core":
        core_wheels = [
            path
            for path in dist_dir.iterdir()
            if path.name.startswith(WHEEL_PREFIX_BY_TYPE["core"])
            and py_ver in path.name
            and path.name.endswith(".whl")
        ]
        if not core_wheels:
            print(f"[FAIL] No matching core wheel found for {py_ver}", file=sys.stderr)
            sys.exit(1)
        core_wheel = max(core_wheels, key=lambda path: path.stat().st_mtime)
        print(f"Found core wheel: {core_wheel.name}")

    check_wheel_contents(wheel_file, wheel_type, core_wheel)

    # Choose test method based on platform and wheel type
    if sys.platform == "linux" and "manylinux" in wheel_file.name and not force_local:
        # Test in container for manylinux wheels to ensure compatibility
        returncode = test_in_container(
            wheel_file,
            wheel_type,
            py_ver,
            core_wheel,
            require_cuda,
            require_triton,
        )
    else:
        # For macOS/Windows or non-manylinux wheels, test locally with uv
        returncode = test_locally_with_uv(
            wheel_file,
            wheel_type,
            py_ver,
            core_wheel,
            require_cuda,
            require_triton,
        )

    sys.exit(returncode)


if __name__ == "__main__":
    main()
