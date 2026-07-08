# Momentum AI Assistant Guide

Momentum is a C++ library for human kinematic motion, character representation,
and numerical optimization solvers. The repository also contains PyMomentum,
the Python bindings and Python utilities built on top of the C++ library.

This guide is for AI coding agents working in the public repository. Keep future
instructions codebase-focused and self-contained: use public paths, public
commands, and public documentation. Do not add private organization-specific
workflow, review, source-control, or infrastructure rules here.

For end-user installation, building from source, and the contribution workflow,
see `README.md` and `CONTRIBUTING.md`.

## Repository Map

- `momentum/` - C++ core library.
  - `common/` - exceptions, checks, logging, shared utilities.
  - `math/` - transforms, meshes, random utilities, numerical helpers.
  - `character/` - skeletons, characters, skinning, collision geometry.
  - `solver/` - generic solver framework.
  - `character_solver/` - character-specific inverse kinematics solvers.
  - `character_solver_simd/` - SIMD-accelerated solver error functions.
  - `character_sequence_solver/` - multi-frame sequence solvers.
  - `diff_ik/` - differentiable inverse kinematics support.
  - `simd/` - SIMD helpers and abstractions shared by the compute paths.
  - `io/` - file I/O including FBX, glTF, URDF, C3D, and related formats.
  - `marker_tracking/`, `rasterizer/`, `camera/`, `gui/` - application-facing
    support libraries.
  - `test/` - C++ unit tests and test helpers.
  - `examples/`, `tutorials/` - example applications and tutorials.
  - `website/` - documentation site sources.
- `pymomentum/` - Python bindings and Python modules.
  - `geometry/` - NumPy-oriented character and geometry bindings.
  - `diff_geometry/`, `solver/`, `tensor_momentum/` - PyTorch-backed modules.
  - `solver2/`, `marker_tracking/`, `renderer/`, `camera/`, `axel/` - Python
    extension modules and helpers.
  - `test/` - Python tests.
- `axel/` - bundled Axel acceleration library (sources under `axel/axel/`).
- `cmake/` - CMake helper functions and source lists. `cmake/mt_defs.cmake`
  defines the `mt_library`, `mt_test`, and `mt_executable` helpers used
  throughout `CMakeLists.txt`.
- `scripts/` - wheel, packaging, and CI helper scripts.
- `.github/workflows/` - GitHub Actions CI, nightly, wheel, and docs workflows.
- `CMakeLists.txt`, `pixi.toml`, `pyproject.toml` - top-level build, task, and
  Python packaging entry points.

## Build And Test

Use Pixi tasks for normal development. Dependencies install automatically on the
first run.

```bash
pixi run build          # Release C++ build, with Python bindings enabled
pixi run build_dev      # Debug C++ build
pixi run test           # Release C++ tests
pixi run test_dev       # Debug C++ tests
pixi run test_verbose   # Release C++ tests with verbose output
pixi run build_py       # Build Python bindings
pixi run test_py        # Python tests
pixi run lint           # Format C++ code with clang-format
pixi run lint-check     # Check C++ formatting
pixi task list          # Show all available tasks
```

Useful environment-specific commands:

```bash
pixi run -e py313 build_py
pixi run -e py313 test_py
pixi run -e gpu build_py
```

Windows developers can use `pixi run open_vs` to open the configured Visual
Studio project.

## Platform Notes

- macOS builds are supported on both Intel and Apple Silicon.
- On Linux the FBX SDK is downloaded automatically during environment setup
  (the `install_fbxsdk` task).
- FBX I/O is available on Linux, macOS, and Windows (x86_64); USD I/O requires
  the optional USD dependencies.
- SIMD-accelerated paths target x86_64 (AVX2/FMA); other architectures such as
  ARM fall back to scalar implementations.

## Source Lists And Generated Files

- When adding, removing, or renaming C++ files under `momentum/`, update
  `cmake/build_variables.bzl`. CMake reads these lists through
  `cmake/mt_defs.cmake`.
- When adding, removing, or renaming PyMomentum C++ binding files, update
  `pymomentum/cmake/build_variables.bzl`.
- When adding new public C++ classes that need forward declarations, update
  `momentum/gen_fwd_input.toml`. The per-module `fwd.h` files are generated from
  it — each class gets `_p`/`_u`/`_w` (and `_const_*`) smart-pointer aliases. Do
  not hand-edit `fwd.h`; change the TOML source instead.
- When changing PyPI packaging, edit `pyproject-pypi.toml.j2` and regenerate the
  derived files with `pixi run generate_pyproject`.
- Do not hand-edit generated build or packaging outputs when a source template
  exists.

## C++ Conventions

- Follow the C++ style guide in the developer guide on the documentation site:
  https://facebookresearch.github.io/momentum/docs_cpp/developer_guide/style_guide
- Use Doxygen-style `///` comments in public headers. Use `@param` and `@return`
  tags, and document new public classes, structs, and functions.
- Use `camelCase` for functions, variables, and members; `PascalCase` for class
  names; `ALL_CAPS` for macros; `snake_case` for files and directories; and a
  trailing underscore for private members.
- Write abbreviations in `PascalCase` too, e.g. `Fbx` and `Gltf`, not `FBX` or
  `GLTF`.
- Prefer early return, `continue`, or `break` over deeply nested control flow.
- Prefer `std::span` for read-only or mutable contiguous ranges accepted by
  function parameters.
- Use `MT_THROW("message: {}", value)` for exceptions and `MT_CHECK(cond, ...)`
  for runtime assertions, both from `<momentum/common/exception.h>`.
- Use `momentum::pi<T>()` and `momentum::twopi<T>()` from
  `<momentum/math/constants.h>` instead of `M_PI` for Windows compatibility.
- Include ordering:
  1. Corresponding header.
  2. Momentum headers.
  3. Third-party headers.
  4. Standard library headers.
- Public headers should include Momentum headers with angle brackets, for
  example `#include <momentum/common/checks.h>`. Implementation files usually
  include local Momentum headers with quotes.

## PyMomentum Conventions

- Python-facing argument names use `snake_case`, even when the C++ API uses
  `camelCase`.
- Bind multi-dimensional Eigen inputs as `pybind11::array_t<float>` to avoid
  NumPy row-major versus Eigen column-major confusion. One-dimensional vectors
  can use pybind11 Eigen conversion.
- For quaternion inputs and outputs, include
  `pymomentum/python_utility/eigen_quaternion.h`; the Python format is
  `[x, y, z, w]`.
- For optional Eigen arguments, prefer `std::optional<Eigen::...>` instead of
  default Eigen values in the binding declaration.
- Use `py::ssize_t`, not POSIX `ssize_t`, in pybind11 code so Windows builds
  compile.
- For constructors or option structs with multiple arguments, prefer
  `py::kw_only()` plus a lambda-based `py::init` to avoid silent bugs from
  positional argument reordering.
- Public bindings should have Sphinx-friendly docstrings. Keep `:param:`,
  `:return:`, and cross-reference names aligned with the exposed Python
  argument names.
- Format Python with `black` and follow PEP 8, including type hints on public
  APIs.
- `pymomentum.geometry` is NumPy-oriented and should stay suitable for fast
  forward operations, visualization, and non-differentiable workflows.
- `pymomentum.diff_geometry` and `pymomentum.solver` are PyTorch-backed modules
  for differentiable operations and must respect the supported PyTorch ABI.
- After changing C++ bindings, rebuild before testing Python: `pixi run
  build_py`. Pure Python changes do not require a rebuild.

## Tests

- Add or update tests with behavior changes.
- C++ tests live under `momentum/test/`; Python tests live under
  `pymomentum/test/`.
- Prefer existing test helpers:
  - `createTestCharacter(numJoints)` (minimum 3 joints) from
    `<momentum/test/character/character_helpers.h>` instead of manually
    assembling skeletons, parameter transforms, meshes, locators, and collision
    geometry.
  - `temporaryFile()` and `temporaryDirectory()` from
    `<momentum/test/io/io_helpers.h>` instead of directly using
    `std::filesystem::temp_directory_path()`.
- Seed random tests deterministically. Prefer a fixture-owned
  `Random<> rng{12345};` from `<momentum/math/random.h>`. When extending an
  older test that already uses the singleton helpers, seed
  `Random<>::GetSingleton()` at the start of each test. Do not use Eigen's
  `Random()` / `UnitRandom()` — they seed from `std::random_device` and produce
  different values every run.
- For optional Python dependencies, gate the whole `unittest.TestCase` class
  behind an import flag rather than using per-test skip decorators. This keeps
  pytest collection clean when optional dependencies are unavailable.
- To run a single Python test, enter `pixi shell` and invoke `pytest` directly
  with a `-k` filter.

## Documentation

- Keep code examples out of the `momentum/website/` docs — they go stale.
  Describe the approach and reference a test or example instead; when code is
  unavoidable, prefer pseudocode.
- Public documentation lives on the docs site: the Python and C++ API references
  and the getting-started guides linked from `README.md`.

## CI And Packaging

- GitHub Actions workflows live in `.github/workflows/`. Keep platform matrices
  intentional and use existing workflow patterns before adding new mechanisms.
- Wheel builds have three variants:
  - `core` - NumPy/SciPy modules and torch-backed pure Python helpers, without
    Torch C++ extension modules.
  - `cpu` - includes Torch C++ extension modules for CPU PyTorch.
  - `gpu` - includes Torch C++ extension modules for CUDA PyTorch.
- Build and test wheel variants with:

```bash
pixi run -e py313 generate_pyproject
PYMOMENTUM_VARIANT=core pixi run -e py313 wheel_build
PYMOMENTUM_VARIANT=core pixi run -e py313 wheel_check
PYMOMENTUM_VARIANT=core pixi run -e py313 wheel_test
```

Use `PYMOMENTUM_VARIANT=cpu` or `PYMOMENTUM_VARIANT=gpu` for the other wheel
families. `wheel_check` validates the generated packaging; `wheel_test` checks
clean-environment imports and PyTorch import-order compatibility.

## Change Checklist

Before handing off code changes, run the smallest relevant checks first, then a
broader pass when the change affects shared behavior.

- C++ changes: `pixi run lint`, relevant C++ test target or `pixi run test`.
- PyMomentum binding changes: `pixi run build_py`, relevant Python tests, then
  `pixi run test_py` when practical.
- Packaging changes: `pixi run generate_pyproject`, then relevant
  `wheel_build`, `wheel_check`, and `wheel_test` tasks.
- CI changes: inspect the affected workflow and, when possible, run the command
  the workflow invokes locally.
