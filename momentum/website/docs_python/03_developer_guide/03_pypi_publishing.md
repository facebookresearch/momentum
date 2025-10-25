# PyPI Publishing Guide

This guide explains how to maintain and update PyPI publishing configuration for `pymomentum-cpu` and `pymomentum-gpu` packages.

## System Overview

The PyPI publishing system uses a **template-based approach** to generate platform and Python version-specific configurations:

- **Template**: `pyproject-pypi.toml.j2` - Jinja2 template with PyTorch version placeholders
- **Generator**: `scripts/generate_pyproject.py` - Renders template with version constraints
- **Generated files**: `pyproject-pypi-cpu.toml` and `pyproject-pypi-gpu.toml` (temporary, gitignored)
- **CI workflow**: `.github/workflows/publish_to_pypi.yml` - Builds and publishes wheels

## Updating PyTorch Versions

When new PyTorch versions are released, update the defaults in `scripts/generate_pyproject.py`:

```python
# Linux PyTorch versions
parser.add_argument("--torch-min-py312", default="2.8.0")  # Update here
parser.add_argument("--torch-max-py312", default="2.9")    # Update here

# macOS PyTorch versions (usually older due to platform limitations)
parser.add_argument("--torch-min-py312-macos", default="2.2.0")  # Update here
parser.add_argument("--torch-max-py312-macos", default="2.3")    # Update here
```

**Note**: `--torch-max` is **exclusive** (e.g., `2.9` means `<2.9`, so PyTorch 2.9.x is NOT allowed).

## Adding New Python Versions

To support Python 3.14+:

1. **Update `scripts/generate_pyproject.py`** - Add new CLI arguments:
   ```python
   parser.add_argument("--torch-min-py314", default="X.X.X")
   parser.add_argument("--torch-max-py314", default="X.X")
   parser.add_argument("--torch-min-py314-macos", default="X.X.X")
   parser.add_argument("--torch-max-py314-macos", default="X.X")
   ```

2. **Update `pyproject-pypi.toml.j2`** - Add version-specific constraints:
   ```jinja
   "torch>={{ torch_min_py314 }},<{{ torch_max_py314 }}; ... and python_version == '3.14'",
   "torch>={{ torch_min_py314_macos }},<{{ torch_max_py314_macos }}; ... and python_version == '3.14'",
   ```

3. **Update `pixi.toml`** - Add new feature and environment:
   ```toml
   [feature.py314]
   dependencies = { python = "3.14.*" }

   [environments]
   py314 = ["py314"]
   ```

4. **Update CI workflow** - Add to matrix:
   ```yaml
   matrix:
     python-version: ['3.12', '3.13', '3.14']
     include:
       - python-version: '3.14'
         pixi-environment: py314
   ```

## Platform Support

### Supported Platforms
- **Linux** (x86_64): Full support for CPU and GPU packages
- **macOS** (Intel and ARM): CPU package only
- **Windows**: Currently NOT supported

### Windows Support Status

**Windows wheels are currently unavailable** due to technical limitations:

1. **RPATH limitation**: Unlike Linux/macOS, Windows doesn't support RPATH for dynamic library loading. PyTorch DLLs must either be:
   - Bundled in the wheel (exceeds PyPI's 100MB limit)
   - Located via system PATH or explicit code

2. **Package size**: Including PyTorch DLLs results in wheels >100MB, exceeding PyPI's size limit for projects without special approval.

We are tracking Windows support via a [pending request to increase the package size limit](#). Users requiring Windows support can:
- Build from source using `pixi run -e py312 build_py`
- Wait for PyPI size limit approval

### Platform-Specific Constraints

PyTorch constraints are platform-aware using pip environment markers:

```python
# Linux uses latest PyTorch
"torch>=2.8.0,<2.9; platform_system == 'Linux'"

# macOS uses older versions
"torch>=2.2.0,<2.3; platform_system == 'Darwin'"
```

The same wheel works on all platforms - pip selects the correct constraint at install time.

## Testing Changes Locally

Generate and inspect configs:
```bash
pixi run generate_pyproject
cat pyproject-pypi-cpu.toml    # Check CPU constraints
cat pyproject-pypi-gpu.toml    # Check GPU constraints
```

Build test wheels:
```bash
pixi run -e py312 clean_dist
pixi run -e py312 build_pypi_wheel
pixi run -e py312 check_pypi
```

## CI Workflow

The CI automatically:
1. Generates config files with default versions
2. Builds wheels for all Python versions and platforms
3. Publishes to PyPI when:
   - Pushing git tags matching `v*` (e.g., `v0.2.0`)
   - Manual workflow dispatch with `publish=true`

## Troubleshooting

**Windows Unicode errors**: Avoid Unicode characters (✓, ✗) in print statements - they fail on Windows CI with `cp1252` encoding.

**CUDA not found in CI**: GPU builds require the shared CUDA setup action (`.github/actions/setup-cuda`). All GPU workflows use this to mock CUDA on CI runners.

**Generated files tracked by git**: Ensure `pyproject-pypi*.toml` is in `.gitignore` - these files are temporary and generated during builds.
