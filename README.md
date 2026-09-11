# Momentum

[![CI Windows][ci-windows-badge]][ci-windows]
[![CI macOS][ci-macos-badge]][ci-macos]
[![CI Ubuntu][ci-ubuntu-badge]][ci-ubuntu]
[![Publish Website][website-badge]][website]
[![PyPI Wheels][pypi-wheels-badge]][pypi-wheels]

[ci-windows-badge]: https://github.com/facebookresearch/momentum/actions/workflows/ci_windows.yml/badge.svg
[ci-windows]: https://github.com/facebookresearch/momentum/actions/workflows/ci_windows.yml
[ci-macos-badge]: https://github.com/facebookresearch/momentum/actions/workflows/ci_macos.yml/badge.svg
[ci-macos]: https://github.com/facebookresearch/momentum/actions/workflows/ci_macos.yml
[ci-ubuntu-badge]: https://github.com/facebookresearch/momentum/actions/workflows/ci_ubuntu.yml/badge.svg
[ci-ubuntu]: https://github.com/facebookresearch/momentum/actions/workflows/ci_ubuntu.yml
[website-badge]: https://github.com/facebookresearch/momentum/actions/workflows/publish_website.yml/badge.svg
[website]: https://github.com/facebookresearch/momentum/actions/workflows/publish_website.yml
[pypi-wheels-badge]: https://github.com/facebookresearch/momentum/actions/workflows/publish_to_pypi.yml/badge.svg
[pypi-wheels]: https://github.com/facebookresearch/momentum/actions/workflows/publish_to_pypi.yml

Momentum provides foundational algorithms for human kinematic motion and
numerical optimization solvers to apply human motion in various applications.

<p align="center">
  <img src="momentum/website/static/img/momentum_1.png" width="30%" alt="Forward and Inverse Kinematics with Interpretable Parameterization" />
  <img src="momentum/website/static/img/momentum_3.png" width="30%" alt="RGBD Body Tracking Solver" />
  <img src="momentum/website/static/img/momentum_4.png" width="30%" alt="Monocular RGB Body Tracking Solver" />
</p>

## Quick Start

### Installation

Pre-built binaries are available for Windows, macOS, and Linux:

Install full conda/pixi packages -- recommended
```bash
# Install full python package from pixi, including modules with cpp torch dependencies
pixi add pymomentum

# Install full python package from conda, including modules with cpp torch dependencies
conda install -c conda-forge pymomentum

# Install cpp momentum package from pixi, including modules with cpp torch dependencies
pixi add momentum-cpp

# Install cpp momentum package from conda, including modules with cpp torch dependencies
conda install -c conda-forge momentum-cpp
```

Install PyPI wheels -- these are experimental ⚠️
```bash
# Install the NumPy/native core without PyTorch
pip install pymomentum-core

# Install core package with optional visualization helpers
pip install "pymomentum-core[viser,rerun]"

# Add diff_geometry, Torch helpers, and differentiable solvers for CPU PyTorch
pip install pymomentum-cpu

# Install full package including diff_geometry and differentiable solver modules, linked against CUDA PyTorch
pip install "torch>=2.8,<2.9" --index-url https://download.pytorch.org/whl/cu129
pip install pymomentum-gpu
```

The same layered packages are available from conda-forge. `pymomentum` remains a
compatibility package that installs the CPU or GPU stack selected by the solver.

The `pymomentum-core` package owns the NumPy/native modules `geometry`,
`solver2`, `marker_tracking`, `axel`, `camera`, and `renderer`. It does not
install PyTorch. `pymomentum-cpu` and `pymomentum-gpu` are mutually exclusive
add-ons: each installs the Torch helpers plus `diff_geometry` and `solver`, and
depends on a compatible `pymomentum-core` release.

> ⚠️ **PyPI support is experimental.** For the most stable experience, we recommend using Conda or Pixi.

**📦 Browse packages:** [conda-forge](https://anaconda.org/conda-forge/momentum) • [prefix.dev](https://prefix.dev/channels/conda-forge/packages/momentum) • [PyPI](https://pypi.org/search/?q=pymomentum)

### Building from Source

```bash
git clone https://github.com/facebookresearch/momentum
cd momentum
pixi run build      # Builds C++ library and Python bindings
pixi run test       # Runs tests
pixi run hello_world  # Runs example
```

**For detailed instructions**, see the comprehensive guides on our website:
- 📘 [**Python Getting Started**](https://facebookresearch.github.io/momentum/pymomentum/user_guide/getting_started) - Installation, building from source, troubleshooting
- 📗 [**C++ Getting Started**](https://facebookresearch.github.io/momentum/docs/user_guide/getting_started) - Full build instructions, FBX support, examples

## 📖 Documentation

Visit our [**documentation website**](https://facebookresearch.github.io/momentum/) for comprehensive guides, examples, and API references:

- 🐍 [**Python API Reference**](https://facebookresearch.github.io/momentum/python_api_doc/index.html) - Complete Python API documentation
- ⚙️ [**C++ API Reference**](https://facebookresearch.github.io/momentum/doxygen/index.html) - Complete C++ API documentation

## Contributing

Check our [contributing guide](CONTRIBUTING.md) to learn about how to contribute
to the project.

## License

Momentum is licensed under the MIT License. A copy of the license
[can be found here.](LICENSE)
