# Momentum

[![CI Windows][ci-windows-badge]][ci-windows]
[![CI macOS][ci-macos-badge]][ci-macos]
[![CI Ubuntu][ci-ubuntu-badge]][ci-ubuntu]
[![Publish Website][website-badge]][website]

[ci-windows-badge]: https://github.com/facebookresearch/momentum/actions/workflows/ci_windows.yml/badge.svg
[ci-windows]: https://github.com/facebookresearch/momentum/actions/workflows/ci_windows.yml
[ci-macos-badge]: https://github.com/facebookresearch/momentum/actions/workflows/ci_macos.yml/badge.svg
[ci-macos]: https://github.com/facebookresearch/momentum/actions/workflows/ci_macos.yml
[ci-ubuntu-badge]: https://github.com/facebookresearch/momentum/actions/workflows/ci_ubuntu.yml/badge.svg
[ci-ubuntu]: https://github.com/facebookresearch/momentum/actions/workflows/ci_ubuntu.yml
[website-badge]: https://github.com/facebookresearch/momentum/actions/workflows/publish_website.yml/badge.svg
[website]: https://github.com/facebookresearch/momentum/actions/workflows/publish_website.yml

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

```bash
# Python (PyPI)
pip install pymomentum-cpu      # CPU version
pip install pymomentum-gpu      # GPU version with CUDA

# Python (Conda/Pixi)
pixi add pymomentum             # Auto-detects GPU/CPU
conda install -c conda-forge pymomentum

# C++ (Conda/Pixi only)
pixi add momentum-cpp
conda install -c conda-forge momentum-cpp
```

**📦 Browse packages:** [PyPI](https://pypi.org/search/?q=pymomentum) • [conda-forge](https://anaconda.org/conda-forge/momentum) • [prefix.dev](https://prefix.dev/channels/conda-forge/packages/momentum)

### Quick Example

```bash
# Install and run
pip install pymomentum-cpu
python -c "import pymomentum as pm; print(pm.__version__)"
```

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
