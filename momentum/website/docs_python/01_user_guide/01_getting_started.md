---
sidebar_position: 1
---

# Getting Started

This page guides you through the process of installing and using PyMomentum, the Python bindings for Momentum.

## Installing PyMomentum

PyMomentum binary builds are available for Windows, macOS, and Linux via [Pixi](https://prefix.dev/) or the Conda package manager.

### [Pixi](https://prefix.dev/channels/conda-forge/packages/momentum)

```bash
# Auto-detects GPU/CPU
pixi add pymomentum

# With specific backend
pixi add pymomentum-gpu  # or pymomentum-cpu
```

### [Conda](https://anaconda.org/conda-forge/momentum)

```bash
# Replace 'pixi add' with 'conda install -c conda-forge'
conda install -c conda-forge pymomentum
conda install -c conda-forge pymomentum-gpu  # or pymomentum-cpu
```

## Building PyMomentum from Source

### Prerequisite

Complete the following steps only once:

1. Install Pixi by following the instructions on the [website](https://prefix.dev/).

1. Clone the repository and navigate to the root directory:

   ```bash
   git clone https://github.com/facebookresearch/momentum
   cd momentum
   ```

   Ensure that all subsequent commands are executed in the project's root directory unless specified otherwise.

### Build and Test

- Build the project with the following command (note that the first run may take a few minutes as it installs all dependencies):

   ```bash
   pixi run build
   ```

- Run the Python tests with:

   ```bash
   pixi run test_py
   ```

### Building from Source for Custom Environments

If you need to use PyMomentum in your own conda environment with specific dependency versions (e.g., particular PyTorch or CUDA versions), you have two options:

#### Option 1: Add your code/packages to the pixi environment (recommended for quick testing)

For quick testing and development, you can add your packages or code to the existing pixi environment by modifying `pixi.toml`:

```bash
# Add a package dependency to pixi.toml
pixi add your-package-name

# Or manually edit pixi.toml to add your custom dependencies
```

This approach avoids the complexity of managing dependencies manually and ensures compatibility with the pixi-managed environment.

#### Option 2: Build in your custom conda environment

For production use or when you need specific dependency versions:

1. Activate your target conda environment:

   ```bash
   conda activate your_environment_name
   ```

2. Navigate to the momentum source directory:

   ```bash
   cd path/to/momentum
   ```

3. Install required dependencies:

   When building with `pip install .` in your custom environment, you need to manually install PyMomentum's dependencies. See the `dependencies` section in [pixi.toml](https://github.com/facebookresearch/momentum/blob/main/pixi.toml) for the complete list of required packages. Key dependencies include:

   ```bash
   # Install core dependencies (adjust versions as needed)
   conda install -c conda-forge pytorch eigen ceres-solver
   ```

4. Build and install PyMomentum:

   ```bash
   pip install .
   ```

   This builds and installs PyMomentum using the dependencies in your activated environment.

#### Troubleshooting

**I built the latest unreleased version with `pixi run build_py`. Can I use it in my conda environment?**

No. `pixi run build_py` builds within pixi's isolated environment. To use the latest version in your conda environment, activate it and run `pip install .` as shown above.

**Finding available PyMomentum versions:**

```bash
pixi search -c conda-forge pymomentum
# or
conda search -c conda-forge pymomentum
```

**Installing specific versions with CUDA constraints:**

```bash
# Example: Install version 0.1.74 with CUDA 12.9
conda install -c conda-forge pymomentum=0.1.74="cuda129*"
```

**Dependency conflicts or version incompatibility:**

- Ensure your environment has compatible PyTorch, CUDA, and build dependencies (`scikit-build-core`, `pybind11`, CMake, C++ compiler)
- PyMomentum links against PyTorch's CUDA libraries, so CUDA versions must match
- For custom PyTorch/CUDA combinations not available on conda-forge, build from source (as described above) or contribute a backport to the [momentum-feedstock repository](https://github.com/conda-forge/momentum-feedstock)
