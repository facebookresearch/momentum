---
sidebar_position: 1
---

# Getting Started

This page guides you through the process of installing and using PyMomentum, the Python bindings for Momentum.

## Installing PyMomentum

PyMomentum binary builds are available for Windows, macOS, and Linux via [Pixi](https://prefix.dev/) or the Conda package manager.

### [Pixi](https://prefix.dev/channels/conda-forge/packages/momentum)

```
# Auto-detects GPU/CPU
pixi add pymomentum

# With specific backend
pixi add pymomentum-gpu  # or pymomentum-cpu
```

### [Conda](https://anaconda.org/conda-forge/momentum)

```
# Replace 'pixi add' with 'conda install -c conda-forge'
conda install -c conda-forge pymomentum
conda install -c conda-forge pymomentum-gpu  # or pymomentum-cpu
```

## Building PyMomentum from Source

### Prerequisite

Complete the following steps only once:

1. Install Pixi by following the instructions on https://prefix.dev/

1. Clone the repository and navigate to the root directory:

   ```
   git clone https://github.com/facebookresearch/momentum
   cd momentum
   ```

   Ensure that all subsequent commands are executed in the project's root directory unless specified otherwise.

### Build and Test

- Build the project with the following command (note that the first run may take a few minutes as it installs all dependencies):

  ```
  pixi run build
  ```

- Run the Python tests with:

  ```
  pixi run test_py
  ```
