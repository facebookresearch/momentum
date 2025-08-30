---
sidebar_position: 1
---

# Getting Started

This page guides you through the process of building Momentum (C++ library) and running the examples.

## Installing Momentum

Momentum C++ binary builds are available for Windows, macOS, and Linux via [Pixi](https://prefix.dev/) or the Conda package manager.

For Windows, please install [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) or greater.

### [Pixi](https://prefix.dev/channels/conda-forge/packages/momentum)

```
pixi add momentum-cpp
```

### [Conda](https://anaconda.org/conda-forge/momentum)

```
conda install -c conda-forge momentum-cpp
```

## Building Momentum from Source

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

- Run the tests with:

  ```
  pixi run test
  ```

To view all available command lines, run `pixi task list`.

### Hello World Example

To run the `hello_world` example:

```
pixi run hello_world
```

Alternatively, you can directly run the executable:

```
# Linux and macOS
./build/hello_world

# Windows
./build/Release/hello_world.exe
```

### Running Other Examples

To run other examples:

```
pixi run glb_viewer --help
```

For more examples, please refer to the [Examples](../examples/viewers) page.

### Clean Up

If you need to start over for any reason:

```
pixi run clean
```

Momentum uses the `build/` directory for CMake builds, and `.pixi/` for the Pixi virtual environment. You can clean up everything by either manually removing these directories or by running the command above.

### FBX Support

Momentum uses OpenFBX to load Autodesk's FBX file format, which is enabled by default. To save files in FBX format, you must install the FBX SDK 2020.3.

#### Linux

The FBX SDK will be automatically installed when you run `pixi run config`, so no additional steps are required.

#### macOS and Windows

You can download it from Autodesk's [website](https://aps.autodesk.com/developer/overview/fbx-sdk) or use direct links ([macOS](https://damassets.autodesk.net/content/dam/autodesk/www/files/fbx202037_fbxsdk_clang_mac.pkg.tgz), [Windows](https://damassets.autodesk.net/content/dam/autodesk/www/files/fbx202037_fbxsdk_vs2019_win.exe)). After installing the SDK, build Momentum from source with `MOMENTUM_BUILD_WITH_FBXSDK=ON` option as:

```
# macOS
MOMENTUM_BUILD_WITH_FBXSDK=ON pixi run <target>

# Windows (Powershell)
$env:MOMENTUM_BUILD_WITH_FBXSDK = "ON"; pixi run <target>

# Windows (cmd)
set MOMENTUM_BUILD_WITH_FBXSDK=ON && pixi run <target>
```

For example, file conversion can be run as follows:

```
# Windows (Powershell)
$env:MOMENTUM_BUILD_WITH_FBXSDK = "ON"; pixi run convert_model -d <input.glb> -o <out.fbx>
```
