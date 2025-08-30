---
sidebar_position: 2
---

# Creating Your Applications

This page guides you on how to create your own project that depends on the Momentum library.

:::info

We continue to use Pixi's virtual environment for building your project with Momentum and its dependencies. For alternative setups, please refer to the [Development Environment](../developer_guide/development_environment) page.

:::

## Install Momentum

First, install Momentum in the virtual environment by running:

```
pixi run install
```

This command builds (in Release mode) and installs Momentum to `.pixi/envs/default/{include,lib,share}/` (Windows may have slightly different path). The necessary environment variables are set so that CMake can find Momentum (and other dependencies) using the environment variables in the virtual environment.

## Writing Source Code

Create a new file named `main.cpp` in your project root with the following content:

```cpp
#include <momentum/math/mesh.h>

using namespace momentum;

int main() {
  auto mesh = Mesh();
  mesh.updateNormals();
  return EXIT_SUCCESS;
}
```

## Writing CMake Script

Create a `CMakeLists.txt` file in the same directory as `main.cpp`.

To add momentum to your CMake project, first find the momentum package using the
`find_package` function and then add the appropriate `momentum::<target>` as a
dependency to your library or executable. For example, if you want to use the
character functionality from momentum, you would add `momentum::character` as a
dependency:

```cmake
cmake_minimum_required(VERSION 3.16.3)

project(momentum)

find_package(momentum CONFIG REQUIRED)
add_executable(hello_world main.cpp)
target_link_libraries(hello_world PRIVATE momentum::math)
```

Refer to the example project located at `momentum/examples/hello_world/` for the complete source code.

If you are developing a library that depends on Momentum:

```cmake
add_library(my_lib SHARED my_lib.hpp my_lib.cpp)  # shared
add_library(my_lib STATIC my_lib.hpp my_lib.cpp)  # static
target_link_libraries(my_lib PUBLIC momentum::math)
```

## Building using CMake

Assuming your project directory now contains:

```
<root>
 - CMakeLists.txt
 - main.cpp
```

For convenience, we assume that your project root is located at `momentum/examples/hello_world/` because this code example is provided in that directory. You can use this working example as a reference, but feel free to adjust the path according to your actual project root.

Here, we assume you are not using Pixi to build your project, but you are still within the Pixi environment for managing dependencies.

To run any command in the virtual environment, use:

```
pixi run <command>
```

Run the native CMake commands in the virtual environment as follows:

To configure the application, run:

```
# Linux and macOS
pixi run cmake -S momentum/examples/hello_world -B momentum/examples/hello_world/build -DCMAKE_BUILD_TYPE=Release

# Windows
pixi run cmake -S momentum/examples/hello_world -B momentum/examples/hello_world/build
```

To build the application, run:

```
# Linux and macOS
pixi run cmake --build momentum/examples/hello_world/build

# Windows
pixi run cmake --build momentum/examples/hello_world/build --config Release
```

## Run the Application

Execute the application with:

```
# Linux and macOS
./momentum/examples/hello_world/build/hello_world

# Windows
momentum/examples/hello_world/build/Release/hello_world.exe
```

## Configuring Your Project with Pixi

If you wish to use Pixi for your project similar to how it's implemented in Momentum, please visit [this website](https://pixi.sh/latest/basic_usage/) for detailed instructions.
