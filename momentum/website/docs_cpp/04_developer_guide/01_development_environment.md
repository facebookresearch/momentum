---
sidebar_position: 1
---

# Development Environment

## Supported Environments

* **OS**: Windows, Linux, macOS

## Package Manager

Before developing Momentum, it is necessary to install various dependencies. This process can be platform-dependent and tedious. To simplify this, Momentum utilizes [Pixi](https://prefix.dev/).

Pixi facilitates building Momentum in a virtual environment across different platforms (Windows, macOS Intel/ARM, Linux) using consistent command lines.

For those interested, you can examine the `pixi.toml` file to see how dependencies are specified and to explore the available Pixi tasks for Momentum.

:::info

If you choose not to use Pixi, you will need to manually install all dependencies using platform-specific package managers. These typically install dependencies into the system directory. Ensure you have the appropriate package managers installed for your OS: [Homebrew](https://brew.sh/) for macOS, [Vcpkg](https://vcpkg.io/en/) for Windows, and apt for Ubuntu/Debian. After installation, refer to `pixi.toml` for guidance on what and how to install.

:::

## Running Custom Commands in Shell

To execute additional commands in the virtual environment other than the predefined tasks (to see the full tasks: `pixi task list`), such as using CMake directly or running an executable, activate the virtual environment with:

```
pixi shell
```

To exit the virtual environment, simply run:

```
exit
```

### Developing with Microsoft Visual Studio (Windows Only)

To open the project in Visual Studio 2022, use the command:

  ```
  pixi run open_vs
  ```
