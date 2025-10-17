# Contributing to Momentum

We want to make contributing to this project as easy and transparent as possible.

## Quick Start

1. **Fork and clone:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/momentum.git
   cd momentum
   git remote add upstream https://github.com/facebookresearch/momentum.git
   ```

2. **Build:** (dependencies auto-install)
   ```bash
   pixi run build        # C++ library
   pixi run build_py     # Python bindings
   ```

3. **Test:**
   ```bash
   pixi run test         # C++ tests
   pixi run test_py      # Python tests
   ```

4. **Make changes:**
   ```bash
   git checkout -b feature/your-feature
   # Make your changes
   pixi run lint         # Format code
   pixi run test         # Verify tests pass
   ```

5. **Submit:**
   ```bash
   git commit -m "Your message"
   git push origin feature/your-feature
   # Open PR on GitHub
   ```

## Prerequisites

- **Git**
- **Pixi** package manager ([install guide](https://prefix.dev/docs/pixi/overview))

## Building

Dependencies are automatically installed when running any Pixi task.

### C++ Development

```bash
pixi run build         # Release build
pixi run build_dev     # Debug build
pixi run test          # Run tests
```

**Platform-specific:**
- Windows: `pixi run open_vs` opens Visual Studio
- Linux: FBX SDK auto-installs during build

View all tasks: `pixi task list`

### Python Development

```bash
pixi run build_py      # Build Python bindings
pixi run test_py       # Run Python tests
pixi run doc_py        # Build Python docs
```

**Note:** Rebuild after C++/pybind11 changes. Pure Python changes don't require rebuilding.

## Code Style

### C++
- See [Style Guide](https://facebookresearch.github.io/momentum/docs_cpp/developer_guide/style_guide)
- Format: `pixi run lint`
- Check: `pixi run lint-check`

### Python
- Follow [PEP 8](https://pep8.org/)
- Use type hints and docstrings
- Format: `black pymomentum/`

## Testing

```bash
pixi run test              # C++ tests
pixi run test_py           # Python tests
pixi run test_verbose      # C++ verbose output
pixi run test_py_verbose   # Python verbose output
```

Run specific tests:
```bash
pixi shell
pytest pymomentum/test/test_character.py -v
exit
```

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

### How Pull Requests Work

Note: Pull requests are not imported into the GitHub repository in the usual way. There is an internal Meta repository that is the "source of truth" for this project. The GitHub repository is generated *from* the internal Meta repository. Pull requests must first be imported into the internal Meta repository, where they are reviewed. Once approved, changes are automatically reflected from the internal Meta repository back to GitHub. This is why you won't see your PR being directly merged, but you will still see your changes in the repository once it reflects the imported changes.

### CI Process

PRs automatically run:
- Builds on Ubuntu, macOS, Windows
- All C++ and Python tests
- Code formatting checks
- Documentation builds

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Troubleshooting

**Build fails:** `pixi run clean && pixi run build`

**Format errors:** `pixi run lint`

**Test fails:** Use `pixi run test_verbose` for details

**Merge conflicts:**
```bash
git fetch upstream
git rebase upstream/main
# Resolve conflicts
git rebase --continue
git push origin feature/your-feature --force
```

## Development Tips

### Python Fast Iteration
- Pure Python changes: Edit and test directly (no rebuild)
- C++ binding changes: `pixi run build_py`

### Debug Bindings
```bash
pixi shell
gdb python
run pymomentum/test/test_character.py
```

## Code Organization

```
momentum/             # C++ core library
pymomentum/          # Python bindings
├── pymomentum/      # Pure Python code
├── test/            # Python tests
└── bindings/        # pybind11 bindings
```

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

When filing a bug, include:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, compiler, Python version)
- Minimal example code

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe disclosure of security bugs. In those cases, please go through the process outlined on that page and do not file a public issue.

## Getting Help

- **Questions:** [GitHub Discussions](https://github.com/facebookresearch/momentum/discussions)
- **Bugs:** [Issue Tracker](https://github.com/facebookresearch/momentum/issues)
- **Documentation:** [Project Website](https://facebookresearch.github.io/momentum/)

## License

By contributing to this project, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.

---

Thank you for contributing to Momentum!
