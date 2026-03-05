---
description: Full pre-diff verification for pymomentum
---
Full pre-diff verification for pymomentum. Runs build, tests, stubs, and checks.

Do the following steps in order. Stop and report at the first failure.

1. **Read build modes**: Read `.llms/rules/overview.md` to get the correct build mode (pymomentum uses `@fbcode//mode/opt` for all platforms).

2. **Regenerate stubs**: If any pybind11 bindings were changed (new classes, functions, or signature changes), regenerate the affected stub files. Follow `.llms/rules/code_style.md` for the stubgen procedure.

3. **Build**: Build all of pymomentum:
   `buck build @fbcode//mode/opt fbsource//arvr/libraries/pymomentum/...`

4. **Test**: Run all pymomentum tests:
   `buck test @fbcode//mode/opt fbsource//arvr/libraries/pymomentum/...`

5. **Downstream dependency check**: Follow the downstream dependency testing procedure in `.llms/rules/code_style.md`. Search for direct dependents of changed targets and build/test the most obvious ones.

6. **Build file check**: Follow `.llms/rules/code_style.md` — verify `build_variables.bzl` is updated for new files, BUCK and `__github__/CMakeLists.txt` are in sync.

7. **Documentation check**: Follow `.llms/rules/code_style.md` — verify all new bindings have Sphinx-format docstrings with `:param:`, `:return:`, and cross-references.

8. **OSS checks**: Follow `arvr/libraries/momentum/.llms/rules/oss_maintenance.md` — check for banned words, diff summary rules, and `__github__/` sync (pymomentum shares momentum's shipit config).

9. **Lint**: Run `arc lint -a` on all changed files (by path). Then run `arc lint -e extra` for comprehensive slower linters. This runs last because earlier steps may modify code.

10. **Report**: Summarize all results (stubs, build, test, downstream deps, file checks, docs, OSS, lint).
