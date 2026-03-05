---
description: Full pre-diff verification for momentum
---
Full pre-diff verification for momentum. Runs build, tests, and checks.

Do the following steps in order. Stop and report at the first failure.

1. **Read build modes**: Read `.llms/rules/overview.md` to get the correct build mode for the current platform.

2. **Build**: Build all of momentum using the platform-appropriate mode:
   `buck build @BUILD_MODE //arvr/libraries/momentum/...`

3. **Test**: Run all momentum tests:
   `buck test @BUILD_MODE //arvr/libraries/momentum/...`

4. **Downstream dependency check**: Follow the downstream dependency testing procedure in `.llms/rules/code_style.md`. Build and test the most obvious direct dependents.

5. **Build file check**: Follow the code organization rules in `.llms/rules/code_style.md` to verify that any new files or classes have been registered in the appropriate build files.

6. **BUCK/CMake sync check**: Follow the BUCK/CMake sync rule in `.llms/rules/code_style.md` to verify that dependency changes are reflected in both build systems.

7. **OSS checks**: Follow the rules in `.llms/rules/oss_maintenance.md`:
   - Check for banned words in changed files and the diff summary.
   - Check that the public portion of the diff summary contains no internal URLs, SEV references, customer names, or internal team names. Remind the user to use the `Internal:` section for internal-only context.
   - If changes affect public API surface, module structure, or documentation, check whether `__github__/README.md` or `__github__/CLAUDE.md` needs updating.

8. **Doc sync check**: Follow the rules in `.llms/rules/ai_config_maintenance.md`:
   - If directories were added/removed/renamed, check that `.llms/rules/overview.md` structure tree is up to date.
   - If core modules were added/removed, check that `README.md` module list is up to date.

9. **Benchmarks**: Follow `.llms/rules/benchmarking.md` to determine if benchmarks should be run based on which modules were changed. Run the relevant benchmarks and report results.

10. **Lint**: Run `arc lint -a` on all changed files (by path, not by revision — so they get fully linted). Then run `arc lint -e extra` for comprehensive slower linters (CLANGTIDY, etc.). This runs last because earlier steps may modify code.

11. **Report**: Summarize all results (build, test, downstream deps, file checks, OSS checks, doc sync, benchmarks, lint).
