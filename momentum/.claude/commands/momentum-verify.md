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

7. **Diff review checks**: Run the automated diff review script locally:
   `python3 arvr/libraries/momentum/ci/diff_review.py`
   This checks diff summary OSS compliance, build file updates, BUCK/CMake sync, and reviewer assignment — the same checks that CI runs. Fix any issues it reports before continuing.

8. **OSS summary review (AI)**: Read the public portion of the diff summary (everything before `Internal:`) and evaluate it as if you were an external open-source contributor reading a GitHub commit message. Flag anything that:
   - References Meta-internal context (teams, orgs, projects, people, partners)
   - Describes motivation in terms of internal requests or workflows
   - Mentions infrastructure, tools, or systems that only exist inside Meta
   - Would confuse someone who has no knowledge of Meta's internal organization
   - References files or directories that are not exported to GitHub (e.g., `.llms/`, internal BUCK targets)
   Suggest moving such content behind an `Internal:` line. Also check whether `__github__/README.md` or `__github__/CLAUDE.md` needs updating if the diff affects public API surface or documentation.

9. **Doc sync check**: Follow the rules in `.llms/rules/ai_config_maintenance.md`:
   - If directories were added/removed/renamed, check that `.llms/rules/overview.md` structure tree is up to date.
   - If core modules were added/removed, check that `README.md` module list is up to date.

10. **Benchmarks**: Follow `.llms/rules/benchmarking.md` to determine if benchmarks should be run based on which modules were changed. Run the relevant benchmarks and report results.

11. **Lint**: Run `arc lint -a` on all changed files (by path, not by revision — so they get fully linted). Then run `arc lint -e extra` for comprehensive slower linters (CLANGTIDY, etc.). This runs last because earlier steps may modify code.

12. **Report**: Summarize all results (build, test, downstream deps, file checks, OSS checks, doc sync, benchmarks, lint).
