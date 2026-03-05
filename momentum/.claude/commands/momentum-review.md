---
description: Review a momentum or pymomentum diff against project rules
---
Review a momentum or pymomentum diff against project rules.

If $ARGUMENTS is provided, treat it as a diff number (e.g., D{number}). Otherwise, review the current working directory changes.

Do the following:

1. **Load the diff**: Read the diff details including summary, test plan, and raw diff content.

2. **Read project rules**: Read the relevant `.llms/rules/` files for the project(s) touched by the diff:
   - `arvr/libraries/momentum/.llms/rules/overview.md` — commit format, reviewer
   - `arvr/libraries/momentum/.llms/rules/code_style.md` — code organization, downstream deps
   - `arvr/libraries/momentum/.llms/rules/cpp_style.md` — C++ style
   - `arvr/libraries/momentum/.llms/rules/oss_maintenance.md` — OSS export, diff summary, banned words
   - `arvr/libraries/pymomentum/.llms/rules/code_style.md` — pybind11 conventions (if pymomentum files touched)

3. **Diff summary review**:
   - Check that the public portion of the summary contains no internal references (team names, internal URLs, task/SEV/diff IDs, internal tools, internal file paths, project codenames). Follow the guidance in `oss_maintenance.md`.
   - Check that internal context is behind `Internal:` on its own line.
   - Check that the title follows the commit format in `overview.md`.
   - Check that `Reviewers: momentum` is set.

4. **Code review**:
   - Check changed code against `cpp_style.md` (naming, docs, headers, error handling).
   - Check that new `.cpp`/`.h` files are added to `build_variables.bzl`.
   - Check that new classes are added to `gen_fwd_input.toml` (momentum only).
   - Check BUCK/CMake sync if BUCK was modified.
   - Check for banned words in the shipit config (read from `oss_maintenance.md`).
   - If pybind11 bindings changed, check for Sphinx docstrings and stub regeneration.

5. **Benchmark recommendation**: Follow `.llms/rules/benchmarking.md` to determine if the changed modules have corresponding benchmarks. If so, note which benchmarks should be run in the report.

6. **Report**: Provide a structured review with findings organized by severity (must-fix, should-fix, nit). Include specific file:line references and benchmark recommendations if applicable.
