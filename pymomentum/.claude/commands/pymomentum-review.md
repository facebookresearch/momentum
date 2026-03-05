---
description: Review a pymomentum diff against project rules
---
Review a pymomentum diff against project rules.

If $ARGUMENTS is provided, treat it as a diff number (e.g., D{number}). Otherwise, review the current working directory changes.

Do the following:

1. **Load the diff**: Read the diff details including summary, test plan, and raw diff content.

2. **Read project rules**: Read the relevant rules files:
   - `.llms/rules/overview.md` — commit format, reviewer, build commands, relationship to momentum
   - `.llms/rules/code_style.md` — pybind11 conventions, stubs, Sphinx docs, conditional test skipping
   - `arvr/libraries/momentum/.llms/rules/cpp_style.md` — C++ style (inherited)
   - `arvr/libraries/momentum/.llms/rules/oss_maintenance.md` — OSS export, diff summary, banned words

3. **Diff summary review**:
   - Check that the public portion of the summary contains no internal references. Follow the guidance in `oss_maintenance.md`.
   - Check that internal context is behind `Internal:` on its own line.
   - Check that the title follows the commit format in `overview.md` (`[PyMomentum] ...`).
   - Check that `Reviewers: momentum` is set.

4. **Code review**:
   - Check changed C++ code against `cpp_style.md` (naming, docs, headers, error handling).
   - Check pybind11 code against `code_style.md` (snake_case args, Eigen types, `py::ssize_t`, quaternions).
   - Check that new bindings have Sphinx docstrings.
   - Check that `.pyi` stubs are regenerated if bindings changed.
   - Check that new files are added to `build_variables.bzl`.
   - Check BUCK/CMake sync if BUCK was modified.

5. **Report**: Provide a structured review with findings organized by severity (must-fix, should-fix, nit). Include specific file:line references.
