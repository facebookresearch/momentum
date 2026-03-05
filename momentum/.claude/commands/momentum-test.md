---
description: Run momentum tests
---
Run momentum tests. Auto-detects platform.

Do the following:
1. Read the build modes from `.llms/rules/overview.md` to get the correct mode for the current platform.
2. If $ARGUMENTS is provided, treat it as a specific test target and run:
   `buck test @BUILD_MODE //arvr/libraries/momentum:$ARGUMENTS`
3. If no arguments, run all tests:
   `buck test @BUILD_MODE //arvr/libraries/momentum/...`
4. Report the result (pass/fail counts, any failures with details).
