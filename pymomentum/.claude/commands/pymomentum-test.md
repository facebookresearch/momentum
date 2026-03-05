---
description: Run pymomentum tests
---
Run pymomentum tests.

Do the following:
1. If $ARGUMENTS is provided, treat it as a specific test target and run:
   `buck test @fbcode//mode/opt fbsource//arvr/libraries/pymomentum:$ARGUMENTS`
2. If no arguments, run all tests:
   `buck test @fbcode//mode/opt fbsource//arvr/libraries/pymomentum/...`
3. Report the result (pass/fail counts, any failures with details).
