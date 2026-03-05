---
description: Build the momentum project
---
Build the momentum project. Auto-detects platform.

Do the following:
1. Read the build modes from `.llms/rules/overview.md` to get the correct mode for the current platform.
2. If $ARGUMENTS is provided, treat it as a specific target name and run:
   `buck build @BUILD_MODE //arvr/libraries/momentum:$ARGUMENTS`
3. If no arguments, build everything:
   `buck build @BUILD_MODE //arvr/libraries/momentum/...`
4. Report the result (success or failure with errors).
