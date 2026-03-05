---
description: Build pymomentum
---
Build pymomentum.

Do the following:
1. If $ARGUMENTS is provided, treat it as a specific target name and run:
   `buck build @fbcode//mode/opt fbsource//arvr/libraries/pymomentum:$ARGUMENTS`
2. If no arguments, build everything:
   `buck build @fbcode//mode/opt fbsource//arvr/libraries/pymomentum/...`
3. Report the result (success or failure with errors).
