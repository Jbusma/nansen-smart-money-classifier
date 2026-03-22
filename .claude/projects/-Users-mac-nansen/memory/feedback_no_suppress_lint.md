---
name: no-suppress-lint-errors
description: Never suppress lint errors with ignore rules — fix the actual code instead
type: feedback
---

Don't fix lint errors by suppressing them with ignore rules in config. Fix the actual code.

**Why:** User considers this lazy and not a real fix. Even if it's ML convention (e.g. uppercase X for matrices), if the linter flags it, rename the variables to be lowercase.

**How to apply:** When lint errors come up, always fix the code itself. Never add rules to ruff ignore, noqa comments, or similar suppression mechanisms unless the user explicitly asks for it.
