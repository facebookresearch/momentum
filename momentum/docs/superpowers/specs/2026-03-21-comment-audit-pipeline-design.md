# Comment Audit Pipeline for AI Readability

**Date:** 2026-03-21
**Goal:** Optimize code comments across momentum/ and pymomentum/ for AI agent consumption.

## Problem

A 44-commit stack auditing comments across 44 folders exists but cannot be landed efficiently:

- **Stack structure** forces sequential landing — a conflict in commit N blocks N+1 through 44
- **Staleness** — many commits have conflicts with upstream changes
- **No parallelism** — single eden worktree limits concurrent work
- **Context drift** — Claude stops following audit rules on large/complex files (leaves filler comments it should remove)
- **Original generation script was lost** — the prompt cannot be re-run as-is

The commits are independent (no cross-folder dependencies), so the stack structure is unnecessary.

## Audit Rules

The audit rules — fix outdated comments, remove filler, add documentation for high-value locations (contracts, units, algorithms, side effects, edge cases, data formats, dependencies), flag code issues with TODOs, and consistency principles — live in **`arvr/libraries/momentum/.llms/rules/comment_style.md`**. That file is the source of truth and is auto-loaded for any C++ file edit; consult it for the full rule text and examples. The original inline rules in this spec have moved there to avoid duplication.

Doxygen format conventions (use `///` over `/** */`, `@param`/`@return`, no `@brief`) live in `arvr/libraries/momentum/.llms/rules/cpp_style.md` Documentation section.

**Scope constraint:** Comment-only changes. No functional code modifications (except adding TODO comments for issues found). pymomentum bindings are not affected by comment-only changes in momentum/ — each folder is audited independently.

## Design

### Architecture

```
                    +------------------+
                    |     Tracker      |
                    | (tracker.md)     |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |  Triage Phase   |          |  Audit Phase    |
     |  (sequential)   |          |  (parallel via  |
     |                 |          |   background    |
     |                 |          |   Agent tool)   |
     +--------+--------+          +--------+--------+
              |                             |
              v                             v
     existing-clean/stale          edit plans on disk
              |                             |
              +-------------+---------------+
                            |
                   +--------v--------+
                   | Verification    |
                   | (parallel via   |
                   |  background     |
                   |  Agent tool)    |
                   +--------+--------+
                            |
                   +--------v--------+
                   | Commit & Submit |
                   | (sequential,    |
                   |  sl goto trunk  |
                   |  between each)  |
                   +--------+--------+
                            |
                   +--------v--------+
                   | Land            |
                   | (after review)  |
                   +-----------------+
```

### Component 1: Tracker

File: `arvr/libraries/momentum/.llms/comment-tasks/tracker.md`

The tracker lives **in the repo** so progress survives when the long-lived audit-infrastructure commit is rebased onto trunk. It belongs to the same local commit as this design spec; amend updates into that commit rather than committing them separately.

A manifest tracking every folder's status:

```markdown
| Folder | Status | Diff | Notes |
|--------|--------|------|-------|
| common/ | existing-clean | D96091113 | Rebases cleanly, needs verification |
| math/ | existing-stale | — | Conflicts in quaternion.h |
| solver/ | pending | — | Not yet audited |
```

**Statuses:**

| Status | Meaning | Next action |
|--------|---------|-------------|
| `existing-clean` | Existing commit rebases cleanly onto trunk | Run verification |
| `existing-stale` | Existing commit has rebase conflicts | Fresh audit |
| `pending` | No existing work | Fresh audit |
| `audited` | Edit plan produced and saved to disk | Run verification |
| `needs-rework` | Verification failed; issues documented in plan file | Re-audit with issues as input |
| `verified` | Verification pass confirmed edit plan quality | Commit and submit |
| `submitted` | Diff submitted as independent draft | Wait for review / check status |
| `landed` | Diff has landed | Done |

A folder can cycle `audited → needs-rework → audited` at most 2 times. After 2 failed verification cycles, escalate to the user for manual review of the edit plan.

### Component 2: Triage (one-time, sequential)

For each of the 44 existing commits:

1. Attempt to rebase the commit onto trunk independently (cherry-pick style, not stack rebase)
2. If clean → status `existing-clean`
3. If conflicts → status `existing-stale`, record which files conflict
4. Undo the rebase attempt (restore worktree to trunk)

`existing-clean` folders still go through verification (Component 4) because the original audit had known context drift issues. Verification for these operates on the commit diff (`sl diff -c <commit>`) rather than an edit plan file. If verification passes, the commit is used as-is (no re-audit needed). If verification fails, the folder transitions to `pending` for a fresh audit from scratch — no attempt to patch the existing commit.

`existing-stale` folders are treated the same as `pending` — fresh audit from scratch.

This must be sequential because it modifies the worktree.

### Component 3: Audit (parallelizable)

Dispatch Claude subagents via the `Agent` tool with `run_in_background: true`. Each subagent:

1. Reads all files in the folder (read-only — no worktree modifications)
2. Applies the 5 audit rules
3. Produces a structured edit plan saved to `~/.claude/projects/-home-yutingye-fbsource-scratch-arvr-libraries-momentum/comment-audit/plans/<folder-name>.md`

Multiple background agents can run concurrently since they only read files.

#### Edit Plan Format

Edit plans use **semantic descriptions anchored by code context**, not line numbers. This makes them robust to upstream trunk changes.

```markdown
# Edit Plan: common/

## File: common/types.h

### Edit 1: Remove filler comment
- **Location:** Above `struct Foo`, inside namespace `momentum`
- **Action:** Remove
- **Current comment:**
  ```cpp
  // Foo struct
  ```
- **Reason:** Restates the struct name; no information added

### Edit 2: Add documentation
- **Location:** Method `Bar::compute()` in class `Bar`
- **Action:** Add above function
- **New comment:**
  ```cpp
  /// Computes the weighted average of joint positions, skipping
  /// joints with zero weight. Returns identity if all weights are zero.
  ```
- **Reason:** Non-obvious fallback behavior undocumented

### Edit 3: Fix inaccurate comment
- **Location:** Above `calculateBlendWeights()`, comment mentioning "linear interpolation"
- **Action:** Replace
- **Current comment:**
  ```cpp
  // Uses linear interpolation between keyframes
  ```
- **New comment:**
  ```cpp
  // Uses spherical linear interpolation (slerp) for rotations,
  // linear interpolation for translations.
  ```
- **Reason:** Comment says linear but code uses slerp for rotations
```

**Anchoring convention:** Locations are described by function/class/struct name and namespace context — never by line number alone. Include enough surrounding context (e.g., "above `Bar::compute()` in class `Bar`") that the location is unambiguous even after upstream changes.

**Staleness tracking:** Each edit plan file records the trunk revision it was generated against in its header:

```markdown
# Edit Plan: common/
**Trunk revision:** abc123def456
```

#### Chunking for Large Files

Files over ~300 lines are processed in chunks of ~200-300 lines. The chunking threshold should be calibrated empirically — start at 300 and adjust if drift is still observed. The subagent:

1. Reads the full file to understand structure
2. Processes chunks sequentially, keeping the audit rules in the prompt throughout
3. Produces a single unified edit plan for the file

### Component 4: Verification (parallelizable)

A separate Claude session (fresh context, not the audit session) reads:

1. The edit plan from `plans/<folder-name>.md`
2. The original source files

**Verification checklist:**

- [ ] Were all filler comments identified and removed? (Scan for remaining constructor labels, restatements, section banners — see Rule 2 patterns)
- [ ] Are added comments accurate — do they match what the code actually does?
- [ ] Do added comments target high-value locations? (Contracts, units, algorithms, side effects — see Rule 3 priority list)
- [ ] Were any non-comment changes introduced? (Only comment changes and TODO additions are allowed)
- [ ] Are TODO comments actionable and correctly flagging real issues? (Not style opinions or wishlists — see Rule 4)
- [ ] Doxygen format: uses `///` not `/** */`, uses `@param`/`@return` not `\param`/`\return`, no `@brief`
- [ ] No substantive comments were incorrectly flagged as filler (false-positive removal check — see Rule 2 "keep these")
- [ ] Terminology matches the codebase (no synonym drift — see Consistency principles)
- [ ] Comment density matches surrounding code style

**On pass:** Status → `verified`.
**On fail:** Status → `needs-rework`. Issues are documented in the edit plan file itself as a `## Verification Issues` section at the top, so the next audit session sees them.

Verification agents are dispatched via `Agent` tool with `run_in_background: true`, same as audit agents.

### Component 5: Commit & Submit (sequential, user-approved)

For each `verified` folder:

1. `sl goto trunk` — ensure we're on a clean trunk
2. Apply the edit plan to the worktree (Claude reads the plan and makes the edits)
3. `sl commit` with the standard commit message format
4. `jf submit --draft` as an independent diff
5. `sl goto trunk` — reset to trunk for the next folder
6. Update tracker with diff number, status → `submitted`

This creates independent commits, each rooted on trunk, avoiding an accidental stack.

This is sequential (single worktree) and requires user approval for each `jf submit`.

**Submission cadence:** Submit in waves of 5-8 diffs. Adjust wave size based on reviewer throughput after the first wave.

**Staleness check before applying:** Before applying an edit plan, verify that the target files haven't changed since the plan was created. If files have changed, re-run verification on the plan against the current file contents. If verification fails, re-audit.

### Component 6: Landing & Status Checks

- Claude checks diff status via `jf status` or Phab queries
- After reviewer approval, land the diff
- Update tracker status → `landed`
- Individual diffs can be reverted independently since they are not stacked

## Session Workflow

### Simplified workflow (current)

This is the workflow actually used today. The triage / edit-plan / verification phases above are preserved as design reference but not exercised in this mode.

1. User asks: "audit the next N folders" (or names specific folders).
2. Claude opens `arvr/libraries/momentum/.llms/comment-tasks/tracker.md` and picks the next N rows with status `pending`.
3. For each folder, dispatch a background `Agent` (rooted on master) that:
   - Reads `arvr/libraries/momentum/.llms/rules/comment_style.md` (auto-loaded for C++ files).
   - Reads any per-folder hints in `.llms/comment-tasks/<folder>.md`.
   - Edits comments in that folder per the rules.
   - Commits the change as one diff (`[Momentum] Comment audit: <folder>`) and runs `jf submit --draft`.
4. Claude updates each row in the tracker to `submitted` with the diff number.
5. Claude amends the tracker change into this local audit-infrastructure commit so progress survives the next rebase. Do not create a separate commit for tracker updates.
6. After review and landing, the user (or a follow-up session) flips rows to `landed`.

Folders are independent — different sessions can pick different rows in any order. Subagents work in parallel safely as long as no two pick the same row.

### Full pipeline workflow (design reference)

When auditing the entire backlog at once with verification gates:

1. User starts a session: "continue the comment audit"
2. Claude reads the tracker file
3. Claude picks the next action based on current state:
   - If triage hasn't been done → run triage (sequential)
   - If folders need auditing → dispatch background audit agents (parallel)
   - If edit plans need verification → dispatch background verification agents (parallel)
   - If verified folders are ready → commit and submit (sequential, user approves)
   - If submitted diffs need status checks → check and update tracker
4. Claude updates the tracker before session ends

## File Structure

In-repo (committed to the long-lived audit-infrastructure commit):

```
arvr/libraries/momentum/
  .llms/
    rules/
      comment_style.md            # Audit rules (source of truth)
    comment-tasks/
      tracker.md                  # Status manifest (this is what to update each session)
      character.md                # Per-folder hints (only some folders have these)
      character_solver.md
      marker_tracking.md
      math.md
      simd.md
      solver.md
  docs/superpowers/specs/
    2026-03-21-comment-audit-pipeline-design.md  # This file
```

Edit plans (Component 3) are only used in the full pipeline; the simplified per-session workflow audits and submits in one session without persisting plans.

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Edit plans go stale | Semantic anchoring (not line numbers); staleness check before applying |
| Verification misses drift issues | Verification agent gets fresh context with rules front-loaded; explicit checklist |
| Verification loops forever | Max 2 rework cycles per folder, then escalate to user |
| Accidental commit stacking | Explicit `sl goto trunk` between each commit |
| Large review backlog | Submit in waves; adjust wave size based on reviewer throughput |
| Landed diff causes incorrect docs | Each diff is independent and can be reverted individually |
