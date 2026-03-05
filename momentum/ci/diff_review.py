#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Automated diff review checks for momentum and pymomentum.

These checks complement existing CI signals — they do NOT duplicate:
- Lint (handled by arc lint / Sandcastle lint signal)
- Build/test (handled by Sandcastle build/test signals)
- Banned words in code (handled by shipit's github-export-checks signal)

This script checks things that existing CI does NOT cover:
- Diff summary OSS compliance (internal URLs, task/SEV refs in public portion)
- build_variables.bzl updates when new source files are added
- BUCK/CMake sync when BUCK dependencies change
- Reviewer field includes momentum
"""

import json
import re
import subprocess
import sys


def get_changed_files():
    """Get list of files changed in the current diff."""
    result = subprocess.run(
        ["hg", "status", "--change", ".", "--no-status"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Failed to get changed files: {result.stderr}")
        sys.exit(1)
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


def _get_phabricator_summary(diff_number):
    """Fetch the diff summary from Phabricator via Conduit API."""
    try:
        result = subprocess.run(
            [
                "arc",
                "call-conduit",
                "differential.query",
                "--",
                f'{{"ids": [{diff_number}]}}',
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        response = data.get("response", [])
        if response:
            return (
                response[0].get("title", "") + "\n\n" + response[0].get("summary", "")
            )
    except (json.JSONDecodeError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_diff_summary():
    """Get the diff summary, preferring Phabricator over local commit message.

    ShipIt uses the Phabricator diff summary (not the local commit message)
    when exporting to GitHub. This function tries to fetch the Phabricator
    summary first, falling back to the local commit message if unavailable.
    """
    # Get local commit message (always needed for reviewer check, diff number)
    result = subprocess.run(
        ["hg", "log", "-r", ".", "-T", "{desc}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    local_message = result.stdout

    # Try to extract diff number and fetch Phabricator summary
    diff_match = re.search(r"Differential Revision:.*D(\d+)", local_message)
    if diff_match:
        diff_number = diff_match.group(1)
        phab_summary = _get_phabricator_summary(diff_number)
        if phab_summary is not None:
            # Append metadata from local message that Phabricator summary
            # may not include (Reviewers, Test Plan, etc.)
            return phab_summary + "\n" + local_message

    return local_message


def check_diff_summary_oss(summary):
    """Check diff title and summary public portion for internal references.

    Shipit's github-export-checks already catches banned words. This check
    catches internal references that shipit does not check: URLs, task IDs,
    SEV IDs, diff IDs, internal tool names, and internal file paths in the
    public portion of the diff title and summary.
    """
    issues = []
    parts = re.split(r"^Internal:", summary, flags=re.MULTILINE)
    public_portion = parts[0]

    # Extract title (first non-empty line) for separate checking
    title = ""
    for line in public_portion.splitlines():
        stripped = line.strip()
        if stripped:
            # Strip bracket tags like [Momentum] that shipit removes
            title = re.sub(r"\[.*?\]\s*", "", stripped)
            break

    # Strip metadata lines that are not exported by shipit
    # (Reviewers, Reviewed By, Differential Revision, Pull Request resolved,
    # Tags, Test Plan, and everything after them are handled separately)
    metadata_pattern = re.compile(
        r"^(?:Reviewers?:|Reviewed By:|Differential Revision:|"
        r"Pull Request resolved:|Tags:|Test Plan:).*",
        re.MULTILINE | re.DOTALL,
    )
    public_portion = metadata_pattern.sub("", public_portion).strip()

    # All pattern groups: (regex, human-readable name, case_insensitive)
    all_patterns = [
        # Internal URLs
        (r"internalfb\.com", "internalfb.com URL", False),
        (r"our\.intern\.facebook\.com", "intern.facebook.com URL", False),
        (r"fb\.workplace\.com", "workplace URL", False),
        (r"bunnylol\b", "bunnylol reference", False),
        # Task/SEV/diff references
        (r"\bT\d{6,}\b", "Task reference (T...)", False),
        (r"\bS\d{5,}\b", "SEV reference (S...)", False),
        (r"\bD\d{6,}\b", "Diff reference (D...)", False),
        # Internal tools and paths
        (
            r"\bbuck\s+(?:build|test|run|query|targets)\b",
            "buck command reference",
            True,
        ),
        (r"\barc\s+(?:lint|diff|land)\b", "arc command reference", True),
        (r"\b(?:fbsource|fbcode)\b", "internal repo path reference", True),
        (r"\bConfigerator\b", "Configerator reference", True),
        (r"\bSandcastle\b", "Sandcastle reference", True),
        (r"\bCitadel\b", "Citadel reference", True),
        (r"\bSkycastle\b", "Skycastle reference", True),
        (r"\bDevmate\b", "Devmate reference", True),
        (r"\.llms/", ".llms/ path reference", True),
        (
            r"\bci_(?:shell_test|skycastle|sandcastle)\b",
            "internal CI macro reference",
            True,
        ),
        (r"\barvr/", "internal file path reference", True),
    ]

    # Split public_portion into body (without title line) to avoid duplicate
    # reports. The title is the first non-empty line of public_portion, so
    # patterns matching in the title would otherwise be reported twice: once
    # for "in public portion" and once for "in diff title".
    lines = public_portion.splitlines()
    body_lines = []
    title_skipped = False
    for line in lines:
        if not title_skipped and line.strip():
            title_skipped = True
            continue
        body_lines.append(line)
    body_without_title = "\n".join(body_lines)

    for pattern, name, case_insensitive in all_patterns:
        flags = re.IGNORECASE if case_insensitive else 0
        if re.search(pattern, body_without_title, flags):
            issues.append(
                f"  {name} in public portion of diff summary. "
                "Move behind 'Internal:' line."
            )
        if title and re.search(pattern, title, flags):
            issues.append(
                f'  {name} in diff title: "{title}". '
                "The title becomes the GitHub commit title."
            )

    return issues


def check_build_variables(changed_files):
    """Check if new .cpp/.h files are added to build_variables.bzl."""
    issues = []
    momentum_prefix = "arvr/libraries/momentum/"
    pymomentum_prefix = "arvr/libraries/pymomentum/"

    new_source_files = []
    build_vars_changed = False

    for f in changed_files:
        if f.startswith(momentum_prefix):
            rel = f[len(momentum_prefix) :]
            if rel == "build_variables.bzl":
                build_vars_changed = True
            elif re.match(r".*\.(cpp|h)$", rel) and not rel.startswith(
                ("__github__/", "ci/", "test/")
            ):
                new_source_files.append(f)
        elif f.startswith(pymomentum_prefix):
            rel = f[len(pymomentum_prefix) :]
            if rel == "build_variables.bzl":
                build_vars_changed = True
            elif re.match(r".*\.(cpp|h)$", rel) and not rel.startswith(
                ("__github__/", "ci/", "cpp_test/")
            ):
                new_source_files.append(f)

    if new_source_files and not build_vars_changed:
        issues.append(
            "  New .cpp/.h files added but build_variables.bzl not updated:\n"
            + "\n".join(f"    - {f}" for f in new_source_files)
        )

    return issues


def _buck_diff_has_oss_relevant_changes(buck_path):
    """Check if BUCK file changes affect targets relevant to the OSS build.

    CI-only macros (ci_shell_test, ci_skycastle, ci_sandcastle) are internal
    and have no CMakeLists.txt counterpart. Changes that only touch these
    targets should not trigger the BUCK/CMake sync warning.
    """
    result = subprocess.run(
        ["hg", "diff", "--change", ".", buck_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # If we can't get the diff, assume it's relevant to be safe.
        return True

    ci_only_patterns = re.compile(
        r"^(?:"
        r"ci_shell_test\(|"
        r"ci_skycastle\(|"
        r"ci_sandcastle\(|"
        r"ci_srcs\b|"
        r"oncall\b|"
        r"name\s*=|"
        r"cmd\s*=|"
        r"load\(.*ci_(?:shell_test|skycastle|sandcastle).*|"
        r'"[^"]*[*][^"]*",?\s*$|'
        r"[)\]\[],?\s*$|"
        r"#"
        r")"
    )

    for line in result.stdout.splitlines():
        # Only look at added/removed lines (not context or headers)
        if not line.startswith(("+", "-")):
            continue
        # Skip diff headers
        if line.startswith(("+++", "---")):
            continue
        # Skip blank added/removed lines
        stripped = line[1:].strip()
        if not stripped:
            continue
        # If any changed line is NOT a CI-only pattern, it's OSS-relevant
        if not ci_only_patterns.match(stripped):
            return True

    return False


def check_buck_cmake_sync(changed_files):
    """Check if OSS-relevant BUCK changes are synced with CMakeLists.txt."""
    issues = []
    prefixes = [
        "arvr/libraries/momentum/",
        "arvr/libraries/pymomentum/",
    ]

    for prefix in prefixes:
        buck_changed = any(f == prefix + "BUCK" for f in changed_files)
        cmake_changed = any(
            f == prefix + "__github__/CMakeLists.txt" for f in changed_files
        )
        if buck_changed and not cmake_changed:
            buck_path = prefix + "BUCK"
            if _buck_diff_has_oss_relevant_changes(buck_path):
                issues.append(
                    f"  {prefix}BUCK was modified but {prefix}__github__/CMakeLists.txt "
                    "was not. These should be kept in sync for OSS builds."
                )

    return issues


def check_reviewer(summary):
    """Check that momentum is listed as reviewer."""
    issues = []
    if "Reviewers:" in summary:
        reviewers_match = re.search(r"Reviewers:\s*(.*)", summary)
        if reviewers_match:
            reviewers = reviewers_match.group(1).lower()
            if "momentum" not in reviewers:
                issues.append(
                    "  Reviewers field does not include 'momentum'. "
                    "Add 'Reviewers: momentum' to the commit message."
                )
    else:
        issues.append(
            "  No Reviewers field found. "
            "Add 'Reviewers: momentum' to the commit message."
        )
    return issues


def main():
    changed_files = get_changed_files()
    if not changed_files:
        print("No changed files found.")
        sys.exit(0)

    relevant_files = [
        f
        for f in changed_files
        if f.startswith("arvr/libraries/momentum/")
        or f.startswith("arvr/libraries/pymomentum/")
    ]
    if not relevant_files:
        print("No momentum/pymomentum files changed.")
        sys.exit(0)

    print(f"Checking {len(relevant_files)} momentum/pymomentum files...")

    summary = get_diff_summary()
    all_issues = []

    issues = check_diff_summary_oss(summary)
    if issues:
        all_issues.append("Diff summary OSS compliance:")
        all_issues.extend(issues)

    issues = check_build_variables(relevant_files)
    if issues:
        all_issues.append("Build file updates:")
        all_issues.extend(issues)

    issues = check_buck_cmake_sync(relevant_files)
    if issues:
        all_issues.append("BUCK/CMake sync:")
        all_issues.extend(issues)

    issues = check_reviewer(summary)
    if issues:
        all_issues.append("Reviewer check:")
        all_issues.extend(issues)

    if all_issues:
        print("\n=== ISSUES FOUND ===\n")
        for issue in all_issues:
            print(issue)
        issue_count = len([i for i in all_issues if i.startswith("  ")])
        print(f"\nTotal: {issue_count} issue(s)")
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
