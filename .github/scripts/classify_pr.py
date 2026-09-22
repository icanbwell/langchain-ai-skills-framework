#!/usr/bin/env python3
"""Deterministic PR risk/type/semver classifier.

Reads PR signals (title, changed files, diff stat, optional CI/mergeable status)
plus a per-repo risk-mapping config, and prints the risk:*, type:*, and semver
labels to apply. No model call — cheap enough to run on every push.
Design/rationale in ../references/classification.md.

This script is meant to be COPIED into the adopting repo (at .github/scripts/
classify_pr.py by this skill's adoption checklist), not referenced from the
plugin's own install path — a GitHub Actions runner in another repo has no
access to this marketplace repo's checkout.

Usage (as invoked by templates/pr-label-classify.yml):
    python3 classify_pr.py \
        --mapping .github/pr-risk-mapping.yml \
        --title "$PR_TITLE" \
        --files-file changed_files.txt \
        --additions 42 --deletions 10 \
        [--ci-status success|failure|pending|unknown] \
        [--mergeable true|false|unknown]

Prints one label per line to stdout (e.g. "risk:standard", "type:feature",
"semver:patch"). risk:* and semver:* always print — the org mandate (EA
review, 2026-09-03) requires both on every PR, so each falls back to its most
conservative value rather than staying silent. A PR matching no type_keywords
entry gets no type:* line — that axis stays best-effort, since a wrong type
label is a nuisance to notice while a missing one is cheap to add by hand.
Exits non-zero only if the mapping file itself is missing/malformed.
"""

import argparse
import fnmatch
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

RISK_TIERS = ("deep-review", "standard", "auto-approve")

# Bump whenever classify_pr.py's logic changes (not on comment-only edits).
# check_setup.py compares this against an adopting repo's deployed copy so a
# repo that copy-pasted this script once and never re-synced shows up as
# drifted instead of silently missing every bug fix landed since.
CLASSIFIER_VERSION = 2


def load_mapping(path: Path) -> dict:
    if yaml is None:
        print(
            "error: PyYAML is required to read the risk-mapping file (pip install pyyaml)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not path.is_file():
        print(f"error: mapping file not found: {path}", file=sys.stderr)
        raise SystemExit(2)
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        print(f"error: mapping file is not valid YAML: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if not isinstance(data, dict):
        print("error: mapping file must be a YAML mapping at the top level", file=sys.stderr)
        raise SystemExit(2)
    return data


def matches_glob(path_str: str, pattern: str) -> bool:
    """fnmatch, but treats a leading `**/` as also matching zero directories.

    `fnmatch.fnmatch`'s `*` matches `/` like any other character, so `**/`
    still translates to a regex requiring a literal `/` before the rest of
    the pattern — a root-level file can never match `**/*.md` even though
    that pattern is written to mean "anywhere in the tree, including root".
    Verified: `fnmatch.fnmatch("README.md", "**/*.md")` is False,
    `fnmatch.fnmatch("docs/README.md", "**/*.md")` is True.
    """
    if fnmatch.fnmatch(path_str, pattern):
        return True
    if pattern.startswith("**/"):
        return fnmatch.fnmatch(path_str, pattern[3:])
    return False


def matches_any(path_str: str, patterns: list[str]) -> bool:
    return any(matches_glob(path_str, pattern) for pattern in patterns)


_ALNUM_RE = re.compile(r"[a-z0-9]")
_GLOB_KEYWORD_RE = re.compile(r"^\*+/?\*([^*/]+)\*$")


def _has_token(text: str, keyword: str) -> bool:
    """True if `keyword` appears in `text` flanked by the start/end of `text`
    or a non-alphanumeric separator on both sides — a *token* match, not a
    bare substring match."""
    pattern = rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def _title_has_keyword(keyword: str, lowered_title: str) -> bool:
    """Match `keyword` as a whole token in `lowered_title`, not a bare
    substring of an unrelated word — a plain `in` check matches "fix" inside
    "fixtures", "add" inside "address", "sev" inside "seventeen", mislabeling
    any PR whose title happens to contain one of those. A keyword that
    already supplies its own right-hand separator (the trailing space in
    "add ", the colon in "deps:") needs no boundary added after it; every
    other keyword gets one, since it would otherwise still match into the
    start of the next word."""
    kw = keyword.lower()
    left = r"(?<![a-z0-9])"
    right = "" if kw and not _ALNUM_RE.match(kw[-1]) else r"(?![a-z0-9])"
    return re.search(left + re.escape(kw) + right, lowered_title) is not None


def _test_pattern_keyword(pattern: str) -> str | None:
    """test_path_patterns ships shaped like `**/*test*` — a glob meant to say
    "test appears in the filename", not "test appears anywhere as a raw
    substring". Plain fnmatch can't tell the difference: `fnmatch.fnmatch`'s
    `*` matches any run of characters (including `/`), so `**/*test*` matches
    any path containing the literal substring "test" *anywhere*, e.g.
    `plugins/foo/latest-model/SKILL.md` or `.../attestation/...` — neither of
    which is a test file. Recognize the `**/*KEYWORD*` shape and return the
    literal keyword so callers can token-match it instead (see `_has_token`);
    returns None for any other pattern shape, which falls back to plain
    fnmatch (`matches_any`), unchanged."""
    m = _GLOB_KEYWORD_RE.match(pattern)
    return m.group(1) if m else None


# fnmatch's `*` matches `/`, so a pattern like `**/*test*` only matches a path whose
# FILENAME contains "test" — it has no notion of "lives under a directory named
# test/". A file like `test/helpers.py` (test directory, non-test-looking filename)
# falls through both the pattern and this fallback would otherwise miss it. Used only
# for the require_tests_for_auto_approve check below — other axes (high_risk_paths,
# semver_none_path_patterns) keep plain fnmatch, since neither the substring-overmatch
# gap above nor this directory gap was reported there.
TEST_DIR_NAMES = {"test", "tests", "spec", "specs"}


def is_test_path(path_str: str, patterns: list[str]) -> bool:
    parts = Path(path_str).parts
    for pattern in patterns:
        keyword = _test_pattern_keyword(pattern)
        if keyword is not None:
            if any(_has_token(part, keyword) for part in parts):
                return True
        elif matches_glob(path_str, pattern):
            return True
    return any(part.lower() in TEST_DIR_NAMES for part in parts[:-1])


def classify_risk(
    files: list[str],
    additions: int,
    deletions: int,
    title: str,
    mapping: dict,
) -> tuple[str, str]:
    """Returns (tier, reason). First matching rule wins — see
    references/classification.md's "Classifier decision order"."""
    high_risk_paths = mapping.get("high_risk_paths") or []
    for f in files:
        for pattern in high_risk_paths:
            if matches_glob(f, pattern):
                return "deep-review", f"changed file {f!r} matches high_risk_paths {pattern!r}"

    lowered_title = title.lower()
    for kw in mapping.get("high_risk_title_keywords") or []:
        if _title_has_keyword(kw, lowered_title):
            return "deep-review", f"title contains high_risk_title_keywords entry {kw!r}"

    total_lines = additions + deletions
    threshold = mapping.get("small_diff_line_threshold", 30)
    if total_lines > threshold:
        return (
            "standard",
            f"{total_lines} changed lines exceeds small_diff_line_threshold ({threshold})",
        )

    if mapping.get("require_tests_for_auto_approve", True):
        test_patterns = mapping.get("test_path_patterns") or []
        if not any(is_test_path(f, test_patterns) for f in files):
            return (
                "standard",
                "require_tests_for_auto_approve is true and no changed file matches test_path_patterns",
            )

    return (
        "auto-approve",
        f"{total_lines} changed lines within threshold, no deep-review trigger matched",
    )


def classify_type(title: str, type_keywords: dict) -> str | None:
    lowered = title.lower()
    for type_name, keywords in (type_keywords or {}).items():
        for kw in keywords:
            if _title_has_keyword(kw, lowered):
                return type_name
    return None


BREAKING_MARKER = re.compile(r"^\w+(\([^)]*\))?!:")  # conventional-commit "feat!:" / "fix!:"

DEFAULT_SEMVER_LABELS = {
    "major": "semver:major",
    "minor": "semver:minor",
    "patch": "semver:patch",
    "none": "semver:none",
}
DEFAULT_SEMVER_MAJOR_TITLE_KEYWORDS = ["breaking", "breaking change"]
DEFAULT_SEMVER_MINOR_TITLE_KEYWORDS = ["feat", "feature", "add "]
DEFAULT_SEMVER_NONE_PATH_PATTERNS = ["**/*.md", "docs/**", "**/*test*", "**/*spec*"]


def classify_semver(
    title: str,
    files: list[str],
    mapping: dict,
) -> tuple[str, str]:
    """Returns (label, reason). Always resolves to one of the four tiers — this
    axis is mandatory (EA review, 2026-09-03), so unlike type:* it never stays
    silent. Label *names* are configurable (semver_labels in the mapping) so a
    repo already running bwell-sdk/health-data-service's older bare-word
    scheme (`Major`/`Minor`/`Patch`/`do-not-release`) can keep its existing,
    working gate instead of renaming labels for no benefit."""
    labels = {**DEFAULT_SEMVER_LABELS, **(mapping.get("semver_labels") or {})}
    lowered_title = title.lower()

    major_keywords = mapping.get("semver_major_title_keywords") or DEFAULT_SEMVER_MAJOR_TITLE_KEYWORDS
    if BREAKING_MARKER.match(title) or any(_title_has_keyword(kw, lowered_title) for kw in major_keywords):
        return labels["major"], "title has a conventional-commit breaking marker or major keyword"

    none_patterns = mapping.get("semver_none_path_patterns") or DEFAULT_SEMVER_NONE_PATH_PATTERNS
    if files and all(matches_any(f, none_patterns) for f in files):
        return labels["none"], "every changed file matches semver_none_path_patterns"

    minor_keywords = mapping.get("semver_minor_title_keywords") or DEFAULT_SEMVER_MINOR_TITLE_KEYWORDS
    if any(_title_has_keyword(kw, lowered_title) for kw in minor_keywords):
        return labels["minor"], "title contains a semver_minor_title_keywords entry"

    return labels["patch"], "no major/none/minor signal matched — most conservative default"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--files-file",
        type=Path,
        help="Path to a file listing changed repo-relative paths, one per line",
    )
    parser.add_argument("--additions", type=int, default=0)
    parser.add_argument("--deletions", type=int, default=0)
    parser.add_argument(
        "--ci-status",
        choices=["success", "failure", "pending", "unknown"],
        default="unknown",
    )
    parser.add_argument("--mergeable", choices=["true", "false", "unknown"], default="unknown")
    parser.add_argument("--verbose", action="store_true", help="Print the classification reason to stderr")
    args = parser.parse_args(argv)

    mapping = load_mapping(args.mapping)

    files: list[str] = []
    if args.files_file and args.files_file.is_file():
        files = [line.strip() for line in args.files_file.read_text().splitlines() if line.strip()]

    risk, reason = classify_risk(files, args.additions, args.deletions, args.title, mapping)

    if risk == "auto-approve" and (args.ci_status not in ("success", "unknown") or args.mergeable == "false"):
        risk = "standard"
        reason += "; downgraded from auto-approve — CI/mergeable status isn't clean yet"

    semver_label, semver_reason = classify_semver(args.title, files, mapping)

    if args.verbose:
        print(f"risk reason: {reason}", file=sys.stderr)
        print(f"semver reason: {semver_reason}", file=sys.stderr)

    print(f"risk:{risk}")
    print(semver_label)

    type_label = classify_type(args.title, mapping.get("type_keywords") or {})
    if type_label:
        print(f"type:{type_label}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
