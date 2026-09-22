#!/bin/bash
set -euo pipefail

# Opens a PR with its risk:*/type:*/semver:* labels attached at creation time,
# in one call -- so an agent doesn't have to hand-assemble the git-diff /
# classify_pr.py / gh pr create pipeline itself every time it opens a PR (that
# reconstruction is itself an extra reasoning step, and a place to get body
# quoting or the diff/awk one-liner wrong). See SKILL.md's "When Claude opens
# the PR itself" section for why the pre-push guess is safe to make.
#
# This script is meant to be COPIED into the adopting repo (at
# .github/scripts/create_labeled_pr.sh, alongside classify_pr.py), not
# referenced from this plugin's install path -- same reason as classify_pr.py
# itself (see that script's own header).
#
# Usage:
#   .github/scripts/create_labeled_pr.sh --title "..." --body-file body.md \
#       [--base main] [--dry-run] [-- <extra gh pr create args>]
#
# Requires .github/scripts/classify_pr.py and .github/pr-risk-mapping.yml to
# compute labels (Phase 2). If either is missing, opens the PR unlabeled and
# says so on stderr -- no hand-guessing labels without a calibrated mapping.

BASE="main"
TITLE=""
BODY_FILE=""
DRY_RUN="false"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --title) TITLE="$2"; shift 2 ;;
    --body-file) BODY_FILE="$2"; shift 2 ;;
    --base) BASE="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$TITLE" ]; then
  echo "error: --title is required" >&2
  exit 2
fi
if [ -z "$BODY_FILE" ] || [ ! -f "$BODY_FILE" ]; then
  echo "error: --body-file is required and must point to an existing file" >&2
  exit 2
fi

LABEL_ARGS=()
if [ -f ".github/scripts/classify_pr.py" ] && [ -f ".github/pr-risk-mapping.yml" ]; then
  # origin/$BASE may not be fetched in a shallow/fresh checkout; fall back to
  # the local branch of the same name so this doesn't hard-fail on that alone.
  base_ref="origin/$BASE"
  git rev-parse --verify -q "$base_ref" >/dev/null || base_ref="$BASE"
  base_sha="$(git merge-base "$base_ref" HEAD)"

  changed_files="$(mktemp)"
  trap 'rm -f "$changed_files"' EXIT
  git diff --name-only "$base_sha" HEAD > "$changed_files"

  read -r additions deletions <<< "$(
    git diff --shortstat "$base_sha" HEAD | awk '
      { for (i = 1; i <= NF; i++) { if ($i ~ /^[0-9]+$/) n = $i
          if ($i ~ /insertion/) a = n; if ($i ~ /deletion/) d = n } }
      END { print (a ? a : 0), (d ? d : 0) }')"

  while IFS= read -r label; do
    [ -n "$label" ] && LABEL_ARGS+=(--label "$label")
  done < <(python3 .github/scripts/classify_pr.py \
    --mapping .github/pr-risk-mapping.yml \
    --title "$TITLE" \
    --files-file "$changed_files" \
    --additions "$additions" --deletions "$deletions")
else
  echo "note: Phase 2 classifier not adopted in this repo (.github/scripts/classify_pr.py or .github/pr-risk-mapping.yml missing) -- opening unlabeled" >&2
fi

# Built as one array, then either printed (--dry-run) or exec'd, so the two
# paths can't drift apart. The `ARR[@]+"${ARR[@]}"` guard on each append is
# required for bash 3.2 (macOS's default /bin/bash) -- expanding an empty
# array's [@] directly under `set -u` is a hard error on that version, fixed
# only in bash 4.4+.
ARGS=(--title "$TITLE" --body-file "$BODY_FILE" --base "$BASE")
ARGS+=(${LABEL_ARGS[@]+"${LABEL_ARGS[@]}"})
ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

if [ "$DRY_RUN" = "true" ]; then
  printf 'gh pr create'
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

gh pr create "${ARGS[@]}"
