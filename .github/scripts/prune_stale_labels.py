#!/usr/bin/env python3
"""Decides which of a PR's currently-attached risk:*/type:*/semver:* labels
are safe to remove before applying a freshly computed classification.

A label's `description` lives on the label DEFINITION, which is repo-wide —
once any PR's classify run has ever set a label's description to "Applied by
pr-label-classify", every future PR that carries that label name inherits the
same description, including one a human hand-applies to escalate a
*different* PR. Scoping stale-label removal by description (as the workflow
used to) therefore silently strips a human's manual override the next time
this workflow runs on that same PR — verified live in ui-platform.

This module scopes by the PR's own issue timeline instead, which is per-PR:
a currently-attached label is stale (safe to remove) only if the most recent
`labeled` event for that exact label name, on this exact PR, was performed by
one of the classifier's own identities — not a human, and not some other
automation. Design/rationale: ../references/classification.md's
"Attribution" section.

Takes a set of actor logins, not a single one: the calling workflow can run
under either the default GITHUB_TOKEN (bot identity `github-actions[bot]`) or
an optional PR_LABEL_TOKEN PAT (that PAT owner's own login), and which one is
configured can change over a PR's lifetime (e.g. a repo adds the recommended
PR_LABEL_TOKEN secret after already having labeled PRs under the default
token). Matching only the CURRENT run's resolved identity would make every
label applied under the previous identity permanently un-prunable the moment
the token configuration changes — the caller passes both candidate identities
(the current one and the well-known default) so that transition doesn't
strand old labels.

Usage (as invoked by templates/pr-label-classify.yml):
    gh pr view "$PR" --json labels \
        -q '.labels[] | select(.name | test("^(risk|type|semver):")) | .name' \
        > current_managed_labels.txt
    gh api --paginate --slurp "repos/$REPO/issues/$PR/timeline" > timeline.json
    printf '%s\n' "$ACTOR_LOGIN" "github-actions[bot]" | sort -u > actor_logins.txt
    python3 prune_stale_labels.py \
        --current-labels-file current_managed_labels.txt \
        --timeline-file timeline.json \
        --actor-logins-file actor_logins.txt

Prints one label per line — the subset of the current labels that are safe to
remove. No network calls, no token, same as classify_pr.py — fetching the
labels and the timeline is the caller's job.
"""

import argparse
import json
import sys
from pathlib import Path


def stale_labels(
    current_labels: list[str],
    timeline_pages: list,
    actor_logins,
) -> list[str]:
    """`timeline_pages` is the raw `gh api --paginate --slurp .../timeline`
    output: a list of pages, each itself a list of timeline event dicts.
    `--slurp` (rather than filtering with `-q` during pagination) sidesteps
    gh concatenating each page's jq output as separate JSON values instead of
    one parseable array. `actor_logins` is any collection supporting `in`
    (a set, list, or tuple) of logins that all count as "the classifier"."""
    last_actor: dict[str, str | None] = {}
    for page in timeline_pages:
        for event in page:
            if event.get("event") != "labeled":
                continue
            label_name = (event.get("label") or {}).get("name")
            if not label_name:
                continue
            last_actor[label_name] = (event.get("actor") or {}).get("login")
    return [name for name in current_labels if last_actor.get(name) in actor_logins]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-labels-file", required=True, type=Path)
    parser.add_argument("--timeline-file", required=True, type=Path)
    parser.add_argument("--actor-logins-file", required=True, type=Path)
    args = parser.parse_args(argv)

    current = [line.strip() for line in args.current_labels_file.read_text().splitlines() if line.strip()]
    timeline_pages = json.loads(args.timeline_file.read_text() or "[]")
    actor_logins = {line.strip() for line in args.actor_logins_file.read_text().splitlines() if line.strip()}

    for name in stale_labels(current, timeline_pages, actor_logins):
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
