# /// script
# requires-python = ">=3.12"
# ///

import json
import re
import sys
from collections.abc import Sequence
from typing import Any

HELP_TEXT = """Usage: count_the.py [OPTIONS]

Count instances of the word \"the\" in input text and print a JSON result.

Options:
  --help             Show this help message and exit

Examples:
  count_the.py
  echo '{"text": "the cat and The dog"}' | count_the.py

Input JSON keys (stdin, all optional):
  text               string (default: built-in sample text)
"""


def _read_args() -> dict[str, Any]:
    """Read a JSON object from stdin, defaulting to an empty object."""
    raw_input = sys.stdin.read().strip()
    if not raw_input:
        return {}

    try:
        parsed = json.loads(raw_input)
    except json.JSONDecodeError as error:
        raise ValueError("stdin must contain valid JSON") from error

    if not isinstance(parsed, dict):
        raise ValueError("stdin must be a JSON object")

    return parsed


def _count_the_instances(text: str) -> int:
    # Count whole-word matches for "the" (case-insensitive).
    return len(re.findall(r"\bthe\b", text, flags=re.IGNORECASE))


def _parse_cli_args(argv: Sequence[str]) -> bool:
    """Return True when help was requested, otherwise validate args."""
    if len(argv) == 1:
        return False
    if len(argv) == 2 and argv[1] in {"--help", "-h"}:
        return True
    raise ValueError("unsupported arguments; use --help")


def main() -> int:
    try:
        if _parse_cli_args(sys.argv):
            sys.stdout.write(HELP_TEXT)
            return 0

        args = _read_args()
        text = str(args.get("text", ""))

        if not text:
            # return error that no text was sent
            raise ValueError("No text was sent to count_the.py")

        result = {
            "the_count": _count_the_instances(text),
            "text": text,
        }
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return 0
    except Exception as error:
        json.dump({"error": str(error)}, sys.stderr)
        sys.stderr.write("\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

