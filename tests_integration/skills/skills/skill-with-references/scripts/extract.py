# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "beautifulsoup4>=4.12.0",
# ]
# ///

import json
import sys
from collections.abc import Sequence
from typing import Any

from bs4 import BeautifulSoup

DEFAULT_HTML = "<html><body><h1>Hello</h1><p>dependency-check</p></body></html>"
HELP_TEXT = """Usage: extract.py [OPTIONS] INPUT_FILE

Extract text from HTML input and print a JSON result.

Options:
  --help             Show this help message and exit
  INPUT_FILE         Logical input identifier (default: document.pdf)

Examples:
  extract.py
  echo '{"html": "<h1>Hello</h1>"}' | extract.py

Input JSON keys (stdin, all optional):
  file_path          string (default: document.pdf)
  page_range         string (default: all)
  html               string (default: built-in sample HTML)
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


def _extract_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


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
        # Agent Skills conventions use lowercase argument names.
        file_path = str(args.get("file_path", "document.pdf"))
        page_range = str(args.get("page_range", "all"))
        html = str(args.get("html", DEFAULT_HTML))

        result = {
            "extracted_text": _extract_text(html),
            "file_path": file_path,
            "page_range": page_range,
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
