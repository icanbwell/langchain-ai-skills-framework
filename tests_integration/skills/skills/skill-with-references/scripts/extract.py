# /// script
# dependencies = [
#   "beautifulsoup4",
# ]
# ///
import sys
import json

# Read arguments from stdin
args = json.load(sys.stdin)

# https://agentskills.io/skill-creation/using-scripts#self-contained-scripts
# PEP 723 defines a standard format for inline script metadata. Declare dependencies in a TOML block inside # /// markers:

from bs4 import BeautifulSoup

# ✅ Use lowercase parameter names for compatibility
file_path = args.get("file_path", "document.pdf")
page_range = args.get("page_range", "all")

sample_html = "<html><body><h1>Hello</h1><p>dependency-check</p></body></html>"
parsed_text = BeautifulSoup(sample_html, "html.parser").get_text(" ", strip=True)

# Process data
result = {"extracted_text": parsed_text}

# Output JSON to stdout
print(json.dumps(result))
