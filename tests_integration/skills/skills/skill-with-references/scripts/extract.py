# /// script
# dependencies = [
#   "beautifulsoup4",
# ]
# ///


# https://agentskills.io/skill-creation/using-scripts#self-contained-scripts
# PEP 723 defines a standard format for inline script metadata. Declare dependencies in a TOML block inside # /// markers:

from bs4 import BeautifulSoup

sample_html = "<html><body><h1>Hello</h1><p>dependency-check</p></body></html>"
parsed_text = BeautifulSoup(sample_html, "html.parser").get_text(" ", strip=True)
print(parsed_text)
