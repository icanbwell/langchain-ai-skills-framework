import re


def normalize_skill_name(value: str) -> str:
    """Normalize a skill name to a canonical lowercase-hyphenated form.

    Strips whitespace, lowercases, replaces underscores and spaces with
    hyphens, collapses consecutive hyphens, and strips leading/trailing
    hyphens.
    """
    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-")
