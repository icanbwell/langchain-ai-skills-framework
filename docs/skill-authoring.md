# Skill Authoring Guide

This guide describes how to author a valid `SKILL.md` for the aiskills loader.

## Location and layout
- Each skill lives in its own directory under the configured skills root.
- The skill file must be named `SKILL.md` and live directly in that directory.
- The directory name must match the normalized skill name.

Example layout:
```
<skills_root>/
  customer-support/
    SKILL.md
  sales-analytics/
    SKILL.md
```

## Required frontmatter
`SKILL.md` must start with YAML frontmatter and a terminator line:
```
---
name: customer-support
description: Handle customer support requests and escalation steps.
---
```
Rules enforced by the loader:
- Frontmatter must start with `---` and include a terminating `---`.
- `name` is required, lower-case, hyphenated, and must match the directory name.
- `description` is required and must be a non-empty string.

## Optional frontmatter
You may include these optional fields:
- `license`: string
- `compatibility`: non-empty string (max 500 chars)
- `metadata`: mapping of string keys to string values
- `allowed-tools`: space-delimited string of tool names

## Name normalization rules
The loader normalizes names by:
- Lowercasing
- Replacing underscores with hyphens
- Collapsing repeated hyphens
- Trimming leading or trailing hyphens

Your `name` must already match the normalized form and match the directory name.

## Body content
After the frontmatter terminator, include the full skill body. This content is returned by the `LoadSkillTool` and should contain detailed instructions, policies, or procedures. Avoid secrets or sensitive content.

## Full example
```
---
name: customer-support
description: Handle customer support requests and escalation steps.
license: Apache-2.0
compatibility: Works with v1 support policies.
metadata:
  owner: support-team
  severity: standard
allowed-tools: load_skill
---
# Customer Support

Use this skill to guide support interactions. Follow escalation rules for billing or security issues.
```

