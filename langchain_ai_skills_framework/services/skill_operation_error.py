"""Framework-agnostic error raised by skill service operations.

LangChain tools translate this into ``ToolException``; other tool
frameworks can map it to their own error type.
"""


class SkillOperationError(Exception):
    """A skill operation failed with a user-facing message."""
