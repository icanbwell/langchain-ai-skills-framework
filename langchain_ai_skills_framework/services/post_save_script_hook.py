from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PostSaveScriptHook(Protocol):
    """Hook invoked after a script is saved successfully."""

    async def on_script_saved(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
    ) -> None: ...
