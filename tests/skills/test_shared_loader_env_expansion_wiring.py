"""The container builds the shared loader's plugin manager from PLUGINS_MARKETPLACE_ENV_EXPANSION_INCLUDE."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from simple_container.container.simple_container import ContainerError

from langchain_ai_skills_framework.container.container_factory import _build_shared_loader
from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.marketplace_plugin_manager import PluginEntry


class _Container:
    def __init__(self) -> None:
        self._env = LangchainAISkillsFrameworkEnvironmentVariables()

    def resolve(self, service_type: type) -> Any:
        if (
            service_type is LangchainAISkillsFrameworkEnvironmentVariables
            or getattr(service_type, "__name__", "") == "EnvironmentVariables"
        ):
            return self._env
        if getattr(service_type, "__name__", "") == "BaseStore":
            raise ContainerError("no store")
        return MagicMock()


def _read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, include: str | None) -> str | None:
    monkeypatch.setenv("PLUGINS_MARKETPLACE", "github://example/repo/plugins?ref=main")
    monkeypatch.setenv("WIRING_TEST_VAR", "http://expanded")
    if include is None:
        monkeypatch.delenv("PLUGINS_MARKETPLACE_ENV_EXPANSION_INCLUDE", raising=False)
    else:
        monkeypatch.setenv("PLUGINS_MARKETPLACE_ENV_EXPANSION_INCLUDE", include)
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"s": {"url": "${WIRING_TEST_VAR}/x"}}}))
    loader = _build_shared_loader(c=_Container())  # type: ignore[arg-type]
    result = loader._plugin_manager.read_mcp_configs(PluginEntry(name="other-plugin", path=tmp_path))  # type: ignore[attr-defined]
    return result.entries[0].url if result.entries else None


def test_unset_keeps_todays_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _read(tmp_path, monkeypatch, include=None) == "http://expanded/x"


def test_a_list_without_the_plugin_leaves_the_placeholder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _read(tmp_path, monkeypatch, include="bailey") == "${WIRING_TEST_VAR}/x"


def test_a_list_with_the_plugin_expands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _read(tmp_path, monkeypatch, include="other-plugin") == "http://expanded/x"


def test_the_per_variable_map_is_passed_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLUGINS_MARKETPLACE", "github://example/repo/plugins?ref=main")
    monkeypatch.setenv("PLUGINS_MARKETPLACE_ENV_EXPANSION_INCLUDE", "bailey")
    monkeypatch.setenv("WIRING_TEST_VAR", "http://expanded")
    monkeypatch.setenv("WIRING_OTHER_VAR", "secret")
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"s": {"url": "${WIRING_TEST_VAR}/${WIRING_OTHER_VAR}"}}})
    )
    loader = _build_shared_loader(
        c=_Container(),  # type: ignore[arg-type]
        env_expansion_variables={"other-plugin": frozenset({"WIRING_TEST_VAR"})},
    )

    result = loader._plugin_manager.read_mcp_configs(PluginEntry(name="other-plugin", path=tmp_path))  # type: ignore[attr-defined]

    assert result.entries[0].url == "http://expanded/${WIRING_OTHER_VAR}"
