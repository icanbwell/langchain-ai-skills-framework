from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from langchain_ai_skills_framework.loaders.exceptions.skill_validation_error import (
    SkillValidationError,
)
from langchain_ai_skills_framework.loaders.github_skill_downloader import (
    GithubSkillDownloader,
)


def test_download_uses_expected_storage_options_and_cache_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    captured_storage_options: dict[str, object] = {}
    get_calls: list[tuple[str, str, bool]] = []

    class _FakeGithubFilesystem:
        def get(
            self, remote_path: str, local_path: str, recursive: bool = False
        ) -> None:
            get_calls.append((remote_path, local_path, recursive))

        def ls(self, path: str, detail: bool = False) -> Sequence[str]:
            del path, detail
            return ()

    def _fake_filesystem(
        protocol: str, **storage_options: object
    ) -> _FakeGithubFilesystem:
        assert protocol == "github"
        captured_storage_options.update(storage_options)
        return _FakeGithubFilesystem()

    monkeypatch.setattr(
        "langchain_ai_skills_framework.loaders.github_directory_downloader.fsspec",
        SimpleNamespace(filesystem=_fake_filesystem),
    )

    downloader = GithubSkillDownloader()
    downloaded_path = downloader.download(
        cache_path=tmp_path / "cache",
        skills_directory="github://my-org/private-skills/skills?ref=main",
        github_token="token-value",
    )

    assert captured_storage_options == {
        "org": "my-org",
        "repo": "private-skills",
        "sha": "main",
        "username": "x-access-token",
        "token": "token-value",
    }
    assert len(get_calls) == 1
    assert get_calls[0][0] == "skills"
    assert get_calls[0][2] is True
    assert downloaded_path.name.startswith("my-org-private-skills-")
    assert downloaded_path.parent == (tmp_path / "cache").resolve()


def test_download_raises_validation_error_when_fsspec_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    def _raise_filesystem(protocol: str, **storage_options: object) -> object:
        del protocol, storage_options
        raise RuntimeError("network error")

    monkeypatch.setattr(
        "langchain_ai_skills_framework.loaders.github_directory_downloader.fsspec",
        SimpleNamespace(filesystem=_raise_filesystem),
    )

    downloader = GithubSkillDownloader()
    with pytest.raises(
        SkillValidationError,
        match="Unable to download github:// directory into cache",
    ):
        downloader.download(
            cache_path=tmp_path / "cache",
            skills_directory="github://my-org/private-skills/skills",
            github_token=None,
        )


def test_download_omits_auth_fields_when_github_token_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    captured_storage_options: dict[str, object] = {}

    class _FakeGithubFilesystem:
        def get(
            self, remote_path: str, local_path: str, recursive: bool = False
        ) -> None:
            del remote_path, local_path, recursive

        def ls(self, path: str, detail: bool = False) -> Sequence[str]:
            del path, detail
            return ()

    def _fake_filesystem(
        protocol: str, **storage_options: object
    ) -> _FakeGithubFilesystem:
        assert protocol == "github"
        captured_storage_options.update(storage_options)
        return _FakeGithubFilesystem()

    monkeypatch.setattr(
        "langchain_ai_skills_framework.loaders.github_directory_downloader.fsspec",
        SimpleNamespace(filesystem=_fake_filesystem),
    )

    downloader = GithubSkillDownloader()
    downloader.download(
        cache_path=tmp_path / "cache",
        skills_directory="github://my-org/private-skills/skills?ref=main",
        github_token=None,
    )

    assert captured_storage_options == {
        "org": "my-org",
        "repo": "private-skills",
        "sha": "main",
    }


@pytest.mark.parametrize(
    ("skills_directory", "message"),
    [
        (
            "https://github.com/my-org/private-skills",
            "URI must use the github:// scheme",
        ),
        (
            "github://my-org/private-skills/skills#fragment",
            "github:// URI must not include a fragment",
        ),
        (
            "github://my-org/private-skills/skills?x=1",
            "github:// URI supports only '?ref=' query parameter; got: x",
        ),
    ],
)
def test_download_validates_github_uri(
    skills_directory: str,
    message: str,
    tmp_path: Path,
) -> None:
    downloader = GithubSkillDownloader()
    with pytest.raises(SkillValidationError, match=re.escape(message)):
        downloader.download(
            cache_path=tmp_path / "cache",
            skills_directory=skills_directory,
            github_token=None,
        )
