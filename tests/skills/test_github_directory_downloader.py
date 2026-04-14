from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from langchain_ai_skills_framework.loaders.github_directory_downloader import (
    GithubDirectoryDownloader,
    GitLocation,
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

    downloader = GithubDirectoryDownloader()
    downloaded_path = downloader.download(
        cache_path=tmp_path / "cache",
        source_uri="github://my-org/private-repo/configs?ref=main",
        github_token="token-value",
    )

    assert captured_storage_options == {
        "org": "my-org",
        "repo": "private-repo",
        "sha": "main",
        "username": "x-access-token",
        "token": "token-value",
    }
    assert len(get_calls) == 1
    assert get_calls[0][0] == "configs"
    assert get_calls[0][2] is True
    assert downloaded_path.name.startswith("my-org-private-repo-")
    assert downloaded_path.parent == (tmp_path / "cache").resolve()


def test_download_raises_value_error_when_fsspec_fails(
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

    downloader = GithubDirectoryDownloader()
    downloader._RETRY_BASE_DELAY = 0.0  # no wait in tests
    with pytest.raises(ValueError, match="Download failed after"):
        downloader.download(
            cache_path=tmp_path / "cache",
            source_uri="github://my-org/private-repo/configs",
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

    downloader = GithubDirectoryDownloader()
    downloader.download(
        cache_path=tmp_path / "cache",
        source_uri="github://my-org/private-repo/configs?ref=main",
        github_token=None,
    )

    assert captured_storage_options == {
        "org": "my-org",
        "repo": "private-repo",
        "sha": "main",
    }


@pytest.mark.parametrize(
    ("source_uri", "message"),
    [
        (
            "https://github.com/my-org/private-repo",
            "must use the github:// scheme",
        ),
        (
            "github://my-org/private-repo/path#fragment",
            "must not include a fragment",
        ),
        (
            "github://my-org/private-repo/path?x=1",
            "supports only '\\?ref=' query parameter; got: x",
        ),
    ],
)
def test_parse_github_uri_validates(
    source_uri: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GithubDirectoryDownloader.parse_github_uri(source_uri)


def test_download_preserves_existing_cache_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When a download fails, the previous cached directory must survive."""
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "cache"

    # Pre-populate the cache with known content.
    downloader = GithubDirectoryDownloader()
    git_loc = downloader.parse_github_uri(
        "github://my-org/private-repo/configs?ref=main"
    )
    # Compute the target dir the same way download() does.
    from hashlib import sha256

    key = f"{git_loc.owner}/{git_loc.repository}:main:configs"
    cache_dir_name = (
        f"{git_loc.owner}-{git_loc.repository}"
        f"-{sha256(key.encode('utf-8')).hexdigest()[:12]}"
    )
    target_dir = cache_dir / cache_dir_name
    target_dir.mkdir(parents=True)
    (target_dir / "old_file.txt").write_text("precious data")

    # Now make fsspec fail.
    def _raise_filesystem(protocol: str, **storage_options: object) -> object:
        del protocol, storage_options
        raise RuntimeError("GitHub is down")

    monkeypatch.setattr(
        "langchain_ai_skills_framework.loaders.github_directory_downloader.fsspec",
        SimpleNamespace(filesystem=_raise_filesystem),
    )

    downloader._RETRY_BASE_DELAY = 0.0
    with pytest.raises(ValueError, match="Download failed after"):
        downloader.download(
            cache_path=cache_dir,
            source_uri="github://my-org/private-repo/configs?ref=main",
            github_token=None,
        )

    # The old data must still be there.
    assert target_dir.is_dir()
    assert (target_dir / "old_file.txt").read_text() == "precious data"


def test_download_retries_on_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Download should succeed if a transient failure resolves on retry."""
    monkeypatch.chdir(tmp_path)

    call_count = 0

    class _FlakeyFilesystem:
        def get(
            self, remote_path: str, local_path: str, recursive: bool = False
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient network failure")
            # Third call succeeds — write a file so the directory isn't empty.
            Path(local_path).mkdir(parents=True, exist_ok=True)
            (Path(local_path) / "data.json").write_text("{}")

        def ls(self, path: str, detail: bool = False) -> Sequence[str]:
            del path, detail
            return ()

    monkeypatch.setattr(
        "langchain_ai_skills_framework.loaders.github_directory_downloader.fsspec",
        SimpleNamespace(filesystem=lambda *_a, **_kw: _FlakeyFilesystem()),
    )

    downloader = GithubDirectoryDownloader()
    downloader._RETRY_BASE_DELAY = 0.0
    result = downloader.download(
        cache_path=tmp_path / "cache",
        source_uri="github://my-org/private-repo/configs?ref=main",
        github_token=None,
    )

    assert result.is_dir()
    assert call_count == 3


def test_download_skips_when_disk_cache_is_fresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When another worker already refreshed the cache, download is skipped."""
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "cache"

    # Pre-populate cache and timestamp file.
    downloader = GithubDirectoryDownloader()
    git_loc = downloader.parse_github_uri(
        "github://my-org/private-repo/configs?ref=main"
    )
    from hashlib import sha256

    key = f"{git_loc.owner}/{git_loc.repository}:main:configs"
    cache_dir_name = (
        f"{git_loc.owner}-{git_loc.repository}"
        f"-{sha256(key.encode('utf-8')).hexdigest()[:12]}"
    )
    target_dir = cache_dir / cache_dir_name
    target_dir.mkdir(parents=True)
    (target_dir / "cached_file.txt").write_text("already here")
    # Write a fresh timestamp.
    downloader._mark_cache_fresh(target_dir)

    # fsspec should NOT be called.
    def _should_not_be_called(protocol: str, **kw: object) -> object:
        raise AssertionError("fsspec was called but cache should be fresh")

    monkeypatch.setattr(
        "langchain_ai_skills_framework.loaders.github_directory_downloader.fsspec",
        SimpleNamespace(filesystem=_should_not_be_called),
    )

    result = downloader.download(
        cache_path=cache_dir,
        source_uri="github://my-org/private-repo/configs?ref=main",
        github_token=None,
        cache_ttl_seconds=300,
    )

    assert result.is_dir()
    assert (result / "cached_file.txt").read_text() == "already here"


def test_parse_github_uri_returns_git_location() -> None:
    result = GithubDirectoryDownloader.parse_github_uri(
        "github://my-org/my-repo/path/to/dir?ref=develop"
    )
    assert result == GitLocation(
        repo_url="https://github.com/my-org/my-repo.git",
        owner="my-org",
        repository="my-repo",
        path="path/to/dir",
        branch="develop",
    )
