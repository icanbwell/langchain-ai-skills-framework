import json
import logging
import shutil
import time
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import fsspec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GitLocation:
    repo_url: str
    owner: str
    repository: str
    path: str
    branch: str | None


class GithubDirectoryDownloader:
    """Downloads github:// directories into a local cache path using fsspec."""

    _github_uri_example = "github://my-org/my-repo/path?ref=main"
    _github_token_username = "x-access-token"

    _MAX_RETRIES = 3
    _RETRY_BASE_DELAY = 2.0

    def download(
        self,
        *,
        source_uri: str,
        github_token: str | None,
        cache_path: Path,
        cache_ttl_seconds: int = 0,
        include_directories: AbstractSet[str] | None = None,
        exclude_directories: AbstractSet[str] | None = None,
    ) -> Path:
        """Download a github:// URI to a local directory.

        Args:
            source_uri: github://owner/repo/path?ref=branch
            github_token: Optional GitHub token for private repos.
            cache_path: Local directory for cached downloads.
            cache_ttl_seconds: If > 0, skip download when existing cache
                is younger than this many seconds.
            include_directories: When set, only download these top-level
                directory names under source_path to include.
            exclude_directories: Top-level directory names under source_path
                to skip during download.

        When source_path is empty (repo root) and include/exclude filters are
        set, the downloader first fetches ``.claude-plugin/marketplace.json``
        to discover the ``pluginRoot`` directory, then applies filters to items
        inside that directory while downloading everything else unfiltered.

        Returns:
            Resolved path to the downloaded directory.

        Raises:
            ValueError: If the URI is malformed or download fails.
        """
        git_location = self.parse_github_uri(source_uri)
        source_path = git_location.path.strip("/")
        ref = git_location.branch or "HEAD"

        cache_root = cache_path.expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        inc_part = ",".join(sorted(include_directories)) if include_directories else ""
        exc_part = ",".join(sorted(exclude_directories)) if exclude_directories else ""
        key = f"{git_location.owner}/{git_location.repository}:{ref}:{source_path}:inc={inc_part}:exc={exc_part}"
        cache_dir_name = (
            f"{git_location.owner}-{git_location.repository}-{sha256(key.encode('utf-8')).hexdigest()[:12]}"
        )
        target_dir = (cache_root / cache_dir_name).resolve()
        if not str(target_dir).startswith(str(cache_root)):
            raise ValueError(f"Path traversal detected in github:// URI: {source_uri}")

        # If the on-disk cache is fresh, skip the download.  Multiple
        # workers may check this concurrently — that is fine; the worst
        # case is a few redundant downloads whose atomic swaps are harmless.
        if cache_ttl_seconds > 0 and self._is_cache_fresh(target_dir, cache_ttl_seconds):
            logger.debug(
                "Cache for %s is fresh — skipping download",
                source_uri,
            )
            return target_dir.resolve()

        try:
            self._download_with_retry(
                git_location=git_location,
                source_path=source_path,
                github_token=github_token,
                target_dir=target_dir,
                include_directories=include_directories,
                exclude_directories=exclude_directories,
            )
            self._mark_cache_fresh(target_dir)
        except ValueError:
            # Download failed — fall back to stale cache if it exists.
            if target_dir.is_dir():
                logger.warning(
                    "Download failed for %s — serving stale cache from %s",
                    source_uri,
                    target_dir,
                )
            else:
                raise
        return target_dir.resolve()

    @staticmethod
    def _is_cache_fresh(target_dir: Path, ttl_seconds: int) -> bool:
        """Return True if the cache directory exists and was refreshed recently."""
        ts_file = target_dir.with_name(target_dir.name + ".ts")
        if not ts_file.exists() or not target_dir.is_dir():
            return False
        age = time.time() - ts_file.stat().st_mtime
        return age < ttl_seconds

    @staticmethod
    def _mark_cache_fresh(target_dir: Path) -> None:
        """Write a timestamp marker so other workers know the cache is fresh."""
        ts_file = target_dir.with_name(target_dir.name + ".ts")
        ts_file.write_text(str(time.time()))

    def _download_with_retry(
        self,
        *,
        git_location: GitLocation,
        source_path: str,
        github_token: str | None,
        target_dir: Path,
        include_directories: AbstractSet[str] | None = None,
        exclude_directories: AbstractSet[str] | None = None,
    ) -> None:
        """Try the download up to ``_MAX_RETRIES`` times with exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                self._fetch_to_directory(
                    git_location=git_location,
                    source_path=source_path,
                    github_token=github_token,
                    target_dir=target_dir,
                    include_directories=include_directories,
                    exclude_directories=exclude_directories,
                )
                return
            except Exception as exc:  # noqa: BLE001 - retry loop must catch any failure to back off and retry
                last_exc = exc
                if attempt < self._MAX_RETRIES - 1:
                    delay = self._RETRY_BASE_DELAY * (2**attempt)
                    exc_type = type(exc).__name__
                    logger.warning(
                        "Download attempt %d/%d failed for %s/%s (retrying in %.1fs): [%s] %s",
                        attempt + 1,
                        self._MAX_RETRIES,
                        git_location.owner,
                        git_location.repository,
                        delay,
                        exc_type,
                        exc,
                    )
                    time.sleep(delay)
        source_uri = f"github://{git_location.owner}/{git_location.repository}/{source_path}"
        if git_location.branch:
            source_uri += f"?ref={git_location.branch}"
        token_status = "GITHUB_TOKEN is set" if github_token else "GITHUB_TOKEN is NOT set"
        exc_type = type(last_exc).__name__ if last_exc else "unknown"
        raise ValueError(
            f"Download failed after {self._MAX_RETRIES} attempts for {source_uri} "
            f"({token_status}): [{exc_type}] {last_exc}"
        ) from last_exc

    def _fetch_to_directory(
        self,
        *,
        git_location: GitLocation,
        source_path: str,
        github_token: str | None,
        target_dir: Path,
        include_directories: AbstractSet[str] | None = None,
        exclude_directories: AbstractSet[str] | None = None,
    ) -> None:
        """Download remote content into *target_dir* using atomic swap.

        Downloads into a staging directory first, then swaps it into place.
        If the download fails, the existing *target_dir* is left untouched
        so callers can fall back to stale-but-valid cached data.
        """
        staging_dir = target_dir.with_name(target_dir.name + ".staging")
        old_dir = target_dir.with_name(target_dir.name + ".old")

        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            storage_options: dict[str, object] = {
                "org": git_location.owner,
                "repo": git_location.repository,
            }
            if git_location.branch:
                storage_options["sha"] = git_location.branch
            if github_token:
                storage_options["username"] = self._github_token_username
                storage_options["token"] = github_token

            filesystem = fsspec.filesystem("github", **storage_options)
            if source_path and (include_directories or exclude_directories):
                self._download_filtered(
                    filesystem=filesystem,
                    remote_path=source_path,
                    staging_dir=staging_dir,
                    include_directories=include_directories,
                    exclude_directories=exclude_directories,
                )
            elif source_path:
                filesystem.get(source_path, str(staging_dir), recursive=True)
            else:
                self._download_repo_root(
                    filesystem=filesystem,
                    staging_dir=staging_dir,
                    include_directories=include_directories,
                    exclude_directories=exclude_directories,
                )
        except ValueError:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(staging_dir, ignore_errors=True)
            source_uri = f"github://{git_location.owner}/{git_location.repository}/{source_path}"
            if git_location.branch:
                source_uri += f"?ref={git_location.branch}"
            exc_type = type(exc).__name__
            exc_detail = str(exc) or "(no message)"
            status_code = ""
            if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
                status_code = f" [HTTP {exc.response.status_code}]"
            raise ValueError(
                f"Unable to download github:// directory into cache: {source_uri} — "
                f"{exc_type}{status_code}: {exc_detail}"
            ) from exc

        # Atomic swap: staging → target, target → old
        if old_dir.exists():
            shutil.rmtree(old_dir)
        if target_dir.exists():
            shutil.move(str(target_dir), str(old_dir))
        shutil.move(str(staging_dir), str(target_dir))
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)

    @staticmethod
    def _download_filtered(
        *,
        filesystem: Any,
        remote_path: str,
        staging_dir: Path,
        include_directories: AbstractSet[str] | None,
        exclude_directories: AbstractSet[str] | None,
    ) -> None:
        """Download items from a remote path, applying include/exclude filters."""
        normalized_include = {name.lower() for name in include_directories} if include_directories else None
        normalized_exclude = {name.lower() for name in exclude_directories} if exclude_directories else set()
        for remote_item in filesystem.ls(remote_path, detail=False):
            item_name = Path(str(remote_item)).name
            item_lower = item_name.lower()
            if normalized_include is not None and item_lower not in normalized_include:
                logger.debug("Skipping non-included directory: %s", item_name)
                continue
            if item_lower in normalized_exclude:
                logger.debug("Skipping excluded directory: %s", item_name)
                continue
            destination = staging_dir / item_name
            filesystem.get(str(remote_item), str(destination), recursive=True)

    @staticmethod
    def _read_plugin_root_from_manifest(filesystem: Any, *, has_filters: bool = False) -> str | None:
        """Fetch .claude-plugin/marketplace.json and extract pluginRoot."""
        manifest_path = ".claude-plugin/marketplace.json"
        try:
            with filesystem.open(manifest_path, "r") as f:
                manifest = json.loads(f.read())
            if not isinstance(manifest, dict):
                return None
            metadata = manifest.get("metadata") or {}
            raw_root = metadata.get("pluginRoot", "")
            if raw_root and isinstance(raw_root, str):
                result: str = raw_root.removeprefix("./").strip("/")
                return result
        except Exception:  # noqa: BLE001 - manifest read is best-effort; fsspec backends vary in what they raise for "not found"
            if has_filters:
                logger.warning(
                    "Could not read %s from remote — include/exclude filters "
                    "will not be applied at download time (full repo will be fetched)",
                    manifest_path,
                )
            else:
                logger.debug("Could not read %s from remote", manifest_path)
        return None

    @staticmethod
    def _download_repo_root(
        *,
        filesystem: Any,
        staging_dir: Path,
        include_directories: AbstractSet[str] | None,
        exclude_directories: AbstractSet[str] | None,
    ) -> None:
        """Download repo root, discovering pluginRoot from marketplace.json for filtering."""
        has_filters = bool(include_directories or exclude_directories)
        plugin_root_path: str | None = None

        if has_filters:
            plugin_root_path = GithubDirectoryDownloader._read_plugin_root_from_manifest(filesystem, has_filters=True)
            if plugin_root_path:
                logger.info(
                    "Discovered pluginRoot='%s' from marketplace.json — applying filters to that directory",
                    plugin_root_path,
                )

        normalized_include = {name.lower() for name in include_directories} if include_directories else None
        normalized_exclude = {name.lower() for name in exclude_directories} if exclude_directories else set()

        if has_filters and plugin_root_path:
            plugin_root_parts = Path(plugin_root_path).parts
            top_level_dir = plugin_root_parts[0]

            for remote_item in filesystem.ls("", detail=False):
                item_path = str(remote_item)
                item_name = Path(item_path).name
                if item_name in {".git", ".github"}:
                    continue

                if item_name == top_level_dir:
                    # Walk down to the actual plugin root for multi-segment paths
                    filter_remote_path = item_path
                    for segment in plugin_root_parts[1:]:
                        filter_remote_path = f"{filter_remote_path}/{segment}"

                    local_root_dest = staging_dir / Path(plugin_root_path)
                    local_root_dest.mkdir(parents=True, exist_ok=True)

                    for plugin_item in filesystem.ls(filter_remote_path, detail=False):
                        plugin_name = Path(str(plugin_item)).name
                        plugin_lower = plugin_name.lower()
                        if normalized_include is not None and plugin_lower not in normalized_include:
                            logger.debug("Skipping non-included plugin: %s", plugin_name)
                            continue
                        if plugin_lower in normalized_exclude:
                            logger.debug("Skipping excluded plugin: %s", plugin_name)
                            continue
                        plugin_dest = local_root_dest / plugin_name
                        filesystem.get(str(plugin_item), str(plugin_dest), recursive=True)
                else:
                    destination = staging_dir / item_name
                    filesystem.get(item_path, str(destination), recursive=True)
        else:
            for remote_item in filesystem.ls("", detail=False):
                item_path = str(remote_item)
                item_name = Path(item_path).name
                if item_name in {".git", ".github"}:
                    continue
                destination = staging_dir / item_name
                filesystem.get(item_path, str(destination), recursive=True)

    @classmethod
    def parse_github_uri(cls, source_uri: str) -> GitLocation:
        """Parse a github:// URI into components.

        Raises:
            ValueError: If the URI is not a valid github:// URI.
        """
        parsed = urlsplit(source_uri)
        if parsed.scheme != "github":
            raise ValueError(f"URI must use the github:// scheme, e.g. {cls._github_uri_example}")
        if parsed.fragment:
            raise ValueError("github:// URI must not include a fragment")

        query_values = parse_qs(parsed.query, keep_blank_values=True)
        unsupported_query_params = set(query_values.keys()) - {"ref"}
        if unsupported_query_params:
            unsupported = ", ".join(sorted(unsupported_query_params))
            raise ValueError(f"github:// URI supports only '?ref=' query parameter; got: {unsupported}")

        ref_values = query_values.get("ref")
        if ref_values and len(ref_values) > 1:
            raise ValueError("github:// URI must include a single '?ref=' value")
        if ref_values is not None and not ref_values[0].strip():
            raise ValueError("github:// URI '?ref=' value must not be empty")
        branch_from_query = ref_values[0].strip() if ref_values else None

        owner = parsed.netloc.strip()
        path_parts = [part for part in parsed.path.split("/") if part]

        if ":" in owner:
            repository_without_ref, separator, branch = owner.partition("@")
            if ":" not in repository_without_ref:
                raise ValueError(f"github:// URI must include owner and repo, e.g. {cls._github_uri_example}")
            legacy_owner, repo = repository_without_ref.split(":", 1)
            if not legacy_owner or not repo:
                raise ValueError(f"github:// URI must include owner and repo, e.g. {cls._github_uri_example}")
            if branch_from_query is not None and separator and branch and branch_from_query != branch:
                raise ValueError("github:// URI ref mismatch between legacy '@branch' and '?ref='")
            owner = legacy_owner
            path_value = "/".join(path_parts)
            normalized_branch = (
                branch_from_query if branch_from_query is not None else (branch if separator and branch else None)
            )
        else:
            if not owner or not path_parts:
                raise ValueError(f"github:// URI must include owner and repo, e.g. {cls._github_uri_example}")
            repo = path_parts[0]
            path_value = "/".join(path_parts[1:])
            normalized_branch = branch_from_query

        if not owner or not repo:
            raise ValueError(f"github:// URI must include owner and repo, e.g. {cls._github_uri_example}")

        return GitLocation(
            repo_url=f"https://github.com/{owner}/{repo}.git",
            owner=owner,
            repository=repo,
            path=path_value,
            branch=normalized_branch,
        )
