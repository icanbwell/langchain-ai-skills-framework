import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import fsspec


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

    def download(
        self, *, source_uri: str, github_token: str | None, cache_path: Path
    ) -> Path:
        """Download a github:// URI to a local directory.

        Args:
            source_uri: github://owner/repo/path?ref=branch
            github_token: Optional GitHub token for private repos.
            cache_path: Local directory for cached downloads.

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
        key = f"{git_location.owner}/{git_location.repository}:{ref}:{source_path}"
        cache_dir_name = (
            f"{git_location.owner}-{git_location.repository}"
            f"-{sha256(key.encode('utf-8')).hexdigest()[:12]}"
        )
        target_dir = cache_root / cache_dir_name

        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

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
            if source_path:
                filesystem.get(source_path, str(target_dir), recursive=True)
            else:
                for remote_item in filesystem.ls("", detail=False):
                    item_path = str(remote_item)
                    if item_path in {".git", ".github"}:
                        continue
                    destination = target_dir / Path(item_path).name
                    filesystem.get(item_path, str(destination), recursive=True)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "Unable to download github:// directory into cache"
            ) from exc

        return target_dir.resolve()

    @classmethod
    def parse_github_uri(cls, source_uri: str) -> GitLocation:
        """Parse a github:// URI into components.

        Raises:
            ValueError: If the URI is not a valid github:// URI.
        """
        parsed = urlsplit(source_uri)
        if parsed.scheme != "github":
            raise ValueError(
                f"URI must use the github:// scheme, e.g. {cls._github_uri_example}"
            )
        if parsed.fragment:
            raise ValueError("github:// URI must not include a fragment")

        query_values = parse_qs(parsed.query, keep_blank_values=True)
        unsupported_query_params = set(query_values.keys()) - {"ref"}
        if unsupported_query_params:
            unsupported = ", ".join(sorted(unsupported_query_params))
            raise ValueError(
                f"github:// URI supports only '?ref=' query parameter; got: {unsupported}"
            )

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
                raise ValueError(
                    f"github:// URI must include owner and repo, e.g. {cls._github_uri_example}"
                )
            legacy_owner, repo = repository_without_ref.split(":", 1)
            if not legacy_owner or not repo:
                raise ValueError(
                    f"github:// URI must include owner and repo, e.g. {cls._github_uri_example}"
                )
            if (
                branch_from_query is not None
                and separator
                and branch
                and branch_from_query != branch
            ):
                raise ValueError(
                    "github:// URI ref mismatch between legacy '@branch' and '?ref='"
                )
            owner = legacy_owner
            path_value = "/".join(path_parts)
            normalized_branch = (
                branch_from_query
                if branch_from_query is not None
                else (branch if separator and branch else None)
            )
        else:
            if not owner or not path_parts:
                raise ValueError(
                    f"github:// URI must include owner and repo, e.g. {cls._github_uri_example}"
                )
            repo = path_parts[0]
            path_value = "/".join(path_parts[1:])
            normalized_branch = branch_from_query

        if not owner or not repo:
            raise ValueError(
                f"github:// URI must include owner and repo, e.g. {cls._github_uri_example}"
            )

        return GitLocation(
            repo_url=f"https://github.com/{owner}/{repo}.git",
            owner=owner,
            repository=repo,
            path=path_value,
            branch=normalized_branch,
        )
