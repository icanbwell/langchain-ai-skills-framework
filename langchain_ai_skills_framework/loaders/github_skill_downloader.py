import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import fsspec

from langchain_ai_skills_framework.loaders.exceptions.skill_validation_error import (
    SkillValidationError,
)


@dataclass(frozen=True, slots=True)
class GitLocation:
    repo_url: str
    owner: str
    repository: str
    path: str
    branch: str | None


class GithubSkillDownloader:
    """Downloads github:// skill directories into a local skillkit cache path."""

    _github_uri_example = "github://my-org/private-skills/skills?ref=main"
    _github_token_username = "x-access-token"

    def download(self, *, skills_directory: str, github_token: str | None) -> Path:
        git_location = self.parse_github_uri(skills_directory)
        source_path = git_location.path.strip("/")
        ref = git_location.branch or "HEAD"

        cache_root = Path(".skillkit_cache").expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        key = f"{git_location.owner}/{git_location.repository}:{ref}:{source_path}"
        cache_dir_name = f"{git_location.owner}-{git_location.repository}-{sha256(key.encode('utf-8')).hexdigest()[:12]}"
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
                # fsspec's GitHub backend requires both fields when auth is used.
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
        except Exception as exc:
            raise SkillValidationError(
                "Unable to download github:// skills directory into ./.skillkit_cache"
            ) from exc

        return target_dir.resolve()

    @classmethod
    def parse_github_uri(cls, skills_directory: str) -> GitLocation:
        parsed = urlsplit(skills_directory)
        if parsed.scheme != "github":
            raise SkillValidationError(
                "GitHub skill directory must match github://<owner>/<repo>/<path>?ref=<branch>"
            )
        if parsed.fragment:
            raise SkillValidationError(
                "GitHub skill directory must not include a fragment"
            )

        query_values = parse_qs(parsed.query, keep_blank_values=True)
        unsupported_query_params = set(query_values.keys()) - {"ref"}
        if unsupported_query_params:
            unsupported = ", ".join(sorted(unsupported_query_params))
            raise SkillValidationError(
                f"GitHub skill directory supports only '?ref=' query parameter; got: {unsupported}"
            )

        ref_values = query_values.get("ref")
        if ref_values and len(ref_values) > 1:
            raise SkillValidationError(
                "GitHub skill directory must include a single '?ref=' value"
            )
        if ref_values is not None and not ref_values[0].strip():
            raise SkillValidationError(
                "GitHub skill directory '?ref=' value must not be empty"
            )
        branch_from_query = ref_values[0].strip() if ref_values else None

        owner = parsed.netloc.strip()
        path_parts = [part for part in parsed.path.split("/") if part]

        # Backward compatibility for the legacy owner:repo style while callers migrate.
        if ":" in owner:
            repository_without_ref, separator, branch = owner.partition("@")
            if ":" not in repository_without_ref:
                raise SkillValidationError(
                    f"GitHub skill directory must include owner and repo, e.g. {cls._github_uri_example}"
                )
            legacy_owner, repo = repository_without_ref.split(":", 1)
            if not legacy_owner or not repo:
                raise SkillValidationError(
                    f"GitHub skill directory must include owner and repo, e.g. {cls._github_uri_example}"
                )
            if (
                branch_from_query is not None
                and separator
                and branch
                and branch_from_query != branch
            ):
                raise SkillValidationError(
                    "GitHub skill directory ref mismatch between legacy '@branch' and '?ref='"
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
                raise SkillValidationError(
                    f"GitHub skill directory must include owner and repo, e.g. {cls._github_uri_example}"
                )
            repo = path_parts[0]
            path_value = "/".join(path_parts[1:])
            normalized_branch = branch_from_query

        if not owner or not repo:
            raise SkillValidationError(
                f"GitHub skill directory must include owner and repo, e.g. {cls._github_uri_example}"
            )

        return GitLocation(
            repo_url=f"https://github.com/{owner}/{repo}.git",
            owner=owner,
            repository=repo,
            path=path_value,
            branch=normalized_branch,
        )
