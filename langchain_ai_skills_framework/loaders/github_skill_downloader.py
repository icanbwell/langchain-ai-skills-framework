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
class _GitLocation:
    owner: str
    repository: str
    path: str
    ref: str | None


class GithubSkillDownloader:
    """Downloads github:// skill directories into a local skillkit cache path."""

    _github_uri_example = "github://my-org/private-skills/skills?ref=main"

    def download(self, *, skills_directory: str, github_token: str | None) -> Path:
        git_location = self._parse_github_uri(skills_directory)
        source_path = git_location.path.strip("/")
        ref = git_location.ref or "HEAD"

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
            if git_location.ref:
                storage_options["sha"] = git_location.ref
            if github_token:
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
    def _parse_github_uri(cls, skills_directory: str) -> _GitLocation:
        parsed = urlsplit(skills_directory)
        if parsed.scheme != "github":
            raise SkillValidationError(
                "GitHub skill directory must match github://<owner>/<repo>/<path>?ref=<branch>"
            )
        if parsed.fragment:
            raise SkillValidationError(
                "GitHub skill directory must not include a fragment"
            )

        owner = parsed.netloc.strip()
        path_parts = [part for part in parsed.path.split("/") if part]
        if not owner or not path_parts:
            raise SkillValidationError(
                f"GitHub skill directory must include owner and repo, e.g. {cls._github_uri_example}"
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

        return _GitLocation(
            owner=owner,
            repository=path_parts[0],
            path="/".join(path_parts[1:]),
            ref=ref_values[0].strip() if ref_values else None,
        )
