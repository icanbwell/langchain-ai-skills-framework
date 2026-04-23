from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

import httpx

from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class GitHubMarketplacePublisher:
    """Publishes skills to a GitHub marketplace repo via the Git Data API.

    Supports two modes controlled by ``use_branch``:

    * **Branch mode** (default): creates a deterministic branch
      ``skill-publish/{plugin}/{skill}`` and opens (or updates) a pull
      request targeting ``base_branch``.
    * **Direct mode**: commits directly to ``base_branch`` with a
      fast-forward ref update.  No branch or PR is created.

    Both modes produce atomic multi-file commits (SKILL.md + resources +
    scripts) using the low-level Git Data API.
    """

    def __init__(
        self,
        *,
        access_token: str,
        repo: str,
        base_branch: str = "main",
        use_branch: bool = True,
    ) -> None:
        self._access_token = access_token
        self._repo = repo  # "owner/repo"
        self._base_branch = base_branch
        self._use_branch = use_branch
        self._base_url = "https://api.github.com"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"token {self._access_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "LangchainAISkills-MarketplacePublisher",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def publish_skill(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        skill_content: str,
        resources: dict[str, str],
        scripts: dict[str, str],
        user_id: str,
    ) -> str:
        """Create or update a commit that adds/updates a skill in the marketplace.

        Returns the PR HTML URL (branch mode) or commit SHA (direct mode).

        Raises:
            httpx.HTTPStatusError: If any GitHub API call fails.
            ValueError: If any name contains path traversal segments.
        """
        self._validate_path_segment(plugin_name, "plugin_name")
        self._validate_path_segment(skill_name, "skill_name")

        branch = f"skill-publish/{plugin_name}/{skill_name}"
        files = self._build_file_map(
            plugin_name=plugin_name,
            skill_name=skill_name,
            skill_content=skill_content,
            resources=resources,
            scripts=scripts,
        )
        commit_message = f"Publish skill: {plugin_name}/{skill_name}"
        pr_title = f"Publish skill: {plugin_name}/{skill_name}"
        pr_body = (
            f"## Skill Published\n\n"
            f"- **Plugin**: {plugin_name}\n"
            f"- **Skill**: {skill_name}\n"
            f"- **Shared by**: {user_id}\n"
            f"- **Timestamp**: {datetime.now(timezone.utc).isoformat()}\n"
        )

        if self._use_branch:
            return await self._create_or_update_pr(
                branch=branch,
                files=files,
                commit_message=commit_message,
                pr_title=pr_title,
                pr_body=pr_body,
            )
        return await self._commit_directly(
            files=files,
            commit_message=commit_message,
        )

    async def unpublish_skill(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        user_id: str,
    ) -> str | None:
        """Create a commit that removes a skill directory from the marketplace.

        Returns the PR HTML URL (branch mode), commit SHA (direct mode),
        or ``None`` when no files exist under the skill directory.

        Raises:
            httpx.HTTPStatusError: If any GitHub API call fails.
            ValueError: If any name contains path traversal segments.
        """
        self._validate_path_segment(plugin_name, "plugin_name")
        self._validate_path_segment(skill_name, "skill_name")

        branch = f"skill-unpublish/{plugin_name}/{skill_name}"
        skill_dir = f"{plugin_name}/skills/{skill_name}"
        commit_message = f"Remove skill: {plugin_name}/{skill_name}"
        pr_title = f"Remove skill: {plugin_name}/{skill_name}"
        pr_body = (
            f"## Skill Removed\n\n"
            f"- **Plugin**: {plugin_name}\n"
            f"- **Skill**: {skill_name}\n"
            f"- **Unshared by**: {user_id}\n"
            f"- **Timestamp**: {datetime.now(timezone.utc).isoformat()}\n"
        )

        if self._use_branch:
            return await self._create_or_update_removal_pr(
                branch=branch,
                directory_prefix=skill_dir,
                commit_message=commit_message,
                pr_title=pr_title,
                pr_body=pr_body,
            )
        return await self._remove_directly(
            directory_prefix=skill_dir,
            commit_message=commit_message,
        )

    # ------------------------------------------------------------------
    # Path safety
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_path_segment(value: str, label: str) -> None:
        """Reject path segments that could escape the intended directory.

        Raises ``ValueError`` for empty strings, absolute paths, ``..``
        segments, or embedded path separators.
        """
        if not value or not value.strip():
            raise ValueError(f"{label} must not be empty")
        p = PurePosixPath(value)
        if p.is_absolute():
            raise ValueError(f"{label} must not be an absolute path: {value!r}")
        if ".." in p.parts:
            raise ValueError(f"{label} must not contain '..': {value!r}")

    # ------------------------------------------------------------------
    # File map builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_file_map(
        *,
        plugin_name: str,
        skill_name: str,
        skill_content: str,
        resources: dict[str, str],
        scripts: dict[str, str],
    ) -> dict[str, str]:
        """Build path -> content mapping matching the marketplace directory layout."""
        GitHubMarketplacePublisher._validate_path_segment(plugin_name, "plugin_name")
        GitHubMarketplacePublisher._validate_path_segment(skill_name, "skill_name")
        for name in resources:
            GitHubMarketplacePublisher._validate_path_segment(name, "resource name")
        for name in scripts:
            GitHubMarketplacePublisher._validate_path_segment(name, "script name")

        base = f"{plugin_name}/skills/{skill_name}"
        files: dict[str, str] = {f"{base}/SKILL.md": skill_content}
        for name, content in resources.items():
            files[f"{base}/references/{name}"] = content
        for name, content in scripts.items():
            files[f"{base}/scripts/{name}"] = content
        return files

    # ------------------------------------------------------------------
    # Git Data API operations
    # ------------------------------------------------------------------

    async def _get_ref_sha(self, client: httpx.AsyncClient, branch: str) -> str:
        url = f"{self._base_url}/repos/{self._repo}/git/ref/heads/{branch}"
        resp = await client.get(url, headers=self._headers)
        resp.raise_for_status()
        return str(resp.json()["object"]["sha"])

    async def _get_commit_tree_sha(self, client: httpx.AsyncClient, commit_sha: str) -> str:
        url = f"{self._base_url}/repos/{self._repo}/git/commits/{commit_sha}"
        resp = await client.get(url, headers=self._headers)
        resp.raise_for_status()
        return str(resp.json()["tree"]["sha"])

    async def _create_blob(self, client: httpx.AsyncClient, content: str) -> str:
        url = f"{self._base_url}/repos/{self._repo}/git/blobs"
        payload = {
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "encoding": "base64",
        }
        resp = await client.post(url, headers=self._headers, json=payload)
        resp.raise_for_status()
        return str(resp.json()["sha"])

    async def _create_tree(
        self,
        client: httpx.AsyncClient,
        base_tree_sha: str,
        tree_entries: list[dict[str, Any]],
    ) -> str:
        url = f"{self._base_url}/repos/{self._repo}/git/trees"
        payload = {"base_tree": base_tree_sha, "tree": tree_entries}
        resp = await client.post(url, headers=self._headers, json=payload)
        resp.raise_for_status()
        return str(resp.json()["sha"])

    async def _create_commit(
        self,
        client: httpx.AsyncClient,
        tree_sha: str,
        parent_sha: str,
        message: str,
    ) -> str:
        url = f"{self._base_url}/repos/{self._repo}/git/commits"
        payload = {
            "message": message,
            "tree": tree_sha,
            "parents": [parent_sha],
        }
        resp = await client.post(url, headers=self._headers, json=payload)
        resp.raise_for_status()
        return str(resp.json()["sha"])

    async def _create_branch(self, client: httpx.AsyncClient, branch: str, sha: str) -> None:
        url = f"{self._base_url}/repos/{self._repo}/git/refs"
        payload = {"ref": f"refs/heads/{branch}", "sha": sha}
        resp = await client.post(url, headers=self._headers, json=payload)
        if resp.status_code == 422:
            await self._update_branch(client, branch, sha, force=True)
        else:
            resp.raise_for_status()

    async def _update_branch(
        self,
        client: httpx.AsyncClient,
        branch: str,
        sha: str,
        force: bool = False,
    ) -> None:
        url = f"{self._base_url}/repos/{self._repo}/git/refs/heads/{branch}"
        payload = {"sha": sha, "force": force}
        resp = await client.patch(url, headers=self._headers, json=payload)
        if not force and resp.status_code in {409, 422}:
            raise RuntimeError(
                f"Failed to update branch '{branch}' with a fast-forward-only update "
                "because the branch advanced on GitHub. Re-read the branch ref, rebuild "
                "the commit, and retry the publish operation."
            )
        resp.raise_for_status()

    # ------------------------------------------------------------------
    # PR operations
    # ------------------------------------------------------------------

    async def _find_open_pr(self, client: httpx.AsyncClient, head_branch: str) -> dict[str, Any] | None:
        owner = self._repo.split("/")[0]
        url = f"{self._base_url}/repos/{self._repo}/pulls"
        params = {
            "head": f"{owner}:{head_branch}",
            "base": self._base_branch,
            "state": "open",
        }
        resp = await client.get(url, headers=self._headers, params=params)
        resp.raise_for_status()
        prs = resp.json()
        return prs[0] if prs else None

    async def _create_pull_request(
        self,
        client: httpx.AsyncClient,
        head: str,
        title: str,
        body: str,
    ) -> str:
        url = f"{self._base_url}/repos/{self._repo}/pulls"
        payload = {
            "title": title,
            "body": body,
            "head": head,
            "base": self._base_branch,
        }
        resp = await client.post(url, headers=self._headers, json=payload)
        resp.raise_for_status()
        return str(resp.json()["html_url"])

    async def _update_pull_request(
        self,
        client: httpx.AsyncClient,
        pr_number: int,
        title: str,
        body: str,
    ) -> str:
        url = f"{self._base_url}/repos/{self._repo}/pulls/{pr_number}"
        payload = {"title": title, "body": body}
        resp = await client.patch(url, headers=self._headers, json=payload)
        resp.raise_for_status()
        return str(resp.json()["html_url"])

    # ------------------------------------------------------------------
    # Tree listing (for deletions)
    # ------------------------------------------------------------------

    async def _get_tree_recursive(self, client: httpx.AsyncClient, tree_sha: str) -> list[dict[str, Any]]:
        url = f"{self._base_url}/repos/{self._repo}/git/trees/{tree_sha}"
        resp = await client.get(url, headers=self._headers, params={"recursive": "1"})
        resp.raise_for_status()
        tree: list[dict[str, Any]] = resp.json()["tree"]
        return tree

    # ------------------------------------------------------------------
    # High-level orchestration — branch mode
    # ------------------------------------------------------------------

    async def _create_or_update_pr(
        self,
        *,
        branch: str,
        files: dict[str, str],
        commit_message: str,
        pr_title: str,
        pr_body: str,
    ) -> str:
        """Branch mode: create blobs, tree, commit, branch, and PR."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            base_sha = await self._get_ref_sha(client, self._base_branch)
            base_tree_sha = await self._get_commit_tree_sha(client, base_sha)

            tree_entries: list[dict[str, Any]] = []
            for path, content in files.items():
                blob_sha = await self._create_blob(client, content)
                tree_entries.append(
                    {
                        "path": path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob_sha,
                    }
                )

            new_tree_sha = await self._create_tree(client, base_tree_sha, tree_entries)
            commit_sha = await self._create_commit(client, new_tree_sha, base_sha, commit_message)
            await self._create_branch(client, branch, commit_sha)

            existing_pr = await self._find_open_pr(client, branch)
            if existing_pr:
                pr_url = await self._update_pull_request(client, existing_pr["number"], pr_title, pr_body)
                logger.info("Updated existing PR %s for branch '%s'", pr_url, branch)
            else:
                pr_url = await self._create_pull_request(client, branch, pr_title, pr_body)
                logger.info("Created new PR %s for branch '%s'", pr_url, branch)

            return pr_url

    async def _create_or_update_removal_pr(
        self,
        *,
        branch: str,
        directory_prefix: str,
        commit_message: str,
        pr_title: str,
        pr_body: str,
    ) -> str | None:
        """Branch mode: create a PR that removes all files under a directory prefix."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            base_sha = await self._get_ref_sha(client, self._base_branch)
            base_tree_sha = await self._get_commit_tree_sha(client, base_sha)

            full_tree = await self._get_tree_recursive(client, base_tree_sha)

            prefix = directory_prefix.rstrip("/") + "/"
            delete_entries: list[dict[str, Any]] = [
                {
                    "path": entry["path"],
                    "mode": entry["mode"],
                    "type": entry["type"],
                    "sha": None,
                }
                for entry in full_tree
                if entry["path"].startswith(prefix) and entry["type"] == "blob"
            ]

            if not delete_entries:
                logger.info(
                    "No files found under '%s' to remove — skipping PR",
                    directory_prefix,
                )
                return None

            new_tree_sha = await self._create_tree(client, base_tree_sha, delete_entries)
            commit_sha = await self._create_commit(client, new_tree_sha, base_sha, commit_message)
            await self._create_branch(client, branch, commit_sha)

            existing_pr = await self._find_open_pr(client, branch)
            if existing_pr:
                pr_url = await self._update_pull_request(client, existing_pr["number"], pr_title, pr_body)
                logger.info("Updated existing removal PR %s", pr_url)
            else:
                pr_url = await self._create_pull_request(client, branch, pr_title, pr_body)
                logger.info("Created removal PR %s", pr_url)

            return pr_url

    # ------------------------------------------------------------------
    # High-level orchestration — direct mode
    # ------------------------------------------------------------------

    async def _commit_directly(
        self,
        *,
        files: dict[str, str],
        commit_message: str,
    ) -> str:
        """Direct mode: commit files straight to the base branch."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            base_sha = await self._get_ref_sha(client, self._base_branch)
            base_tree_sha = await self._get_commit_tree_sha(client, base_sha)

            tree_entries: list[dict[str, Any]] = []
            for path, content in files.items():
                blob_sha = await self._create_blob(client, content)
                tree_entries.append(
                    {
                        "path": path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob_sha,
                    }
                )

            new_tree_sha = await self._create_tree(client, base_tree_sha, tree_entries)
            commit_sha = await self._create_commit(client, new_tree_sha, base_sha, commit_message)

            await self._update_branch(client, self._base_branch, commit_sha)
            logger.info("Committed directly to '%s': %s", self._base_branch, commit_sha)
            return commit_sha

    async def _remove_directly(
        self,
        *,
        directory_prefix: str,
        commit_message: str,
    ) -> str | None:
        """Direct mode: remove files under a directory prefix on the base branch."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            base_sha = await self._get_ref_sha(client, self._base_branch)
            base_tree_sha = await self._get_commit_tree_sha(client, base_sha)

            full_tree = await self._get_tree_recursive(client, base_tree_sha)

            prefix = directory_prefix.rstrip("/") + "/"
            delete_entries: list[dict[str, Any]] = [
                {
                    "path": entry["path"],
                    "mode": entry["mode"],
                    "type": entry["type"],
                    "sha": None,
                }
                for entry in full_tree
                if entry["path"].startswith(prefix) and entry["type"] == "blob"
            ]

            if not delete_entries:
                logger.info(
                    "No files found under '%s' to remove — skipping commit",
                    directory_prefix,
                )
                return None

            new_tree_sha = await self._create_tree(client, base_tree_sha, delete_entries)
            commit_sha = await self._create_commit(client, new_tree_sha, base_sha, commit_message)

            await self._update_branch(client, self._base_branch, commit_sha)
            logger.info(
                "Removed '%s' directly on '%s': %s",
                directory_prefix,
                self._base_branch,
                commit_sha,
            )
            return commit_sha
