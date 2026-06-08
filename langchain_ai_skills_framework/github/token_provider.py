from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Protocol, runtime_checkable

import httpx
import jwt

logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"
_TOKEN_EXPIRY_BUFFER_SECONDS = 300


@runtime_checkable
class GitHubTokenProvider(Protocol):
    """Protocol for providing GitHub authentication tokens."""

    async def get_token(self) -> str:
        """Get a valid GitHub token asynchronously."""
        ...

    def get_token_sync(self) -> str:
        """Get a valid GitHub token synchronously."""
        ...


class StaticTokenProvider:
    def __init__(self, *, token: str) -> None:
        self._token = token

    async def get_token(self) -> str:
        return self._token

    def get_token_sync(self) -> str:
        return self._token


class GitHubAppTokenProvider:
    def __init__(
        self,
        *,
        app_id: str,
        private_key: str,
        installation_id: str,
    ) -> None:
        self._app_id = app_id
        self._private_key = private_key
        self._installation_id = installation_id
        self._cached_token: str | None = None
        self._cached_expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        async with self._lock:
            if self._is_token_valid():
                return self._cached_token  # type: ignore[return-value]
            return await self._mint_token()

    def get_token_sync(self) -> str:
        if self._is_token_valid():
            return self._cached_token  # type: ignore[return-value]
        return self._mint_token_sync()

    def _is_token_valid(self) -> bool:
        return self._cached_token is not None and time.time() < self._cached_expires_at - _TOKEN_EXPIRY_BUFFER_SECONDS

    def _build_jwt(self) -> str:
        now = int(time.time())
        payload = {
            "iat": now - 60,
            "exp": now + 600,
            "iss": self._app_id,
        }
        return jwt.encode(payload, self._private_key, algorithm="RS256")

    async def _mint_token(self) -> str:
        app_jwt = self._build_jwt()
        url = f"{_GITHUB_API_BASE}/app/installations/{self._installation_id}/access_tokens"
        headers = {
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        self._cached_token = data["token"]
        self._cached_expires_at = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00")).timestamp()
        logger.debug("Minted new GitHub App installation token (expires %s)", data["expires_at"])
        return self._cached_token

    def _mint_token_sync(self) -> str:
        app_jwt = self._build_jwt()
        url = f"{_GITHUB_API_BASE}/app/installations/{self._installation_id}/access_tokens"
        headers = {
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
        }
        with httpx.Client() as client:
            response = client.post(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        self._cached_token = data["token"]
        self._cached_expires_at = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00")).timestamp()
        logger.debug("Minted new GitHub App installation token (expires %s)", data["expires_at"])
        return self._cached_token
