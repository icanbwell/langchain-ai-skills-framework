from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from langchain_ai_skills_framework.github import (
    GitHubAppTokenProvider,
    GitHubTokenProvider,
    StaticTokenProvider,
)

TEST_RSA_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQCiXezK95eGIz2E
ABM/WkTMjOqDwgckThvmvG6SiiH0+zaCzdEIVJXlHuqRgpMB/v9tdJXdfYSZSl1Q
+GDen/bDZUxbQGbaqlWzRS4cfhF5FVjQvL/Big3mGBpobxGikuzE/aDRT4Uc1itA
aoJxh+1aBnoXm/mQtCj3G9h0GN4QpLjO55tNpOmtbNaBq0CB37IoQwas4RhccDRy
l91EW7VNodNENZBHZZrgdOFGmVr3RYFewM9VUGPYlpoJrQL4+2gad51YcdE6bE/E
/GYsiWTYAcunzswijG8P0z915O9Zqgyt938xFxaRLHlaE4cHY0Y0gDAsyd6YeF7M
Mh0zDS7lAgMBAAECggEAEb9ZgDADCH72nOSTNgQMbB1lDuTY+f9trlFfdrYRkyEY
asDLffEc90/jTOdsYTX5voGVVgH/ye+mdpDHqd3rT51VdM370B/5QSCpMyUWjNkn
/Zz8CtAnx8RPsqWdVFth9QBSIT7jam0Aikh6HKXCbGoz0zvR0h7XMXeCN+J193SV
LAUDYdtDVH1GfyYBVhqGHkVFvgVkTvo/sj5yeuNgogySoM9G2iShLSgpXv+Ov+vf
sXOcwMqPTKtV7yv/cbs0p2+ZN515TKGhjiH0OFWau6pGF3rMLvMKz0oLdULfB/ZH
8FAYgoSaqt60JvnQ8VFRh1n0r88eJ1iPIn2vJz2MYQKBgQDbbHIyxH/98owpOmSZ
a6womyho36JvJcUqqj0DUkJABvglrVj2qMPQl/sJMiv93H8I/FWvkUtt3L1UhtSW
4Ckp+Scb5wwe/QtZO0FHcnLAsyuyXkb01tUIjFaQ5rKNh+0BTMoQ44QvEAExAyJJ
OyIoJpSxzlj0teoqOwM9P0ZZEQKBgQC9bqyJJm8oPblQOQRX/1pJLce5gUyocysY
M+GtkQsppFJHEJotdyAt9eBWAqIDTIjERxGmXxsGahwKuVQVxEfbOURU8W9iSv83
aJpIgr0QjzlreP4x+AfjXao4MDuJwtM4/oHUWFy9GpdnOkPHPIw0q24XlKSN9BzL
qQIS3KXYlQKBgQCGVdd4g1sE40iyOQC7+PKWjZ9ozXmJ6KrUWxMthF/xCRNFJeKw
aFQx0cosMB5EtojDvJDNAvwWD62OIVnn4ObyvooWCBcgpbUb9S4bCtN8bHUVJ6jz
Xs9gA2NAJS0tfwk34YZYXqJfmcHQ+uUzxlM8F5qzXOyTLQhmwGhUR/fOsQKBgQCW
TdpYeEZ6h38iSBtKNzJMHib66b0Ja1gmPAQ004En6VnfSS0MJhlCXnVByZUDSRa7
pig6+ftXe5oEaEhvfO4G48l0HJ1kQF2AeV7xacrZ+Mp2m+oVe9fGb+s/6gVTqWIv
NsGM2w+6e/7lyTU+QKx+ngccbrSiba7raY5bqPdugQKBgQCbYdK0lFaACxifMWpL
imtjciDVGTmxzJ1C1lgD+E3DNf5dlQCRSkHJunWYMLLq0E3xoXjEv+QFdQoxQEWf
5utDJQO+XTkIDVKU+xJUxMGzZx5cACulrTVn9rh5H7eIsJazrqhEj4tCb2MXP3VX
DdZviTUT6f/DbvH27tbG+47d/A==
-----END PRIVATE KEY-----"""


class TestStaticTokenProvider:
    async def test_get_token_returns_configured_token(self) -> None:
        provider = StaticTokenProvider(token="ghp_test123")
        token = await provider.get_token()
        assert token == "ghp_test123"

    def test_get_token_sync_returns_configured_token(self) -> None:
        provider = StaticTokenProvider(token="ghp_test456")
        token = provider.get_token_sync()
        assert token == "ghp_test456"

    def test_satisfies_protocol(self) -> None:
        provider = StaticTokenProvider(token="ghp_test")
        assert isinstance(provider, GitHubTokenProvider)


class TestGitHubAppTokenProvider:
    async def test_mints_token_on_first_call(self) -> None:
        provider = GitHubAppTokenProvider(
            app_id="12345",
            private_key=TEST_RSA_PRIVATE_KEY,
            installation_id="67890",
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "token": "ghs_xxx",
            "expires_at": "2099-01-01T00:00:00Z",
        }
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            token = await provider.get_token()

            assert token == "ghs_xxx"
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "/app/installations/67890/access_tokens" in call_args[0][0]
            assert call_args[1]["headers"]["Accept"] == "application/vnd.github+json"

    async def test_caches_token_within_expiry_window(self) -> None:
        provider = GitHubAppTokenProvider(
            app_id="12345",
            private_key=TEST_RSA_PRIVATE_KEY,
            installation_id="67890",
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "token": "ghs_cached",
            "expires_at": "2099-01-01T00:00:00Z",
        }
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            token1 = await provider.get_token()
            token2 = await provider.get_token()

            assert token1 == "ghs_cached"
            assert token2 == "ghs_cached"
            mock_client.post.assert_called_once()

    async def test_refreshes_expired_token(self) -> None:
        provider = GitHubAppTokenProvider(
            app_id="12345",
            private_key=TEST_RSA_PRIVATE_KEY,
            installation_id="67890",
        )

        expired_time = datetime.now(timezone.utc) + timedelta(seconds=100)
        fresh_time = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response_expired = Mock()
        mock_response_expired.json.return_value = {
            "token": "ghs_expired",
            "expires_at": expired_time.isoformat().replace("+00:00", "Z"),
        }
        mock_response_expired.raise_for_status = Mock()

        mock_response_fresh = Mock()
        mock_response_fresh.json.return_value = {
            "token": "ghs_fresh",
            "expires_at": fresh_time.isoformat().replace("+00:00", "Z"),
        }
        mock_response_fresh.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.side_effect = [mock_response_expired, mock_response_fresh]
            mock_client_class.return_value = mock_client

            token1 = await provider.get_token()
            assert token1 == "ghs_expired"

            await asyncio_sleep(0.1)

            token2 = await provider.get_token()
            assert token2 == "ghs_fresh"
            assert mock_client.post.call_count == 2

    def test_get_token_sync_works(self) -> None:
        provider = GitHubAppTokenProvider(
            app_id="12345",
            private_key=TEST_RSA_PRIVATE_KEY,
            installation_id="67890",
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "token": "ghs_sync",
            "expires_at": "2099-01-01T00:00:00Z",
        }
        mock_response.raise_for_status = Mock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            token = provider.get_token_sync()

            assert token == "ghs_sync"
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "/app/installations/67890/access_tokens" in call_args[0][0]
            assert call_args[1]["headers"]["Accept"] == "application/vnd.github+json"

    def test_satisfies_protocol(self) -> None:
        provider = GitHubAppTokenProvider(
            app_id="12345",
            private_key=TEST_RSA_PRIVATE_KEY,
            installation_id="67890",
        )
        assert isinstance(provider, GitHubTokenProvider)


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
