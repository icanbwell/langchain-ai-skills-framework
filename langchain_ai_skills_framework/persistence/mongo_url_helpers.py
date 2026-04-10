import urllib.parse


class MongoUrlHelpers:
    """Utilities for constructing MongoDB connection strings."""

    @staticmethod
    def add_credentials_to_mongo_url(
        *, mongo_url: str, username: str | None, password: str | None
    ) -> str:
        """Inject URL-encoded credentials into a MongoDB connection string.

        If *username* or *password* is ``None`` or empty the original URL is
        returned unchanged.  Existing credentials in the URL are replaced.
        """
        if not username or not password:
            return mongo_url

        parsed = urllib.parse.urlparse(mongo_url)
        encoded_username = urllib.parse.quote_plus(username)
        encoded_password = urllib.parse.quote_plus(password)

        host = parsed.netloc.split("@")[1] if "@" in parsed.netloc else parsed.netloc
        netloc = f"{encoded_username}:{encoded_password}@{host}"

        return urllib.parse.urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )