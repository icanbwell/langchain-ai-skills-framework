from typing import Protocol, runtime_checkable

from motor.motor_asyncio import AsyncIOMotorDatabase


@runtime_checkable
class MongoDatabaseFactory(Protocol):
    """Protocol for creating MongoDB database connections.

    The consuming application implements this to provide MongoDB
    connections using its own configuration and credentials.
    """

    def create_database(self) -> AsyncIOMotorDatabase[dict[str, object]]:
        """Return an AsyncIOMotorDatabase instance."""
        ...
