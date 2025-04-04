import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    email: str = os.getenv("STUDYDRIVE_EMAIL", "")
    password: str = os.getenv("STUDYDRIVE_PASSWORD", "")

    host: str = os.getenv("STUDYDRIVE_HOST", "127.0.0.1")
    port: int = int(os.getenv("STUDYDRIVE_PORT", 8040))

    redis_url: Optional[str] = os.getenv("STUDYDRIVE_REDIS_URL")

    cache_ttl: int = int(os.getenv("STUDYDRIVE_CACHE_TTL", 7 * 24 * 60 * 60))

    def _validate(self):
        if not self.email or not self.password:
            raise ValueError("Email and password must be provided")
        if not isinstance(self.port, int) or self.port <= 0:
            raise ValueError("Port must be a positive integer")

    def __init__(self):
        self._validate()


_config = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
