import os
from dataclasses import dataclass


@dataclass
class Config:
    email: str = os.getenv("STUDYDRIVE_EMAIL", "")
    password: str = os.getenv("STUDYDRIVE_PASSWORD", "")

    host: str = os.getenv("STUDYDRIVE_HOST", "127.0.0.1")
    port: int = int(os.getenv("STUDYDRIVE_PORT", 8000))

    s3_endpoint: str = os.getenv("STUDYDRIVE_S3_ENDPOINT", "")
    s3_secure: bool = bool(os.getenv("STUDYDRIVE_S3_SECURE", True))
    s3_region: str = os.getenv("STUDYDRIVE_S3_REGION", "us-east-1")
    s3_access_key: str = os.getenv("STUDYDRIVE_S3_ACCESS_KEY", "")
    s3_secret_key: str = os.getenv("STUDYDRIVE_S3_SECRET_KEY", "")
    s3_bucket: str = os.getenv("STUDYDRIVE_S3_BUCKET", "")

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
