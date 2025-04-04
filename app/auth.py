import logging
import re
import time
from typing import Dict, Optional, Tuple

import httpx
from fastapi import HTTPException

from app.config import get_config

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self):
        self.cookies: Optional[Dict[str, str]] = None
        self.csrf_token: Optional[str] = None
        self.created_at: Optional[float] = None
        self.email: Optional[str] = None
        self.password: Optional[str] = None
        self.client = httpx.Client(follow_redirects=True)

    async def get_session_cookie(self) -> Tuple[bool, str]:
        try:
            response = self.client.get("https://www.studydrive.net/en/companies-search")
            if response.status_code != 200 or not "sd_session" in response.cookies:
                logger.error(
                    f"Failed to retrieve session cookie, status: {response.status_code}"
                )
                return False, "Failed to retrieve session cookie"

            self.cookies = dict(response.cookies)
            return True, ""
        except Exception as e:
            logger.exception("Error getting CSRF token")
            return False, str(e)

    async def login(self, email: str, password: str) -> Tuple[bool, str]:
        self.email = email
        self.password = password

        success, error = await self.get_session_cookie()
        if not success:
            return False, error

        try:
            response = self.client.post(
                "https://www.studydrive.net/login",
                json={
                    "email": email,
                    "password": password,
                    "remember": True,
                    "sduni": None,
                },
                headers={
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": "https://www.studydrive.net/en/companies-search",
                    "Content-Type": "application/json",
                    "Accept": "application/json,text/plain,*/*",
                    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:137.0) Gecko/20100101 Firefox/137.0",
                },
                cookies=self.cookies,
            )

            if response.status_code != 200:
                logger.error(
                    f"Login failed with status {response.status_code}: {response.text}"
                )
                return False, f"Login failed: HTTP {response.status_code}"

            self.cookies.update(dict(response.cookies))
            self.created_at = time.time()
            logger.info(f"Successfully logged in with {email}")

            return True, ""
        except Exception as e:
            logger.exception("Login error")
            return False, str(e)

    async def refresh_if_needed(self) -> Tuple[bool, str]:
        if not self.created_at or not self.email or not self.password:
            return False, "No active session"

        if time.time() - self.created_at < 600:
            return True, ""

        logger.info("Session older than 10 minutes, refreshing")
        return await self.login(self.email, self.password)

    async def ensure_authenticated(self) -> Tuple[bool, str]:
        config = get_config()

        if not self.cookies or not self.created_at:

            return await self.login(config.email, config.password)

        return await self.refresh_if_needed()


_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
