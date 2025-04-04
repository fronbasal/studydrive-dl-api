import io
import json
import logging
import re
from typing import Dict, TypedDict, Optional, Callable, Coroutine, Any, AsyncGenerator
from contextlib import asynccontextmanager
from functools import wraps

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from redis import RedisError

from app.auth import get_session_manager, SessionManager
from app.config import get_config, Config


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    config = get_config()
    services["config"] = config

    await initialize_redis()
    yield
    if services["redis"]:
        await services["redis"].aclose()


app = FastAPI(
    title="ShitDrive API",
    description="Exposes a single API endpoint to download StudyDrive PDF documents without signing up.",
    version="0.1.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="app/templates")


class ServicesDict(TypedDict):
    config: Config
    session: SessionManager
    redis: Optional[redis.Redis]


services: ServicesDict = {
    "config": get_config(),
    "session": get_session_manager(),
    "redis": None,
}


async def initialize_redis():
    try:
        if services["config"].redis_url:
            redis_client = redis.from_url(
                services["config"].redis_url, decode_responses=False
            )

            try:
                await redis_client.ping()
                logger.info("Redis connection successful.")
            except ConnectionError as e:
                logger.error(f"Failed to ping Redis: {e}")
                services["redis"] = None
                return

            logger.info("Connected to Redis successfully.")
            services["redis"] = redis_client

        else:
            logger.warning("Redis URL not configured.  Caching disabled.")
    except Exception as e:
        logger.error("Failed to connect to Redis: %s", e)
        services["redis"] = None


async def fetch_preview_url(name: str, doc_id: str, cookies: Dict[str, str]) -> str:
    """
    Fetches the document page and extracts the PDF preview URL.
    Handles errors related to fetching the page or finding the URL.
    """
    url = f"https://www.studydrive.net/en/doc/{name}/{doc_id}"
    async with httpx.AsyncClient(
        cookies=cookies,
        follow_redirects=True,
        timeout=30.0,
    ) as client:
        try:

            logger.debug(f"Fetching document page: {url}")
            response = await client.get(url)
            response.raise_for_status()
            logger.debug(f"Successfully fetched document page for {name}/{doc_id}")

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Error fetching document page {url}: {e.response.status_code} - {e}"
            )

            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Failed to fetch document page: {e.response.status_code}",
            )
        except httpx.RequestError as e:
            logger.error(f"Network error fetching document page {url}: {e}")
            raise HTTPException(
                status_code=504, detail=f"Network error fetching document page: {e}"
            )

        match = re.search(r'"file_preview":([^,}]+)', response.text)
        if not match:
            logger.error(f"Failed to find 'file_preview' URL in page source for {url}")
            logger.error(f"Full Response text: {response.text}")
            raise HTTPException(
                status_code=500,
                detail="Could not extract preview URL from document page.",
            )

        preview_url = json.loads(match.group(1))

        logger.info(f"Extracted preview URL for {name}/{doc_id}: {preview_url}")

        if not preview_url.startswith("https://"):
            logger.error(f"Invalid preview URL format found: {preview_url}")
            raise HTTPException(
                status_code=500, detail="Invalid preview URL format found."
            )

        if "/teaser/" in preview_url:
            logger.warning(
                f"Received teaser instead of document for {name}/{doc_id}: {preview_url}"
            )
            raise HTTPException(
                status_code=403,
                detail="Received teaser instead of full document. This might indicate content requiring premium access or similar restrictions.",
            )

        return preview_url


async def fetch_pdf_content(
    pdf_url: str, cookies: Dict[str, str]
) -> AsyncGenerator[bytes, Any]:
    """Fetches the PDF content from the provided URL."""
    async with httpx.AsyncClient(
        cookies=cookies, follow_redirects=True, timeout=120.0
    ) as client:
        try:
            logger.debug(f"Fetching PDF from URL: {pdf_url}")
            response = await client.get(pdf_url)
            response.raise_for_status()

            async def content_generator():
                async for chunk in response.aiter_bytes():
                    yield chunk

            return content_generator()

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error fetching PDF from {pdf_url}: {e.response.status_code} - {e}"
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Failed to fetch PDF data: {e.response.status_code}",
            )
        except httpx.RequestError as e:
            logger.error(f"Network error fetching PDF from {pdf_url}: {e}")
            raise HTTPException(
                status_code=504, detail=f"Network error fetching PDF: {e}"
            )
        except Exception as e:
            logger.exception(
                f"Unexpected error fetching PDF from {pdf_url}: {type(e).__name__}, {e}"
            )
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while fetching the PDF.",
            )


def cache(prefix: str):
    """
    Cache decorator that caches the streaming response.  The cache will store
    PDF data in chunks.
    """

    def decorator(func: Callable[..., Coroutine[any, any, StreamingResponse]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = kwargs.get("name")
            doc_id = kwargs.get("doc_id")
            download = kwargs.get("download", False)
            cache_key = f"{prefix}:{name}:{doc_id}"

            if not services.get("redis"):
                logger.warning("Redis not configured, skipping cache.")
                return await func(*args, **kwargs)

            try:

                cached_generator = await services["redis"].get(cache_key)
                if cached_generator:
                    logger.info(f"Cache hit for {cache_key}")

                else:
                    logger.info(f"Cache miss with {cache_key} going to source")
                    response: StreamingResponse = await func(*args, **kwargs)

                    async def stream_and_cache():
                        try:
                            async for chunk in response.body_iterator:

                                await services["redis"].append(cache_key, chunk)
                                yield chunk
                        except Exception as e:
                            logger.error("problem setting value")
                            raise e
                        finally:

                            await services["redis"].expire(
                                cache_key, services["config"].cache_ttl
                            )

                    return StreamingResponse(
                        stream_and_cache(),
                        media_type="application/pdf",
                        headers=response.headers,
                    )

                return StreamingResponse(
                    io.BytesIO(cached_generator),
                    status_code=200,
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f'{"attachment" if download else "inline"}; filename="{name}-{doc_id}.pdf"'
                    },
                )

            except RedisError as e:
                logger.error(f"Redis error: {e}")
                return await func(*args, **kwargs)
            except HTTPException as e:
                raise e
            except Exception as e:
                logger.exception("Unexpected error during caching.")
                return await func(*args, **kwargs)

        return wrapper

    return decorator


@app.get(
    "/doc/{name}/{doc_id}",
    description="Retrieve a PDF document from StudyDrive",
    tags=["Document"],
    responses={
        200: {"description": "PDF document"},
        401: {"description": "Authentication failed"},
        403: {"description": "Teaser received instead of document"},
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"},
    },
)
@cache(prefix="pdf_content")
async def get_document(
    name: str,
    doc_id: str = Path(..., regex=r"^\d+$"),
    download: bool = False,
):

    authenticated, auth_error = await services["session"].ensure_authenticated()
    if not authenticated:

        raise HTTPException(
            status_code=401, detail=f"Authentication failed: {auth_error}"
        )

    try:

        pdf_url = await fetch_preview_url(name, doc_id, services["session"].cookies)
        pdf_content_generator = await fetch_pdf_content(
            pdf_url, services["session"].cookies
        )

        return StreamingResponse(
            pdf_content_generator,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'{"attachment" if download else "inline"}; filename="{name}-{doc_id}.pdf"'
            },
        )

    except HTTPException as e:

        raise e
    except Exception as e:
        logger.exception(
            f"Unexpected error fetching preview URL for {name}/{doc_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal error occurred while fetching document details.",
        )


@app.get("/", include_in_schema=False)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        context={
            "request": request,
        },
    )


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run("app.main:app", host=config.host, port=config.port, reload=True)
