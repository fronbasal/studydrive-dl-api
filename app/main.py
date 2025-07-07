import io
import json
import logging
import re
from contextlib import asynccontextmanager
from functools import wraps
from typing import (
    Dict,
    TypedDict,
    Optional,
    Callable,
    Coroutine,
    Any,
    AsyncGenerator,
)

import httpx
from fastapi import FastAPI, HTTPException, Path, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from minio import Minio

from app.auth import get_session_manager, SessionManager
from app.config import get_config, Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="app/templates")


class ServicesDict(TypedDict):
    config: Config
    session: SessionManager
    minio: Optional[Minio]


services: ServicesDict = {
    "config": get_config(),
    "session": get_session_manager(),
    "minio": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_minio()
    yield


def initialize_minio():
    try:
        if not services["config"].s3_endpoint:
            logger.warning("S3 endpoint not configured, skipping Minio initialization.")
            return
        if not services["config"].s3_access_key or not services["config"].s3_secret_key:
            raise ValueError("S3 access key or secret key not provided")
        if not services["config"].s3_bucket:
            raise ValueError("S3 bucket name not provided")

        minio_client = Minio(
            endpoint=services["config"].s3_endpoint,
            secure=services["config"].s3_secure,
            region=services["config"].s3_region,
            access_key=services["config"].s3_access_key,
            secret_key=services["config"].s3_secret_key,
        )
        bucket_name = services["config"].s3_bucket

        found = minio_client.bucket_exists(bucket_name)
        if not found:
            logger.warning(f"Bucket {bucket_name} does not exist. Creating...")
            minio_client.make_bucket(bucket_name)

        services["minio"] = minio_client
        logger.info("Minio client initialized successfully.")

    except Exception as e:
        logger.error(f"Failed to initialize Minio client: {e}")
        services["minio"] = None


app = FastAPI(
    title="ShitDrive API",
    description="Exposes a single API endpoint to download StudyDrive PDF documents without signing up.",
    version="0.1.0",
    lifespan=lifespan,
)


async def fetch_preview_url(name: str, doc_id: str, cookies: Dict[str, str]) -> str:
    """
    Fetches the document page and extracts the PDF preview URL.
    Handles errors related to fetching the page or finding the URL.
    """
    url = f"https://www.studydrive.net/de/doc/{name}/{doc_id}"
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
    Cache decorator that caches the content to Minio S3 using BackgroundTasks.
    This version addresses the RuntimeWarning and ensures the PDF data
    is properly captured. While still recognizing underlying IO may block the tread.
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, StreamingResponse]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = kwargs.get("name")
            doc_id = kwargs.get("doc_id")
            download = kwargs.get("download", False)
            background_tasks: BackgroundTasks = kwargs.get("background_tasks")
            object_name = f"{prefix}/{name}/{doc_id}.pdf"
            bucket_name = services["config"].s3_bucket

            if not services.get("minio"):
                logger.warning("Minio not configured, skipping cache.")
                return await func(*args, **kwargs)

            minio_client = services["minio"]

            try:
                try:
                    minio_client.stat_object(bucket_name, object_name)
                    logger.info(f"Cache hit for {object_name}")
                    obj = minio_client.get_object(bucket_name, object_name)

                    return StreamingResponse(
                        obj.stream(),
                        media_type="application/pdf",
                        headers={
                            "Content-Disposition": f'{"attachment" if download else "inline"}; filename="{name}-{doc_id}.pdf"'
                        },
                    )

                except (
                    Exception
                ):  # More general exception required as stat_object exception is vague
                    logger.info(f"Cache miss with {object_name} going to source")

                response: StreamingResponse = await func(*args, **kwargs)
                # Use io.BytesIO as an in-memory file to buffer chunks
                pdf_buffer = io.BytesIO()

                async def generate():
                    try:
                        async for chunk in response.body_iterator:
                            pdf_buffer.write(chunk)  # Accumulate chunk into the buffer
                            yield chunk  # Pass the chunk through to the client

                    except Exception as e:
                        logger.error(f"Error streaming content: {e}")
                        raise

                # This executes the stream *and* collects all the data into pdf_buffer
                streaming_response = StreamingResponse(
                    generate(),
                    media_type="application/pdf",
                    headers=response.headers,
                )

                # Define the upload function
                async def upload_to_s3(data: io.BytesIO, bucket: str, object_name: str):
                    try:
                        data_length = data.getbuffer().nbytes
                        data.seek(0)  # Reset buffer position to the beginning
                        logger.info(
                            f"Uploading object {object_name} to bucket {bucket} with size {data_length / 1024:.2f} KB"
                        )
                        minio_client.put_object(bucket, object_name, data, data_length)
                        logger.info(f"Successfully cached {object_name} in Minio")
                    except Exception as s3_err:
                        logger.error(f"Error uploading to S3: {s3_err}")

                # After sending, schedule the upload task
                async def upload_after_response(
                    final_buffer: io.BytesIO, b_tasks: BackgroundTasks
                ):
                    if not b_tasks:
                        logger.warning(
                            "BackgroundTasks not provided, cannot schedule upload."
                        )
                        return
                    b_tasks.add_task(
                        upload_to_s3, final_buffer, bucket_name, object_name
                    )

                await upload_after_response(
                    pdf_buffer, background_tasks
                )  # schedule background task.
                return streaming_response

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
@cache(prefix="")
async def get_document(
    name: str,
    background_tasks: BackgroundTasks,
    doc_id: str = Path(..., regex=r"^\d+$"),
    download: bool = False,
) -> StreamingResponse:
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
