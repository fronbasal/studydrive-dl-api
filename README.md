# Studydrive Download API

This project is a robust FastAPI-based proxy for downloading PDF documents from Studydrive without requiring end user sign-up or premium account access. It automates session management, handles authentication, and streams PDF files directly to users, with optional S3 (Minio) caching support.

Code is inspired by [gookie-dev/studydrive](https://github.com/gookie-dev/studydrive).

- **PDF Proxying:** Retrieve Studydrive documents via `/doc/{name}/{doc_id}`—either as downloads or inline view.
- **Session Automation:** Authenticates automatically (credentials/config required).
- **S3 Caching:** Optionally caches/retrieves PDFs from Minio/S3 (configured via environment).
- **Async Streaming:** Efficient non-blocking PDF delivery.
- **Templated Home:** Minimal landing page at `/`.
- **OpenAPI Docs:** Interactive documentation at `/docs`.

## Development Setup

1. **Install dependencies using poetry**
   ```
   git clone https://github.com/fronbasal/studydrive-dl-api
   cd studydrive-dl-api
   poetry install
   ```

2. **Configure Environment**

   Copy `.env.example` to `.env` and set:
   - Studydrive credentials/session (used by `app/auth.py`)
   - S3/Minio config for PDF caching (optional, see `app/config.py` for details)

3. **Run Locally**
   ```
   poetry run uvicorn app.main:app --reload
   ```
   The app will be available at `http://localhost:8000`.

4. **Linting & Formatting**

   [black](https://black.readthedocs.io/en/stable/) is used for Python code formatting:
   ```
   poetry run black .
   ```

5. **Docker Compose (optional)**

   For production or multi-service setup (e.g., Redis/Minio), use:
   ```
   docker-compose up --build
   ```

## Project Layout

- `app/main.py` — Main FastAPI app, route logic, S3 integration, error handling
- `app/auth.py` — Session management and authentication
- `app/config.py` — Environment configuration
- `app/templates/home.html` — Home page template

