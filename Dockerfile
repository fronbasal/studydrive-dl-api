# syntax=docker/dockerfile:1

FROM python:3.13-slim-bookworm AS builder

WORKDIR /app

COPY poetry.lock pyproject.toml ./

RUN pip install --no-cache-dir poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_VIRTUALENVS_CREATE=true
ENV POETRY_CACHE_DIR=/tmp/poetry_cache

RUN poetry install --no-root --no-interaction --without dev

COPY app ./app

FROM python:3.13-slim-bookworm AS runner

WORKDIR /app

COPY --from=builder /app/.venv ./.venv

COPY --from=builder /app/app ./app

ENV PYTHONPATH=.
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
