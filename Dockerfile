# Stage 1: build virtual environment strictly from uv.lock
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# Stage 2: minimal non-root runtime image
FROM python:3.12-slim-bookworm AS runtime

RUN groupadd --system --gid 1001 findex && \
    useradd --system --uid 1001 --gid findex --create-home findex

WORKDIR /app

COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /app/src /app/src
COPY --chown=findex:findex data ./data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH="/app/data/prod_index.bin" \
    PORT="8000"

USER findex

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["sh", "-c", "findex serve --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1}"]
