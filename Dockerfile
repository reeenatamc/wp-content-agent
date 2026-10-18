# The HTTP API in one container. The wp_mcp server runs inside it as a subprocess.
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 1000 app && mkdir /data && chown app /data
COPY --from=build --chown=app /app/.venv /app/.venv
USER app
WORKDIR /home/app
# Listens on every interface, so `wpagent serve` refuses to start without API_TOKEN.
# Mount a volume at /data to keep runs across restarts.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    CHECKPOINT_DB=/data/wpagent.sqlite
EXPOSE 8000
# Hosts such as Railway set PORT; one process, because the per-run lock is in memory.
CMD ["sh", "-c", "exec wpagent serve --port \"${PORT:-8000}\""]
