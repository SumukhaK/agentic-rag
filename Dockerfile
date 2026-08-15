# Containerizes the FastAPI app only (Phase 8 "deployment hardening,"
# scoped with the user to just this piece - Qdrant stays embedded/on-disk
# inside this container's volume, per docs/REQUIREMENTS.md's existing
# "local/embedded mode for now" decision; Ollama is expected to keep
# running on the host, reached over the network the same way local dev
# already reaches it).
#
# IMPORTANT - this image has never been built or run: Docker isn't
# installed in this project's development environment (confirmed via
# `docker --version` -> "command not found"), so this was written
# carefully against documented `uv`/Python conventions but could not be
# verified with a real `docker build`. Treat it as a first draft to
# validate the first time it's actually built, not a proven artifact -
# see PROJECT_TRACKER.md's Phase 8 entry for the full caveat.

FROM python:3.11-slim

# uv is this project's own dependency manager (pyproject.toml/uv.lock) -
# installed via its official standalone installer rather than pip, so
# the image doesn't need pip to bootstrap it.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency manifests first so this layer is cached across builds
# that only change application code, not dependencies.
COPY pyproject.toml uv.lock ./

# --frozen: fail if uv.lock is out of date rather than silently
# resolving a different dependency set than what's been tested.
# --no-dev: exclude the dev group (httpx, pytest) - not needed at
# runtime, and smaller images are less attack surface.
# --no-install-project: install dependencies only in this layer; the
# project itself is installed in a separate step below, after the
# source is copied, so an app-code-only change doesn't invalidate this
# (slower) dependency-install layer's cache.
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY README.md ./

RUN uv sync --frozen --no-dev

# Runs as a non-root user - a container escape or dependency
# vulnerability shouldn't hand an attacker root inside the container.
# Everything up to this point (the venv, installed packages, app code)
# was created as root, so ownership has to move with the user switch -
# a non-root user without read/execute access to a root-owned .venv is
# a common enough Docker mistake to guard against explicitly rather
# than assume the base image's default umask happens to allow it.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Qdrant's embedded storage and the sync snapshot both persist here by
# default (see src/agentic_rag/config.py's qdrant_storage_path/
# sync_snapshot_path) - mount a volume at /app/data to keep them across
# container restarts. The watched document corpus itself
# (WATCHED_FOLDER_PATH, required, no default - see config.py) has no
# default path and must be bind-mounted from the host explicitly.
VOLUME ["/app/data"]

EXPOSE 8000

# Liveness-style check ("is the process up"), not readiness
# ("/health/ready", which also checks Qdrant/Ollama reachability) -
# routing Docker's own HEALTHCHECK at the readiness endpoint would mark
# a perfectly healthy container "unhealthy" the moment Ollama is
# transiently unreachable, which is a query-time problem, not a
# container-restart-worthy one. A real orchestrator (Kubernetes, etc.)
# wiring separate liveness/readiness probes should point them at
# /health and /health/ready respectively - out of scope here, since
# this Dockerfile targets `docker run`, not a specific orchestrator.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uv", "run", "uvicorn", "agentic_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
