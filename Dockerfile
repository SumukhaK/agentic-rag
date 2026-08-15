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

# libgomp1 (GNU OpenMP): onnxruntime - a real dependency of both
# fastembed (this project's sparse/local embedding backend) and
# markitdown's magika file-type detector - dynamically links against
# libgomp.so.1 at runtime but doesn't bundle it, and python:3.11-slim
# doesn't ship it. Without this, the first fastembed/magika model load
# (likely the first real request) fails with
# "OSError: libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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

# The venv's own bin directory first on PATH - every subsequent `python`/
# `uvicorn` (CMD, HEALTHCHECK) resolves to the project's actual installed
# environment, not the base image's bare system interpreter, and CMD can
# invoke `uvicorn` directly instead of through the `uv run` wrapper (see
# the CMD comment below for why that distinction matters for shutdown).
ENV PATH="/app/.venv/bin:${PATH}"

# Runs as a non-root user - a container escape or dependency
# vulnerability shouldn't hand an attacker root inside the container.
# Everything up to this point (the venv, installed packages, app code)
# was created as root, so ownership has to move with the user switch -
# a non-root user without read/execute access to a root-owned .venv is
# a common enough Docker mistake to guard against explicitly rather
# than assume the base image's default umask happens to allow it.
# /app/data is created explicitly (not just implied by VOLUME below) so
# the recursive chown actually reaches it before VOLUME turns it into a
# mount point - VOLUME auto-creating a not-yet-existing directory is not
# guaranteed to preserve the preceding USER's ownership, which would
# otherwise leave a fresh named/anonymous volume root-owned and
# unwritable by appuser at `docker run` time.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
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

# Invokes the venv's uvicorn binary directly, not via `uv run uvicorn
# ...` - uv's own maintainers document that `uv run` does not
# execve()-replace itself with the spawned process (it stays alive as
# the parent "to provide better error messages on failure"), and
# multiple upstream reports describe its SIGTERM/SIGINT forwarding to
# the child as unreliable. If `uv` were PID 1, a `docker stop` might
# never reach uvicorn's graceful-shutdown handling, blocking for the
# full stop grace period before Docker SIGKILLs the whole tree.
# Invoking uvicorn directly makes it PID 1, receiving signals straight
# from the kernel. --factory: agentic_rag.api.main:create is a function
# (builds Settings()/the app only when actually called), not a bare
# module-level `app` object - see main.py's own docstring for why.
CMD ["uvicorn", "agentic_rag.api.main:create", "--factory", "--host", "0.0.0.0", "--port", "8000"]
