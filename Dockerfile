# The voice business card server as a container image, for Fly.io. See docs/hosting-fly.md.
#
# Two things here are for a fast start, because a call can be the reason the machine starts:
# - UV_COMPILE_BYTECODE: Python's compiled files are in the image. Without them the imports take
#   about 6 s in place of 3 s, at every start.
# - The embedding model for the writing search is in the image. Without it each new machine downloads it.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# uv.toml (not copied) asks for a uv-managed Python, which only NixOS needs. Here the image's Python is right.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system \
    UV_NO_DEV=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# The packages first: this layer changes only when uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN /app/.venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBEDDING_MODEL}', cache_dir='/app/models')"

COPY README.md profile.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-editable

# What is true for every container. The paths on the volume and the public URL are in fly.toml.
ENV PATH=/app/.venv/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=8000 \
    PROFILE_PATH=/app/profile.md \
    WRITING_EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    WRITING_MODEL_DIR=/app/models \
    SAVE_CALLER_AUDIO=false

EXPOSE 8000
CMD ["serve"]
