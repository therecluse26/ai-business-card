# The voice business card server as a container image, for Fly.io. See docs/hosting-fly.md.
#
# Two things here are for a fast start, because a call can be the reason the machine starts:
# - UV_COMPILE_BYTECODE: Python's compiled files are in the image. Without them the imports take
#   about 6 s in place of 3 s, at every start.
# - The embedding model for the writing search is in the image. Without it each new machine downloads it.
# - The model of the voice reader is in the image too, for the same reason.
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

# Pipecat Voice Reader, the "base" size (voice/voice_reader.py). The release is the one that the installed Pipecat names.
RUN /app/.venv/bin/python -c "from huggingface_hub import snapshot_download; from pipecat.classifiers.voice.model import VOICE_REPO, VOICE_REVISION; snapshot_download(VOICE_REPO, revision=VOICE_REVISION, allow_patterns=['config.json', '*-base/config.json', '*-base/model.int8.onnx'], cache_dir='/app/models')"

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
    VOICE_MODEL_DIR=/app/models \
    SAVE_CALLER_AUDIO=false

# The git commit of this build (scripts/fly-deploy.sh passes it). Each call's analytics name it, so the
# grader's scores of one version can be compared with the next. It is last: a new commit rebuilds no layer above.
ARG HARNESS_VERSION=unknown
ENV HARNESS_VERSION=${HARNESS_VERSION}

EXPOSE 8000
CMD ["serve"]
