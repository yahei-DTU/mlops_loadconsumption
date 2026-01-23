FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV UV_SOURCE_MODE=files
ENV UV_VERSION=0.0.1
ENV UV_DISABLE_SCM=1

RUN apt update && \
    apt install --no-install-recommends -y \
        tzdata \
        build-essential \
        gcc \
        git && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy relevant code and DVC files
COPY src/ ./src/
COPY train.sh /train.sh
COPY data.dvc .
COPY .dvc/config .dvc/config
COPY models/ ./models/
COPY tests/ ./tests/
COPY configs/ ./configs/
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .

# Create a minimal git repo so UV doesn't fail
RUN git init \
 && git config user.email "ci@local" \
 && git config user.name "CI" \
 && git add . \
 && git commit -m "docker build"

RUN --mount=type=cache,target=/root/.cache/uv uv sync

# Ensure DVC doesn't require Git
RUN uv run dvc config core.no_scm true

RUN chmod +x /train.sh

ENTRYPOINT ["/train.sh"]
