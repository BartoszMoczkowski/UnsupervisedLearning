FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN useradd -m -s /bin/bash unsupervised \
    && mkdir -p /workspaces/UnsupervisedLearning/.venv \
    && chown -R unsupervised:unsupervised /workspaces
USER unsupervised
WORKDIR /workspaces/UnsupervisedLearning