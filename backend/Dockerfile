# syntax=docker/dockerfile:1
# The syntax directive above turns on BuildKit's extended Dockerfile
# features — specifically the `--mount=type=cache` used below, which lets
# pip reuse previously-downloaded wheels (torch's CUDA wheel alone is
# multi-gigabyte) across separate `docker compose build` runs instead of
# re-fetching them from the network every time an earlier layer gets
# invalidated (a requirements.txt change, `--no-cache`, `docker system
# prune`, a fresh CI runner). A normal code-only rebuild was already cheap
# before this — Docker's own layer cache skips both pip installs entirely
# since neither COPY below changes — this only speeds up the cases where
# that layer cache doesn't apply.
FROM python:3.12-slim

# opencv/ultralytics runtime deps, ffmpeg for video frame extraction.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# GPU build by default; override with --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# for a CPU-only image. Installed BEFORE requirements.txt so pip resolves
# ultralytics' torch dependency against this build rather than silently
# pulling a different (CPU) wheel from PyPI — see requirements.txt.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url ${TORCH_INDEX_URL} torch torchvision

COPY requirements.txt requirements-server.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-server.txt

COPY . .

# This image runs the server profile: Celery + Redis + Postgres.
ENV ALF_TASK_QUEUE=celery

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
