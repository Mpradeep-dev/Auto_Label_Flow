FROM python:3.12-slim

# opencv/ultralytics runtime deps + ffmpeg for video frame extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# GPU build by default; override with --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# for a CPU-only image. Installed BEFORE requirements.txt so pip resolves
# ultralytics' torch dependency against this build rather than silently
# pulling a different (CPU) wheel from PyPI — see requirements.txt.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
