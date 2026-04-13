FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Use PyPI as the primary index so all packages resolve correctly, with the
# CPU-only PyTorch index as a secondary source so torch/torchvision/torchaudio
# pull lightweight CPU wheels instead of CUDA builds (nvidia-*, triton, etc.)
# which would bloat the image to 3 GB+ and cause push timeouts.
RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
