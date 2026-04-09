# Turin Hand Pose - Docker image
# Reproduces CPU inference environment exactly.
# For GPU: use --gpus all with nvidia-docker.

FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/

# Default: run tests
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=term-missing"]
