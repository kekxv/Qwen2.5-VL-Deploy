# --- START OF FILE Dockerfile ---
# 1. Choose a base image
# For GPU support, use an NVIDIA PyTorch image.
# Check NVIDIA NGC for available PyTorch versions and corresponding CUDA/cuDNN:
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# Example: PyTorch 2.1.0 with CUDA 11.8
FROM nvidia/pytorch:23.10-py3
# This image (23.10-py3) comes with:
# Ubuntu 22.04, Python 3.10, PyTorch 2.1.0a0+47c3659, CUDA 12.2.1
# If your requirements.txt specifies a different torch version with a specific CUDA (e.g., cu118),
# you might need to find a base image with CUDA 11.8 or adjust requirements.txt.
# Let's assume PyTorch from base image is fine, or requirements.txt will handle it.

# Or for a CPU-only build (simpler, no NVIDIA drivers needed on host):
# FROM python:3.10-slim

USER root

# 2. Set Environment Variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Environment variables for your application
ARG MODEL_BACKEND_ARG=huggingface
ENV MODEL_BACKEND=${MODEL_BACKEND_ARG}
ARG PIP_INDEX_URL_ARG=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL_ARG}
# Default, can be overridden at runtime
ENV HF_ENDPOINT="https://huggingface.co"
# As used in your script, use with caution
ENV CURL_CA_BUNDLE=""

# 3. Install system dependencies (if any beyond what the base image provides)
# The nvidia/pytorch base image is quite comprehensive.
# If using python:3.10-slim, you might need build-essential, git, etc.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     # any other system deps
#     && rm -rf /var/lib/apt/lists/*

# 4. Set working directory
WORKDIR /app

# 5. Copy requirements.txt and install Python dependencies
COPY requirements.txt .
# If using nvidia/pytorch base, torch might already be installed.
# The following pip install will use the specified versions or upgrade.
# Ensure the PyTorch version in requirements.txt is compatible with the base image's CUDA.
# For PyTorch with specific CUDA versions:
RUN pip install --upgrade pip && \
    pip install --requirement requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -i "$PIP_INDEX_URL"
# If using a CPU base or a base where CUDA version matches PyTorch default:
# RUN pip install --upgrade pip && pip install -r requirements.txt

# If modelscope is in requirements, it will be installed.

# 6. Copy your application code into the image
COPY config.py .
COPY device_manager.py .
COPY model_provider.py .
COPY inference_engine.py .
COPY app.py .

# 7. Expose the port your Flask app runs on
# Your script uses 5000 (default) or 8080 (macOS)
# We'll standardize on 5000 for the container.
EXPOSE 5000

# 8. Define the command to run your application
# CMD ["python", "app.py"]
# Using gunicorn for a more production-ready setup (optional)
# RUN pip install gunicorn
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
# For simplicity with `python app.py`:
CMD ["python", "app.py"]

