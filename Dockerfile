FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# NVIDIA GPU ONLY - No CPU support

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Set the Hugging Face home directory for better model caching
ENV HF_HOME=/app/hf_cache
# GPU optimization environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python3 to be python for convenience
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set up working directory
WORKDIR /app

# Copy NVIDIA requirements ONLY (no CPU support)
COPY requirements-nvidia.txt .

# Upgrade pip first
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support first (EXACT versions from original chatterbox)
# Try cu124 if cu121 doesn't have torch 2.6.0
RUN pip3 install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124 || \
    pip3 install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip3 install --no-cache-dir -r requirements-nvidia.txt
# Copy the rest of the application code
COPY . .

# Create required directories for the application (fixed syntax error)
RUN mkdir -p model_cache reference_audio outputs voices logs hf_cache

# Expose the port the application will run on
EXPOSE 8004

# Command to run the application
CMD ["python3", "server.py"]