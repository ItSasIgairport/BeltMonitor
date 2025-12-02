# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    software-properties-common \
    git \
    libgl1 \
    sshpass \
    passwd \
    net-tools \
    yasm \
    libx264-dev \
    libfdk-aac-dev \
    libmp3lame-dev \
    libopus-dev \
    wget \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directories for persistence if they don't exist
RUN mkdir -p logs recordings event_recordings

# Command to run the application
CMD ["python3", "main.py"]
