FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

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
    && rm -rf /var/lib/apt/lists/*


# FFmpeg (statik build, libx264 dahil)
WORKDIR /opt
RUN wget --no-check-certificate https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar -xf ffmpeg-release-amd64-static.tar.xz && \
    mv ffmpeg-*-amd64-static ffmpeg && \
    rm -f ffmpeg-release-amd64-static.tar.xz

# PATH içine statik ffmpeg klasörünü ekle
ENV PATH="/opt/ffmpeg:$PATH"

# Root şifresini ayarla
RUN echo "root:pass123**" | chpasswd

# SSH dizinini oluştur
RUN mkdir /var/run/sshd

# Root login’a izin ver
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH portunu dışa aç
EXPOSE 22

# Uygulama dizinine geç
WORKDIR /app
COPY . /app

# Install local wheels first (to avoid download timeouts for large files)
RUN if ls wheels/*.whl 1> /dev/null 2>&1; then \
    pip install wheels/*.whl; \
    fi

# Python bağımlılıklarını yükle
RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
RUN if [ -f requirements.txt ]; then \
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.nvidia.com \
    --extra-index-url https://pypi.nvidia.com \
    -r requirements.txt; \
    fi

# Başlangıç komutu: SSH ve uygulama birlikte çalışsın
CMD ["sh", "-c", "service ssh start && python3 main.py"]
