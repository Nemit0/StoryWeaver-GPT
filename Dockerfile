# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip wget vim curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version && \
    pip3 install -r requirements.txt

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CMD ["python3", "app.py"]