# Dockerfile
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip wget vim curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version && \
    pip3 install -r requirements.txt

# CMD ["python3", "app.py"]