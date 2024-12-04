# Dockerfile
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

WORKDIR /app

COPY . /app

RUN apt update && apt install -y wget vim python3 python3-pip curl && \
    # echo 'alias python="python3"' >> ~/.bashrc && \
    # echo 'alias pip="pip3"' >> ~/.bashrc && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version

RUN . ~/.bashrc

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt