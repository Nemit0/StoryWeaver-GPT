# Simple demonstration of using cuda inside a docker container

This is a simple demonstration of using cuda inside a docker container. The docker container is based on the official nvidia/cuda image, and cuda version is set as 12.2

The image can be changed on both Dockerfile and docker-compose to fit the cuda version you want to use. The images are listed here: [https://hub.docker.com/r/nvidia/cuda](https://hub.docker.com/r/nvidia/cuda)

Unsupported Versions:[https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/unsupported-tags.md](https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/unsupported-tags.md)


## Prerequisites

Using docker, we do not need any conda or virtualenv, we only need the right docker images. Using
```bash
nvidia-smi
```

You can check the cuda version(if it's set up correctly). This should display something like this:
```bash
Sat Dec  7 06:39:53 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3080        Off | 00000000:29:00.0 Off |                  N/A |
|  0%   41C    P8              19W / 320W |      1MiB / 10240MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

Replace the image with your cuda version accordingly, and run
```bash
docker-compose up -d --build
```

which should run the main.py which will count all the primes up to 10000 using pytorch.