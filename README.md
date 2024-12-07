# mlgroup1

## Project Overview

This project is part of course "Deep Learning" from Kyung Hee University, and aims to recreate a Decoder-only Transformer model using pytorch, but with just the tensor. We aim to recreate everything from scratch, including the attention mechanism, the positional encoding, the feedforward network, and the transformer block.

## 

### Initializing project

#### Using venv and pip

1. Make an venv
```bash <linux>
python3 -m venv venv && source ./venv/bin/activate
```

```powershell <windows>
python -m venv venv && .\venv\Scripts\Activate
```

2. install requirements
```bash <Linux>
pip install -r requirements.txt
```

#### Using docker

1. If there is a gpu with cuda enviromnent, make sure you have necessary nvidia drivers and docker installed, and edit the Dockerfile and docker-compose.yaml to your version of cuda.

#### Using HPC

```bash
sbatch -a 1 -p 24_Fall_Student_1 -G1 train_transformer.sh
```