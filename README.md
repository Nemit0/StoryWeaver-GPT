# StoryWeaver-GPT

### Background Behind the Title
When this project first started, we were aiming to create a GPT-2 model, that were trained on r/WritingPrompts. The idea was to create a model that could generate stories based on a prompt. However, as the project progressed, we realized that the model was too large to train on our local machines, and we decided to switch to a smaller model, the Transformer model. The name StoryWeaver-GPT was kept as a homage to the original idea, and as a reminder of the original goal of the project.
The model since have been trained on much smaller shakesphere dataset, and in term of capability is questionable to just using pytorch.

## Project Overview

This project is part of course "Deep Learning" from Kyung Hee University, and aims to recreate a Decoder-only Transformer model using pytorch, but with just the tensor. We aim to recreate everything from scratch, including the attention mechanism, the positional encoding, the feedforward network, and the transformer block.

## Project Setup

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
For windows or cuda less then 12.0, head to [pytorch](https://pytorch.org/get-started/locally/) and install the correct version.

#### Using docker

1. If there is a gpu with cuda enviromnent, make sure you have necessary nvidia drivers and docker installed, and edit the Dockerfile and docker-compose.yaml to your version of cuda.

#### Using HPC

```bash
sbatch -a 1 -p 24_Fall_Student_1 -G1 <script_name>.sh
```

the train_transformer.sh file is an example of how to run the training on HPC.
