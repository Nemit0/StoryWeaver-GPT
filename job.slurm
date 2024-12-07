#!/bin/bash
#SBATCH --job-name=mlgroup1
#SBATCH --partition=24_Fall_Student_1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=mlgroup1_%j.out
#SBATCH --error=mlgroup1_%j.err

# Load Singularity module
module load singularity

# Print the hostname and date for logging purposes
echo "Running on host $(hostname)"
echo "Time is $(date)"
echo "Directory is $(pwd)"

# Activate GPU support with --nv
singularity exec --nv \
    --bind $PWD:/app \
    --bind ./data:/data \ 
    --bind ./model:/model \
    --bind ./.data:/.data \
    --bind ./logs:/logs \

    docker://nemit0/mlgroup1:test \
    python3 /app/train_transformer.py