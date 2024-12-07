#!/bin/bash

# SBATCH -J MYTESTJOB
# SBATCH -p 24_Fall_Student_1
# SBATCH -o ./.logs/OUTPUT_%A_%a.log
# SBATCH -e ./.logs/OUTPUT_%A_%a.err

hostname

# Using default given conda environment
# source /opt/sw/anaconda3/etc/profile.d/conda.sh
# conda activate torch220_cu118

# Using custom conda environment
# Note if the env is not defined, it may not work.
source /opt/sw/anaconda3/etc/profile.d/conda.sh
conda activate /home/$LOGNAME/.conda/envs/torchenv

python train_transformer.py