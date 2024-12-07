#!/bin/bash

# SBATCH -J MYTESTJOB
# SBATCH -p 24_Fall_Student_1
# SBATCH -o ./.logs/OUTPUT_%A_%a.log
# SBATCH -e ./.logs/OUTPUT_%A_%a.err

hostname

source /opt/sw/anaconda3/etc/profile.d/conda.sh
conda activate torch220_cu118

python torch_test.py