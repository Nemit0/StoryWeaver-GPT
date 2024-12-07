#!/bin/bash
#SBATCH -J TorchJobTimer         # Job 이름
#SBATCH -p 24_Fall_Student_1     # Partition 이름
#SBATCH -o OUTPUT_%A_%a.log      # 표준 출력 로그
#SBATCH -e ERROR_%A_%a.log       # 표준 에러 로그
#SBATCH --array=0-4              # 작업 배열 설정 (5개의 작업)

#SBATCH --time=00:15:00          # 최대 실행 시간
#SBATCH --mem=8G                 # 메모리 요청

# 환경 설정
hostname
source /opt/sw/anaconda3/etc/profile.d/conda.sh
conda activate             # Conda 환경 활성화

# Python 스크립트 실행
python compute_torch_with_timer.py

