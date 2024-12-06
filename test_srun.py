#!/usr/bin/env python
import torch
import time
import os

def main():
    # SLURM 환경 변수 읽기
    job_id = os.getenv('SLURM_JOB_ID', 'Unknown')
    task_id = os.getenv('SLURM_ARRAY_TASK_ID', '0')

    # 연산 시작 시간 기록
    start_time = time.time()

    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 10초가량 걸리는 연산 수행 (Matrix Multiplication Loop)
    n = 10000  # Matrix size
    a = torch.rand(n, n, device=device)
    b = torch.rand(n, n, device=device)

    # 5번 반복 수행 (시간을 늘리기 위해)
    for i in range(5):
        result = torch.matmul(a, b)

    # 연산 종료 시간 기록
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 결과를 텍스트 파일에 저장
    output_file = f"torch_result_{job_id}_{task_id}.txt"
    with open(output_file, "w") as f:
        f.write(f"Job ID: {job_id}, Task ID: {task_id}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
        f.write(f"Result Sum: {result.sum().item()}\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

