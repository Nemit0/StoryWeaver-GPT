
# HPC@SICHPC

## 융합전공 HPC 서버 정보
### 1.1 HPC 사용을 위한 기초 정보
본 HPC 서버는 과학지능정보 융합전공 학생들과 관련 학과 교수, 연구원 등이 융합전공 과목 강의 중 실습과 관련 연구에 사용하실 수 있습니다.
- 계산 자원에 접근하려면 서버에 계정을 생성해야 하고, 이후 SSH를 이용해 서버에 접속해 사용합니다.
- 계산량이 많은 작업은 Slurm을 이용해 계산 노드 자원을 할당받아 처리합니다.

### 1.2 계정 생성하기
#### 실습용 계정 (수강생)
- 학번을 기준으로 `<USER_ID>` 계정을 일괄 생성합니다 (예: `s2023123456`).
- 수업 진행 학기만 사용하고, 계정과 홈디렉토리는 삭제됩니다.

#### 연구용 계정 (연구원 등)
- 교수, 조교 등은 관리자에게 별도로 요청해 계정을 생성할 수 있습니다.
- 장기 미사용자의 경우 계정과 홈디렉토리가 삭제될 수 있습니다.

---

## 사용방법
### 2.1 서버 접속하기
```bash
[LOCAL_ID@localhost ~]$ ssh sichpc.khu.ac.kr -p 222 -l <USER_ID>
```

### 2.2 소프트웨어 사용하기
#### 2.2.1 anaconda/python
- 최신 패키지 사용을 위해 Anaconda를 사용합니다.
```bash
[USER_ID@sichpc ~]$ conda activate
```

#### 2.2.2 matlab
- MATLAB R2023b를 사용 중이며, 다음 명령어로 실행할 수 있습니다:
```bash
[USER_ID@sichpc ~]$ matlab
[USER_ID@sichpc ~]$ matlab -nodisplay -r "script_name; exit"
```

#### 2.2.4 docker/singularity/apptainer
- Singularity를 이용해 Docker 이미지를 실행합니다:
```bash
[USER_ID@sichpc ~]$ singularity run docker://YOUR_DOCKER_IMAGE
```

### 2.3 Slurm 이용해서 자원 할당받기
- 계산 노드 자원을 Slurm으로 할당받아 작업을 처리합니다.
- 주요 명령어:
  - `srun`: 표준 출력으로 결과를 표시.
  - `sbatch`: 스크립트 실행.
  - `squeue`: 작업 상태 확인.
  - `scancel`: 작업 종료.

---

## Slurm 사용 예시
### 3.1 srun
- 명령어를 커맨드라인으로 실행:
```bash
srun -J jobname -p partition_names executable [args ...]
```

### 3.2 sbatch
- Python 스크립트를 실행하는 예시:

#### Python Script (`compute_pi.py`)
```python
#!/usr/bin/env python
import numpy as np
import os

def getenv(var, default=''):
    return os.environ.get(var, str(default))

jobId = int(getenv('SLURM_JOB_ID', 0))
jobSection = int(getenv('SLURM_ARRAY_TASK_ID', 0))
np.random.seed(jobId * 1000 + jobSection)

n = 100000
rx = np.random.uniform(-1, 1, n)
ry = np.random.uniform(-1, 1, n)
r2 = rx * rx + ry * ry
nIn = (r2 < 1).sum()

print(nIn, n, nIn / n * 4)
```

#### Bash Script (`script.sh`)
```bash
#!/bin/bash
#SBATCH -J MYTESTJOB
#SBATCH -p 24SPRING
#SBATCH -o OUTPUT_%A_%a.log
#SBATCH -e OUTPUT_%A_%a.err

hostname

source /opt/sw/anaconda3/etc/profile.d/conda.sh
conda activate

python test.py
```

#### Job Submission
```bash
[USER_ID@sichpc ~]$ sbatch -a 0 -p <partition name> -G1 script.sh
```