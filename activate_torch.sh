srun -p 24_Fall_Student_1 -G1 --pty bash

conda activate torch220_cu118

conda list | grep torch