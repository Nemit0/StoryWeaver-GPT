# HPC Demo

This is a simple guide to using HPC@SICHPC.

#### Activate a sample bash env with a gpu
```bash
srun -p 24_Fall_Student_1 -G1 --pty bash
```

24_Fall_Student_1 is a partition name, and can be viewed with
```bash
sinfo
```

#### Use sbatch to run a script
```bash
sbatch -p 24_Fall_Student_1 -G1 sample.sh
```

Refer to hpc_guide.md for more information.