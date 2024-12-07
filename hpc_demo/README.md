

#### Activate a sample bash env with a gpu
```bash
srun -p 24_Fall_Student_1 -G1 --pty bash
```

24_Fall_Student_1 is a partition name, and can be viewed with
```bash
sinfo
```