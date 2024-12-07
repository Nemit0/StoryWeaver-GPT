# HPC Demo

This is a simple guide to using HPC@SICHPC.

#### Activate a sample bash env with a gpu
```bash
srun -p 24_Fall_Student_1 -G1 --pty bash
```

Generally, once we have a bash session with a gpu, we use conda to manage our environment. 
```bash
conda env list # list all environments
```

This will list all environments. Output should be something like this:
```bash
# conda environments:
#
base                     /opt/sw/anaconda3
r_env                    /opt/sw/anaconda3/envs/r_env
torch201_cu118           /opt/sw/anaconda3/envs/torch201_cu118
torch220_cu118           /opt/sw/anaconda3/envs/torch220_cu118
```

Then we can activate enviromnent with torch to use pytorch.
```bash
conda activate torch220_cu118
```

Where we can run scripts manually for debugging.

#### Submit a job

24_Fall_Student_1 is a partition name, and can be viewed with
```bash
sinfo
```

This returns names of partitions that we can access.

#### Use sbatch to run a script
```bash
sbatch -p 24_Fall_Student_1 -G1 sample.sh
```

Refer to hpc_guide.md for more information.