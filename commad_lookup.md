
---

## 📝 **Bash Script: `gpu.job`**

```bash
#!/bin/bash

#SBATCH --job-name=gpu-v100         # Name of the job
#SBATCH --output=%j.out             # Output file (%j is the job ID)
#SBATCH --cpus-per-task=5           # Number of CPU cores requested
#SBATCH --gpus=v100                 # Request 1 NVIDIA V100 GPU
#SBATCH --time=02-00:00:00          # Maximum runtime (2 days)
#SBATCH --partition=acltr           # Partition (queue) to submit to
#SBATCH --mem=32G                   # Amount of memory requested

echo "Running on $(hostname):"      # Print hostname of the node
module load Anaconda3               # Load Anaconda3 module
source activate sd                  # Activate the BEB conda environment
python sweep.py                     # Run the Python script
```

> ✅ You can change `--partition=acltr` to `--partition=scavenge` to use the **scavenge partition**.

---

## 🛠️ **SLURM Commands**

| Command                   | Description                                                                                    |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| `sbatch sliding_diffusion_v1_sweep.job`          | **Submits** the `gpu.job` script to the SLURM scheduler.                                       |
| `squeue -u pefu` | **Shows** all your jobs in the SLURM queue (queued or running).       |
| `scancel <job_id>`        | **Cancels** a job using its job ID (you get this from `sbatch` or `squeue`).                   |
| `--partition=scavenge`    | Option to submit to the **scavenge partition**, which may run low-priority jobs on idle nodes. |
| `%j` in `--output=%j.out` | SLURM replaces `%j` with the **job ID**, useful for naming output files uniquely.              |
| `tail -f hpc_job_outputs/job.<jobid>.out`    | follow the output of a job given ID |
|  `tail -f $(ls -t hpc_job_outputs/job.*.out \| head -n 1)`    | follow the output of the latest job |


---

