#!/bin/bash
#SBATCH --job-name=llama_test    # Job name
#SBATCH --partition=kempner_h100 # GPU partition
#SBATCH --account=kempner_undergrads # Account
#SBATCH --time=04:00:00         # Time limit (4 hours)
#SBATCH --mem=320G              # Total memory
#SBATCH --gres=gpu:4            # Request 4 GPUs
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16      # Request 16 CPU cores
#SBATCH --output=output.log      # Standard output log
#SBATCH --error=error.log        # Error log
#SBATCH --array=1-2%2 #2 at a time

module load python

bash my_exe_$SLURM_ARRAY_TASK_ID.sh
