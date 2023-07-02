#!/bin/bash

#SBATCH --job-name=GPTGram_job
#SBATCH --output=GPTGram_job.out
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00

# Train the model
srun python3 ../cmd/train.py


