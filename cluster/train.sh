#!/bin/bash

#SBATCH --job-name=GPTGram_job
#SBATCH --output=GPTGram_job.out
#SBATCH --partition=DGX
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

srun python ../cmd/train.py

