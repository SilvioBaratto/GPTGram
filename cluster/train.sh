#!/bin/bash

#SBATCH --job-name=python_job
#SBATCH --output=python_job.out
#SBATCH --partition=DGX
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00

module load python
srun python3 ../cmd/train.py

