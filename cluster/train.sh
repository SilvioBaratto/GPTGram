#!/bin/bash

#SBATCH --job-name=GPTGram_job
#SBATCH --output=GPTGram_job.out
#SBATCH --partition=DGX
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

# Activate the conda environment
source activate GPTGram

# Prepare the data
srun python3 ../dataset/prepare_data.py --folder=../../data/chats/

# Train the model
srun python3 ../cmd/train.py

