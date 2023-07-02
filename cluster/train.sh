#!/bin/bash

#SBATCH --job-name="script"
#SBATCH --output=log.out
#SBATCH --partition=EPYC
#SBATCH	--nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --nodelist=epyc[005]

module load conda/23.3.1
srun python ../cmd/train.py

