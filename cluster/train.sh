#!/bin/bash

# Ask the user for the number of GPUs
echo "Please enter the number of GPUs:"
read ngpus

# Ask the user for the number of CPUs per task
echo "Please enter the number of CPUs per task:"
read ncpus

# Ask the user for the time needed
echo "Please enter the time required (in the format HH:MM:SS):"
read time_needed

# Ask the user for the partition
echo "Please enter the partition to use:"
read partition

# Check if the inputs are valid
if ! [[ "$ngpus" =~ ^[0-9]+$ ]] || ! [[ "$ncpus" =~ ^[0-9]+$ ]] || ! [[ "$time_needed" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]] || [[ -z "$partition" ]]
then
    echo "Error: Invalid input(s)"
    exit 1
fi

# Create a temporary job script
jobscript=$(mktemp)

# Write the job script
cat << EOF > $jobscript
#!/bin/bash
#SBATCH --job-name=GPTGram
#SBATCH --output=GPTGram.out
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gpus=$ngpus
#SBATCH --cpus-per-task=$ncpus
#SBATCH --time=$time_needed

# Train the model
srun torchrun --standalone --nproc_per_node=$ngpus ../cmd/train.py
EOF

# Submit the job script
sbatch $jobscript

# Clean up the job script
rm $jobscript



