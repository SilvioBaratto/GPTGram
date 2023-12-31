#!/bin/bash

# Ask the user for the number of GPUs
echo "Please enter the number of GPUs or press enter to use none:"
read ngpus

# Ask the user for the number of CPUs per task
echo "Please enter the number of CPUs per task:"
read ncpus

# Ask the user for the memory needed
echo "Please enter the memory required (in GB):"
read mem

# Ask the user for the time needed
echo "Please enter the time required (in the format HH:MM:SS):"
read time_needed

# Ask the user for the partition
echo "Please enter the partition to use:"
read partition

# Check if the inputs are valid
if ! [[ "$ngpus" =~ ^[0-9]+$ ]] || ! [[ "$ncpus" =~ ^[0-9]+$ ]] || ! [[ "$mem" =~ ^[0-9]+$ ]] || ! [[ "$time_needed" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]] || [[ -z "$partition" ]]
then
    echo "Error: Invalid input(s)"
    exit 1
fi

# Create a temporary job script
jobscript=$(mktemp)

# If ngpus is empty or 0, do not include it in the script
if [[ -z "$ngpus" ]] || [[ "$ngpus" == 0 ]]
then
    gpu_string=""
else
    gpu_string="#SBATCH --gpus=$ngpus"
fi

# Write the job script
cat << EOF > $jobscript
#!/bin/bash
#SBATCH --job-name=GPTGram
#SBATCH --output=GPTGram.out
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --tasks=1
$gpu_string
#SBATCH --cpus-per-task=$ncpus
#SBATCH --mem=${mem}gb
#SBATCH --time=$time_needed

# Navigate to parent directory and install the library
cd ..
pip install -r requirements.txt
pip install .

# Return to the original directory
cd cluster/

EOF

# Resume training from an existing model
echo "srun torchrun --standalone --nproc_per_node=${ngpus:-0} ../cmd/train.py --init_from=gpt2-large --batch_size=4 --block_size=1024 --learning_rate=3e-5 --gradient_accumulation=32 --decay_lr=False --walltime=$time_needed" >> $jobscript

# Submit the job script
sbatch $jobscript

# Clean up the job script
rm $jobscript
