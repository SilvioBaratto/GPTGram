# GPTGram Job Submission Script

This Bash script is used to automate the submission of training jobs for GPTGram to an HPC cluster using Slurm workload manager. It prompts the user for various parameters related to the computational resources required and generates a temporary Slurm job submission script, which it then submits to the cluster.

## Usage
To use the script, execute it in your terminal. You'll be asked to enter several parameters:

1. **Number of GPUs**: The number of GPUs to be used. If none are to be used, simply press enter.
2. **Number of CPUs per task**: The number of CPUs required per task.
3. **Memory required (in GB)**: The amount of memory needed, in gigabytes.
4. **Time required**: The estimated time needed for your job in the format HH:MM:SS.
5. **Partition**: The partition of the HPC cluster that the job should be submitted to.

The script will validate your inputs to make sure they're in the right format. If any invalid input is provided, the script will throw an error and exit.

You can also add any extra flags for the torchrun command when prompted.

The script generates a temporary Slurm job submission script that includes all the necessary `#SBATCH` options according to your inputs. It then navigates to the parent directory, installs the necessary requirements from the `requirements.txt` file, installs the library, and returns to the original directory.

You can specify whether you want to resume training from an existing model or start training from scratch. Depending on your answer, the script will add the appropriate torchrun command to the job submission script.

Finally, the script submits the job using the `sbatch` command and cleans up by removing the temporary job submission script.

Please note that this script assumes the existence of a `requirements.txt` file in the parent directory and a `train.py` script in the `../cmd/` directory. Make sure these files exist in the correct locations before running the script.

## Example

Here is a sample interaction with the script:

```
$ ./job_submission.sh
Please enter the number of GPUs or press enter to use none:
2
Please enter the number of CPUs per task:
4
Please enter the memory required (in GB):
16
Please enter the time required (in the format HH:MM:SS):
01:00:00
Please enter the partition to use:
main
Please enter any extra flags for the torchrun command (or press enter for none):
--init_from=resume --folder=../cmd/out/ 
```

In this example, a job is submitted to the "main" partition of the HPC cluster, requesting 2 GPUs, 4 CPUs, and 16GB of memory, with a time limit of 1 hour. The `--init_from` and `--folder` flags are also passed to the torchrun command.
