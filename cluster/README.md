# Python SLURM Script Execution

This project contains a Python script that is intended to be run on a SLURM-managed cluster, specifically targeting the DGX nodes.

## Requirements

To install the necessary Python packages, run the following command:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes the following Python packages:

- torch
- numpy
- transformers
- datasets
- tiktoken
- wandb
- tqdm

## Running the Python Script on SLURM

We use a bash script `run_python_job.sh` to submit the Python script as a job to the SLURM scheduler.

You can submit the job with the following command:

```bash
sbatch run_python_job.sh
```

This command will schedule the Python script to be run on two nodes in the DGX partition.

## Checking Job Output

The output of the Python script will be directed to a file named `python_job.out`.

You can monitor the output of the job while it's running using the `tail` command:

```bash
tail -f python_job.out
```

## Troubleshooting

If you encounter any issues, check the `python_job.out` file for any error messages. Make sure all the Python packages listed in the `requirements.txt` file are installed and that the Python script is correctly located in the `run_python_job.sh` bash script.
```

Remember to replace the placeholders (`your_python_script.py`) with the actual name of your Python script. You can also add more specific instructions or details as necessary for your project.
