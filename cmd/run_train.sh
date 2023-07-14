#!/bin/bash

# Ask the user for the number of gpus
echo "Please enter the number of gpus:"
read ngpus

# Check if the input is an integer
if [[ "$ngpus" =~ ^[0-9]+$ ]]
then
    # Run the torchrun command with the user-specified number of processes
    torchrun --standalone --nproc_per_node=$ngpus train.py
else
    echo "Invalid input. Please enter an integer."
fi
