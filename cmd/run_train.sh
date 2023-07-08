#!/bin/bash

# Ask the user for the number of processes
echo "Please enter the number of processes:"
read nproc

# Check if the input is an integer
if [[ "$nproc" =~ ^[0-9]+$ ]]
then
    # Run the torchrun command with the user-specified number of processes
    torchrun --standalone --nproc_per_node=$nproc train.py
else
    echo "Invalid input. Please enter an integer."
fi
