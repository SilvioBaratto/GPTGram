#!/bin/bash

# Ask the user for any extra flags for the torchrun command
echo "Please enter any extra flags for the torchrun command (or press enter for none):"
read torchrun_flags

# Run the torchrun command with the user-specified number of processes
echo python3 sample.py $torchrun_flags