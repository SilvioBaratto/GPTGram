#!/bin/bash

# Ask the user for the folder path
echo "Please enter the folder path:"
read folder_path

# Check if the input folder path exists
if [ -d "$folder_path" ]
then
    # Run the python script with the user-specified folder path
    python3 prepare.py --folder=$folder_path
else
    echo "Invalid path. Please enter a valid folder path."
fi
