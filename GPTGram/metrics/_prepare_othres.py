
import os
import glob
import re
import numpy as np
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Prepare chat data')
parser.add_argument('--folder', type=str, help='Directory containing the chat files')

# Parse the command line arguments
args = parser.parse_args()

# Directory containing the chat files
base_dir = args.folder
# Get a list of all .txt files in the base directory
chat_files = glob.glob(os.path.join(base_dir, "*/*_chat.txt"))

# List of phrases that if found in a line, that line will be omitted
omitted_phrases = ["image omitted",
                   "sticker omitted",
                   "GIF omitted",
                   "video omitted",
                   "document omitted",
                   "audio omitted",
                   "Contact card omitted"]

# Regular expression to match timestamps
timestamp_regex = r"\[\d{2}/\d{2}/\d{2}, \d{2}:\d{2}:\d{2}\] "

# A list to store all tokens from all files
data_tokens = []
# Loop over all chat files
for chat_file in chat_files:
    # Open the current chat file
    with open(chat_file, 'r') as infile:
        # Read all lines from the chat file
        lines = infile.readlines()
        # Filter out the lines that contain any of the omitted phrases
        filtered_lines = [line for line in lines if not any(phrase in line for phrase in omitted_phrases)]
        # Remove timestamps from the lines
        no_timestamps = [re.sub(timestamp_regex, "", line) for line in filtered_lines]
        # Save the lines in no_timestamps into lines:
        lines = no_timestamps
        # Add lines to the data_tokens list
        data_tokens.extend(lines).casefold()


test_tokens = data_tokens

# Output file paths
test_output_file = os.path.join('test.txt')

# Write the tokens to the txt output files
with open(test_output_file, 'w') as outfile:
    outfile.write('\n'.join(test_tokens))


