"""
This script reads chat files from a directory, filters out specific phrases and timestamps,
and writes the encoded chat content to a binary output file.

The script scans for all text files within subdirectories of the base directory. It then filters
out specific phrases (omitted images, stickers, etc.) and removes timestamps. The filtered chats
are then encoded using Byte-Pair Encoding (BPE) via the tiktoken library. The encoded tokens are 
written to a single binary output file.

This script assumes that the input chat files have a specific format, specifically that timestamps
appear in a format like "[12/12/12, 12:12:12]". If your chat files are formatted differently,
you will need to modify the script to match your format.

Usage:
    python prepare.py --folder <data_directory>

Examples:
    >>> python prepare.py --folder chats/
"""

import os
import glob
import re
import tiktoken
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
# Output file path
output_file = os.path.join('train.bin')

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
        # Encode each line using GPT-2 Byte-Pair Encoding
        enc = tiktoken.get_encoding("gpt2")
        encoded_lines = [enc.encode_ordinary(line) for line in no_timestamps]
        # Add the encoded lines to the data_tokens list
        data_tokens.extend(encoded_lines)

# Concatenate all tokens into a single numpy array
data_tokens = np.concatenate(data_tokens)

# Write the tokens to the binary output file
data_tokens.astype(np.uint16).tofile(output_file)

# Print the total number of tokens
print(f"data has {len(data_tokens):,} tokens")





