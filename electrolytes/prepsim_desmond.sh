#!/bin/bash

num_lines=$(wc -l < elytes.csv)
num_lines=$((num_lines-1))

# Define the numbers as a space-separated list
# Loop through each number
for ((i = 1; i < num_lines; i++)); do
    python run_desmond.py --job_idx=$i --output_path=./
done
