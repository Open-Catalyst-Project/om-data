#!/bin/bash 

num_lines=$(wc -l < elytes.csv)
num_lines=$((num_lines-1))

for ((i = 0; i < num_lines; i++)); do
    cd $i
    cp ../runsystem.py ./
    python runsystem.py $i
    cd ..
done
