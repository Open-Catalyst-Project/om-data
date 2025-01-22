#!/bin/bash 

num_lines=$(wc -l < elytes.csv)
num_lines=$((num_lines-1))

# Define the numbers as a space-separated list
# Loop through each number
# Test with the first 10 systems in elytes.csv for now!
for ((i = 1; i < 10; i++)); do
    rm $i -rf
    mkdir $i
    cp lammps2omm.py ./$i
    
    #Generate and run short simulation of the solvent 
    python generatesolvent_omm.py $i
    python runsolvent.py --job_dir=./ --row_idx=$i --rpmd --nbeads=10
    
    #Generate the system and relax
    python generatesystem_omm.py $i 
    cd $i
    cp ../relaxsystem.py ./
    python relaxsystem.py $i
    cd ..
    python runsystem.py --job_dir=./ --row_idx=$i --rpmd --nbeads=10
done
