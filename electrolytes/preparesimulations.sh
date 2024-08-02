#!/bin/bash 

num_lines=$(wc -l < elytes.csv)
num_lines=$((num_lines-1))

# Define the numbers as a space-separated list
# Loop through each number
for ((i = 0; i < num_lines; i++)); do
    rm $i -rf
    mkdir $i
    cp lammps2omm.py ./$i
    
    #Generate and run short simulation of the solvent 
    python generatesolvent.py $i
    cd $i
    cp ../runsolvent.py ./
    python runsolvent.py $i
    cd ..
    
    #Generate the system and relax
    python generatesystem.py $i 
    cd $i
    cp ../relaxsystem.py ./
    python relaxsystem.py $i
    cd ..
done