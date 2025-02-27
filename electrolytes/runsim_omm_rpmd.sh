#!/bin/bash 

num_lines=$(wc -l < rpmd_elytes.csv)
num_lines=$((num_lines-1))

# Loop through each number
for ((i = 1; i < $num_lines; i++)); do
    rm $i -rf
    mkdir $i
    cp lammps2omm.py ./$i
    
    #Generate and run short simulation of the solvent 
    python generatesolvent_omm.py $i
    python runsolvent.py --job_dir=./ --row_idx=$i --rpmd --nbeads=32
    
    #Generate the system and energy-minimize
    python generatesystem_omm.py $i 
    cd $i
    cp ../relaxsystem.py ./
    python relaxsystem.py $i
    cd ..
    #Run the MD simulation
    python runsystem.py --job_dir=./ --row_idx=$i --rpmd --nbeads=32
done
