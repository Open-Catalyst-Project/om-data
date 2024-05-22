#!/bin/bash 

num_lines=$(wc -l < systems.csv)

for ((i = 0; i < num_lines; i++)); do
    mkdir $i
    cp prepopenmmsim.py ./$i
    cp lammps2omm.py ./$i
    
    #Generate and run short simulation of the solvent 
    python generatesolvent.py $i
    cd $i
    python prepopenmmsim.py solvent
    cp ../runsolvent.py ./
    python runsolvent.py
    cd ..
    
    #Generate the system
    python generatesystem.py $i 
    cd $i
    python prepopenmmsim.py system
    cd ..
done
