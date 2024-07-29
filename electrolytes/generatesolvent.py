"""generatesolvent.py
Author: Muhammad R. Hasyim

Script to generate initial solvent configuration and LAMMPS files using 
the data2lammps.py module. The solvent configuration is needed to run a simulation 
of pure solvent later with OpenMM. Such simulation is needed if we want to figure out
what is the average density of the solvent. 
"""
import sys
import data2lammps as d2l
import lammps2omm as lmm
import os
import csv
import numpy as np

# Read which system # to simulate from command line argument
row_idx = int(sys.argv[1]) 

# Load the CSV file containing systems to simulate
with open("elytes.csv", "r") as f:
    systems = list(csv.reader(f))

# If solvent exists. We have may have pure molten salt or ionic liquid
units = systems[row_idx][3]
if units == 'volume':
    comments = systems[0]

    # Extract indices of columns specifying the solvent
    index_solv, index_solv_ratio = d2l.get_indices(comments, "solvent")

    # Extract solvent species name  and their molar ratios
    solv = d2l.get_species_and_conc(systems, row_idx, index_solv)
    solv_ratio = d2l.get_species_and_conc(systems, row_idx, index_solv_ratio)
    species = solv
    molfrac = np.array(solv_ratio).astype(float)#int)
    molfrac = molfrac/np.sum(molfrac)
    print(solv)
    #Initial boxsize is always 10 nm. 
    boxsize = 100
    num_solv = 10000
    Natoms = []
    Nmols = []
    for j in range(len(solv)):
        elements, counts = d2l.extract_elements_and_counts(solv[j])
        if int(num_solv*molfrac[j]) < 1:
            Nmols.append(1)
            Natoms.append(sum(counts))
        else:
            Nmols.append(int(num_solv*molfrac[j]))
            Natoms.append(sum(counts)*int(num_solv*molfrac[j]))

    #Next we want to cap the total number of atoms
    NMax = 5000
    count = 0
    if sum(Natoms) > NMax:
        fracs = np.array(Natoms)/sum(Natoms)
        for j, frac in enumerate(fracs):
            N = frac*NMax
            count += N
            N = int(N/(Natoms[j]/Nmols[j]))
            Nmols[j] = N
    #print(NMax,count,Nmols)
    
    #Run Packmol, followed up by moltemplate 
    d2l.run_packmol_moltemplate(species,boxsize,Nmols,'solvent',str(row_idx))
    lmm.prep_openmm_sim("solvent",[],[],solv,str(row_idx))
else:
    print("Solvent does not exist. Not an error, but check if system is a pure moltent salt/ionic liquid.")
