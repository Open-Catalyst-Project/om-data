"""generatesolvent.py
Author: Muhammad R. Hasyim

Script to generate initial solvent configuration and LAMMPS files using 
the molbuilder.py module. The solvent configuration is needed to run a simulation 
of pure solvent later with OpenMM. Such simulation is needed if we want to figure out
what is the average density of the solvent. 
"""
import sys
import molbuilder as mb
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
    index_solv, index_solv_ratio = mb.get_indices(comments, "solvent")

    # Extract solvent species name  and their molar ratios
    solv = mb.get_species_and_conc(systems, row_idx, index_solv)
    solv_ratio = mb.get_species_and_conc(systems, row_idx, index_solv_ratio)
    species = solv
    molfrac = np.array(solv_ratio).astype(float)#int)
    molfrac = molfrac/np.sum(molfrac)
    print(solv)
    
    #Initial boxsize is set to 8 nm. 
    rho = 0.5 #g/mL
    solv_mwweight = sum(mb.calculate_mw(solv)*solv_frac for solv, solv_frac in zip(solv, molfrac))
    rho *= 1000 #in g/L
    num_solv = 500
    molrho = rho/solv_mwweight #in mol/L
    Avog = 6.022e23
    volume = num_solv/molrho*1e27/Avog #A3
    boxsize = volume**(1/3)
    
    #Assume some number of solvents, we'll readjust later. 
    Natoms = []
    Nmols = []
    for j in range(len(solv)):
        elements, counts = mb.extract_elements_and_counts(solv[j])
        if int(num_solv*molfrac[j]) < 1:
            Nmols.append(1)
            Natoms.append(sum(counts))
        else:
            Nmols.append(int(num_solv*molfrac[j]))
            Natoms.append(sum(counts)*int(num_solv*molfrac[j]))
    print(boxsize)
    #Run Packmol, followed up by moltemplate 
    mb.run_system_builder([],[],solv,Nmols,'solvent',str(row_idx),boxsize=boxsize,mdengine='openmm')
    lmm.prep_openmm_md("solvent",[],[],solv,Nmols,str(row_idx))
else:
    print("Solvent does not exist. Not an error, but check if system is a pure moltent salt/ionic liquid.")
