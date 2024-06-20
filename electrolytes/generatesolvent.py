"""generatesolvent.py
Author: Muhammad R. Hasyim

Script to generate initial solvent configuration and LAMMPS files using 
the data2lammps.py module. The solvent configuration is needed to run a simulation 
of pure solvent later with OpenMM. Such simulation is needed if we want to figure out
what is the average density of the solvent. 
"""
import sys
import data2lammps as d2l
import os
import csv
import numpy as np

# Read which system # to simulate from command line argument
row_idx = int(sys.argv[1]) + 1

# Load the CSV file containing systems to simulate
with open("elytes.csv", "r") as f:
    systems = list(csv.reader(f))
comments = systems[0]

# Extract indices of columns specifying the solvent
index_neut, index_neut_ratio = d2l.get_indices(comments, "neutral")

# Extract solvent species name  and their molar ratios
neut = d2l.get_species_and_conc(systems, row_idx, index_neut)
neut_ratio = d2l.get_species_and_conc(systems, row_idx, index_neut_ratio)

# If solvent exists. We have may have pure molten salt or ionic liquid
if neut:
    species = neut
    molfrac = np.array(neut_ratio).astype(int)
    molfrac = molfrac/np.sum(molfrac)

    #Initial boxsize is always 5 nm. 
    boxsize = 50 

    num_solv = 500
    Nmols = [int(num_solv*frac) for frac in molfrac]

    #Run Packmol, followed up by moltemplate 
    d2l.run_packmol_moltemplate(species,boxsize,Nmols,'solvent',str(row_idx-1))
else:
    print("Solvent does not exist. Not an error, but check if system is a pure moltent salt/ionic liquid.")
