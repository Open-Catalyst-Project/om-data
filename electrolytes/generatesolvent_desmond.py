"""generatesolvent.py
Author: Muhammad R. Hasyim

Script to generate initial solvent configuration and LAMMPS files using 
the data2lammps.py module. The solvent configuration is needed to run a simulation 
of pure solvent later with OpenMM. Such simulation is needed if we want to figure out
what is the average density of the solvent. 
"""
import sys
import molbuilder as mb
import os
import csv
import numpy as np


def generate_solvent_desmond(row_idx, systems, job_dir):
    # If solvent exists. We have may have pure molten salt or ionic liquid
    units = systems[row_idx][3]
    temperature = float(systems[row_idx][4])
    assert units == 'volume', ("Solvent does not exist. Not an error, but check if system is a pure moltent salt/ionic liquid.")
    comments = systems[0]

    # Extract indices of columns specifying the solvent
    index_solv, index_solv_ratio = mb.get_indices(comments, "solvent")

    # Extract solvent species name  and their molar ratios
    solv = mb.get_species_and_conc(systems, row_idx, index_solv)
    solv_ratio = mb.get_species_and_conc(systems, row_idx, index_solv_ratio)
    species = solv
    molfrac = np.array(solv_ratio).astype(float)#int)
    molfrac = molfrac/np.sum(molfrac)
    
    #Assume some number of solvents, we can readjust later
    num_solv = 500
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
    
    #Run Desmond system builder
    mb.prep_desmond_md('solvent',job_dir, temperature)#,mdengine="desmond")
    command, directory = mb.run_system_builder([],[],solv,Nmols,'solvent',job_dir,mdengine="desmond")
    return command, directory
