"""generatesystem.py
Author: Muhammad R. Hasyim

Script to generate initial system configuration and LAMMPS files using 
the data2lammps.py module. This script assumes that you have done the following steps:

    1. Run generatesolvent.py to generate pure solvent configurations. 
    2. Run prepopenmmsim.py for the pure solvent.
    3. Run the MD simulation of the pure solvent using OpenMM (runsolvent.py). 

These steps generate a solventdata.txt file containing a time series of the density of the solvent, 
which we can use to calculate the number of salt molecules to put inside the simulation box. 
"""

import sys
from data2lammps import *
import os
import csv 
import numpy as np

# Read which system # to simulate from command line argument
i = int(sys.argv[1]) + 1

# Load the CSV file containing systems to simulate
f = open("elytes.csv", "r")
systems = list(csv.reader(f))
comments = systems[0]

# Extract indices of columns specifying the cation, anion,and solvent
index_cat, index_cat_conc = get_indices(comments, "cation")
index_an, index_an_conc = get_indices(comments, "anion")
index_neut, index_neut_ratio = get_indices(comments, "neutral")


# Extract salt species and their concentrations
cat = get_species_and_conc(systems, i, index_cat)
cat_conc = get_species_and_conc(systems, i, index_cat_conc)#.astype(float)
an = get_species_and_conc(systems, i, index_an)
an_conc = get_species_and_conc(systems, i, index_an_conc)#.astype(float)

salt_molfrac = np.array(cat_conc+an_conc).astype(float)
salt_molfrac /= np.sum(salt_molfrac)

# Extract solvent species name and their molar ratios
neut = get_species_and_conc(systems, i, index_neut)
neut_ratio = get_species_and_conc(systems, i, index_neut_ratio)
solv_molfrac = np.array(neut_ratio).astype(float)
solv_molfrac /= np.sum(solv_molfrac)
   
species = cat+an+neut#$system[2:Ncomp+2]
molfrac = salt_molfrac.tolist()+solv_molfrac.tolist()

# Initial boxsize is always 5 nm
boxsize = 50 #In Angstrom


# Calculate how many salt species to add in the system. If units of the salt concentration 
# is in molality (units == 'mass') then, we don't need solvent density. But if the units is
# in molarity (units == 'volume'), then we need the solvent density
units = systems[i][3]

Avog = 6.023*10**23
Nmols = []
num_solv = 500
numsalt = 0
solv_mwweight = 0
salt_conc = np.array(cat_conc+an_conc).astype(float)

for j, solv in enumerate(neut):#solvspecies):
    solv_mwweight += calculate_mw(solv)*solv_molfrac[j]
print(units)
if 'volume' in units:
    # Solvent density in g/ml, obtained from averaging short MD run
    data = list(csv.reader(open(f'{i}/solventdata.txt', 'r')))
    rho = np.array([float(row[3]) for row in data[1:]])
    rho = np.mean(rho[int(len(rho)/2):]) 
    rho *= 1000 #in g/L

    molrho = rho/solv_mwweight #in mol/L
    volume = num_solv/(Avog*molrho) #in L
    numsalt = np.round(salt_conc*volume*Avog).astype(int)
elif 'mass' in units:
    #No need to look at solvent density
    mass = num_solv*solv_mwweight/Avog
    numsalt = np.round(salt_conc*mass*Avog).astype(int)

for j in range(len(cat+an)):
    Nmols.append(int(numsalt[j]))
for j in range(len(neut)):
    Nmols.append(int(num_solv*solv_molfrac[j]))

run_packmol_moltemplate(species,boxsize,Nmols,'system',str(i-1))
