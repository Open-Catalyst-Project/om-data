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
import data2lammps as d2l
import os
import csv 
import numpy as np

# Read which system # to simulate from command line argument
row_idx  = int(sys.argv[1]) + 1

# Load the CSV file containing systems to simulate
with open("elytes.csv", "r") as f:
    systems = list(csv.reader(f))
comments = systems[0]

# Extract indices of columns specifying the cation, anion,and solvent
index_cat, index_cat_conc = d2l.get_indices(comments, "cation")
index_an, index_an_conc = d2l.get_indices(comments, "anion")
index_neut, index_neut_ratio = d2l.get_indices(comments, "neutral")

# Extract salt species and their concentrations
cat = d2l.get_species_and_conc(systems, row_idx, index_cat)
cat_conc = d2l.get_species_and_conc(systems, row_idx, index_cat_conc)#.astype(float)
an = d2l.get_species_and_conc(systems, row_idx, index_an)
an_conc = d2l.get_species_and_conc(systems, row_idx, index_an_conc)#.astype(float)


salt_molfrac = np.array(cat_conc+an_conc).astype(float)
salt_molfrac /= np.sum(salt_molfrac)


# Extract solvent species name and their molar ratios
neut = d2l.get_species_and_conc(systems, row_idx, index_neut)
neut_ratio = d2l.get_species_and_conc(systems, row_idx, index_neut_ratio)
solv_molfrac = np.array(neut_ratio).astype(float)
solv_molfrac /= np.sum(solv_molfrac)

soltorsolv = len(cat+an)*['A']+len(neut)*['B']
   
species = cat+an+neut#$system[2:Ncomp+2]
molfrac = salt_molfrac.tolist()+solv_molfrac.tolist()

# Initial boxsize is always 5 nm
boxsize = 50 #In Angstrom

# Calculate how many salt species to add in the system. If units of the salt concentration 
# is in molality (units == 'mass') then, we don't need solvent density. But if the units is
# in molarity (units == 'volume'), then we need the solvent density. If units is in moles/
# stoichiometry (units == 'numbers'), then we have a molten salt/ionic liquid system 
# and compute the mole fractions directly.

units = systems[row_idx][2]

Avog = 6.023*10**23
Nmols = []
num_solv = 500
numsalt = 0

salt_conc = np.array(cat_conc+an_conc).astype(float)
solv_mwweight = sum(calculate_mw(solv)*solv_frac for solv, solv_frac in zip(neut, solv_molfrac))

if 'volume' == units:
    # Solvent density in g/ml, obtained from averaging short MD run
    data = list(csv.reader(open(f'{i-1}/solventdata.txt', 'r')))
    rho = np.array([float(row[3]) for row in data[1:]])
    rho = np.mean(rho[int(len(rho)/2):]) 
    rho *= 1000 #in g/L

    molrho = rho/solv_mwweight #in mol/L
    volume = num_solv/(Avog*molrho) #in L
    numsalt = np.round(salt_conc*volume*Avog).astype(int)
elif 'mass' == units:
    #No need to look at solvent density
    mass = num_solv*solv_mwweight/Avog
    numsalt = np.round(salt_conc*mass*Avog).astype(int)
elif 'number' == units:
    salt_molfrac = salt_conc/np.sum(salt_conc)
    numsalt = np.round(salt_molfrac*num_solv).astype(int)
    print(numsalt)

for j in range(len(cat+an)):
    Nmols.append(int(numsalt[j]))
for j in range(len(neut)):
    Nmols.append(int(num_solv*solv_molfrac[j]))

d2l.run_packmol_moltemplate(species,boxsize,Nmols,'system',str(row_idx-1))
