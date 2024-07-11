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
import lammps2omm as lmm
import os
import csv 
import numpy as np
from sklearn.cluster import KMeans

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

units = systems[row_idx][3]

Avog = 6.023*10**23
Nmols = []
Natoms = []
num_solv = 5000
numsalt = 0

salt_conc = np.array(cat_conc+an_conc).astype(float)
solv_mwweight = sum(d2l.calculate_mw(solv)*solv_frac for solv, solv_frac in zip(neut, solv_molfrac))

if 'volume' == units:
    # Solvent density in g/ml, obtained from averaging short MD run
    data = np.loadtxt(f'{row_idx-1}/solventdata.txt', skiprows=1, usecols=3,delimiter=',')
    # Reshape data for clustering
    data_reshaped = data.reshape(-1, 1)

    # Use K-means to cluster the data into 3 clusters (assuming plateau is one of the clusters)
    kmeans = KMeans(n_clusters=10, random_state=0).fit(data_reshaped)

    # Get the cluster labels and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Identify the plateau cluster as the one with the smallest variance
    plateau_cluster = np.argmin([np.var(data[labels == i]) for i in range(3)])

    # Find the start index of the plateau region
    plateau_start_index = np.where(labels == plateau_cluster)[0][0]

    # Compute the mean of the plateau region
    plateau = data[labels == plateau_cluster]
    rho = np.mean(plateau[len(plateau)//2:])
    rho *= 1000 #in g/L
    
    molrho = rho/solv_mwweight #in mol/L
    volume = num_solv/(Avog*molrho) #in L
    numsalt = np.round(salt_conc*volume*Avog).astype(int)
elif 'mass' == units:
    #No need to look at solvent density
    mass = 1e-3*num_solv*solv_mwweight/Avog #mw is in g/mol, convert to kg/mol
    numsalt = np.round(salt_conc*mass*Avog).astype(int)
elif 'number' == units or 'Number' == units:
    salt_molfrac = salt_conc/np.sum(salt_conc)
    numsalt = np.round(salt_molfrac*num_solv).astype(int)


salt = cat+an
for j in range(len(salt)):
    elements, counts = d2l.extract_elements_and_counts(salt[j])
    if numsalt[j] < 1:
        Nmols.append(1)
        Natoms.append(sum(counts))
    else:
        Nmols.append(int(numsalt[j]))
        Natoms.append(sum(counts)*int(numsalt[j]))

for j in range(len(neut)):
    elements, counts = d2l.extract_elements_and_counts(neut[j])
    if int(num_solv*solv_molfrac[j]) < 1:
        Nmols.append(1)
        Natoms.append(sum(counts))
    else:
        Nmols.append(int(num_solv*solv_molfrac[j]))
        Natoms.append(sum(counts)*int(num_solv*solv_molfrac[j]))

#Next we want to cap the total number of atoms
NMax = 2500
count = 0
if sum(Natoms) > NMax:
    fracs = np.array(Natoms)/sum(Natoms)
    for j, frac in enumerate(fracs):
        N = frac*NMax
        count += N
        N = int(np.round((N/(Natoms[j]/Nmols[j]))))
        Nmols[j] = N

#print(NMax,count,Nmols)

d2l.run_packmol_moltemplate(species,boxsize,Nmols,'system',str(row_idx-1))
lmm.prep_openmm_sim("system",cat,an,neut,str(row_idx-1))
