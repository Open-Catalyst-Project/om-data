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
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

def load_csv(filename):
    return pd.read_csv(filename)

def get_nmols(charges, natoms, tol=1):
    prob = LpProblem("Integer_Solution_Problem", LpMinimize)
    x = []
    for i in range(len(charges)):
        if natoms[i]-tol <= 0:
            lowbound= 1
        else:
            lowbound = natoms[i]-tol
        x.append(LpVariable(f"x{i}", lowbound, natoms[i]+tol, cat='Integer'))
    prob += lpSum(coeff * var for coeff, var in zip(charges, x)) == 0
    prob.solve()
    return [int(v.varValue) for v in prob.variables()[1:]]

# Read which system # to simulate from command line argument
row_idx  = int(sys.argv[1]) 

# Load the CSV file containing systems to simulate
with open("elytes.csv", "r") as f:
    systems = list(csv.reader(f))
comments = systems[0]
# Extract indices of columns specifying the cation, anion,and solvent
index_cat, index_cat_conc = d2l.get_indices(comments, "cation")
index_an, index_an_conc = d2l.get_indices(comments, "anion")
index_solv, index_solv_ratio = d2l.get_indices(comments, "solvent")

# Extract salt species and their concentrations
cat = d2l.get_species_and_conc(systems, row_idx, index_cat)
cat_conc = d2l.get_species_and_conc(systems, row_idx, index_cat_conc)#.astype(float)
an = d2l.get_species_and_conc(systems, row_idx, index_an)
an_conc = d2l.get_species_and_conc(systems, row_idx, index_an_conc)#.astype(float)

salt_molfrac = np.array(cat_conc+an_conc).astype(float)
salt_molfrac /= np.sum(salt_molfrac)


# Extract solvent species name and their molar ratios
solv = d2l.get_species_and_conc(systems, row_idx, index_solv)
solv_ratio = d2l.get_species_and_conc(systems, row_idx, index_solv_ratio)
solv_molfrac = np.array(solv_ratio).astype(float)
solv_molfrac /= np.sum(solv_molfrac)

soltorsolv = len(cat+an)*['A']+len(solv)*['B']
   
species = cat+an+solv#$system[2:Ncomp+2]
molfrac = salt_molfrac.tolist()+solv_molfrac.tolist()

# Initial boxsize is always 10 nm
boxsize = 100 #In Angstrom

# Calculate how many salt species to add in the system. If units of the salt concentration 
# is in molality (units == 'mass') then, we don't need solvent density. But if the units is
# in molarity (units == 'volume'), then we need the solvent density. If units is in moles/
# stoichiometry (units == 'numbers'), then we have a molten salt/ionic liquid system 
# and compute the mole fractions directly. We first assume that we want to add 5000 solvent molecules
# on average. And then, we rescale things back so that the system size is controlled. 
# In the process, rounding errors cause the system to not be charge solvent. We then adjust
# how much cations and anions we want to add slightly (usually allowing subtraction/addition of one ion 
# is sufficient

units = systems[row_idx][3]

Avog = 6.023*10**23
Nmols = []
Natoms = []
num_solv = 5000
numsalt = 0

salt_conc = 0.1*np.array(cat_conc+an_conc).astype(float)
solv_mwweight = sum(d2l.calculate_mw(solv)*solv_frac for solv, solv_frac in zip(solv, solv_molfrac))

if 'volume' == units:
    # Solvent density in g/ml, obtained from averaging short MD run
    data = np.loadtxt(f'{row_idx}/solventdata.txt', skiprows=1, usecols=3,delimiter=',')
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
    numsalt = salt_conc*np.round(volume*Avog).astype(int)
elif 'mass' == units:
    #No need to look at solvent density
    mass = 1e-3*num_solv*solv_mwweight/Avog #mw is in g/mol, convert to kg/mol
    numsalt = salt_conc*np.round(mass*Avog).astype(int)
elif 'number' == units or 'Number' == units:
    salt_molfrac = salt_conc/np.sum(salt_conc)
    numsalt = salt_molfrac*np.round(num_solv).astype(int)

#stoich = solve_single_equation(charges)
salt = cat+an
salt_counts = []
for j in range(len(salt)):
    elements, counts = d2l.extract_elements_and_counts(salt[j])
    if numsalt[j] < 1:
        Nmols.append(1)
        Natoms.append(sum(counts))
        salt_counts.append(sum(counts))
    else:
        Nmols.append(int(numsalt[j]))
        Natoms.append(sum(counts)*int(numsalt[j]))
        salt_counts.append(sum(counts))

for j in range(len(solv)):
    elements, counts = d2l.extract_elements_and_counts(solv[j])
    if int(num_solv*solv_molfrac[j]) < 1:
        Nmols.append(1)
        Natoms.append(sum(counts))
    else:
        Nmols.append(int(num_solv*solv_molfrac[j]))
        Natoms.append(sum(counts)*int(num_solv*solv_molfrac[j]))


#Next we want to cap the total number of atoms
NMax = 5000
count = 0
if sum(Natoms) > NMax:
    fracs = np.array(Natoms)/sum(Natoms)
    for j, frac in enumerate(fracs):
        N = frac*NMax
        count += N
        N = int(np.round((N/(Natoms[j]/Nmols[j]))))
        Nmols[j] = N

Nmols_salt = Nmols[:len(cat+an)]
#Load the CSV file
cations_file = 'cations.csv'
anions_file = 'anions.csv'
cations = load_csv(cations_file)
anions = load_csv(anions_file)

# Collect rows corresponding to the first match for each known entry
charges = []
for cat_sp in cat:# in known_entries:
    # Find the first row where column 'B' has the known entry
    matching_row = cations[cations['formula'] == cat_sp].iloc[0]
    charges.append(matching_row['charge'])
for an_sp in an:# in known_entries:
    # Find the first row where column 'B' has the known entry
    matching_row = anions[anions['formula'] == an_sp].iloc[0]
    charges.append(matching_row['charge'])

#We should ensure that the final number satisfies charge neutrality again!
print(cat+an)
print(Nmols_salt)
if sum(np.array(charges)*np.array(Nmols_salt)) > 0.0 or any(x == 0 for x in Nmols_salt):
    print(f"Charge neutrality is not satisfied for system {row_idx}")

    print(cat+an)
    print("Previous number of cation/anion molecules: ",Nmols[:len(cat+an)])
    Nmols[:len(cat+an)] = get_nmols(charges, Nmols[:len(cat+an)], tol=1)
    print("New number of cation/anion molecules: ",Nmols[:len(cat+an)])

d2l.run_packmol_moltemplate(species,boxsize,Nmols,'system',str(row_idx))
lmm.prep_openmm_sim("system",cat,an,solv,str(row_idx))#-1))