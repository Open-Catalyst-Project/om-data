"""generatesystem.py
Author: Muhammad R. Hasyim

Script to generate initial system configuration and LAMMPS files using 
the molbuilder.py module. This script assumes that you have done the following steps:

    1. Run generatesolvent.py to generate pure solvent configurations. 
    2. Run prepopenmmsim.py for the pure solvent.
    3. Run the MD simulation of the pure solvent using OpenMM (runsolvent.py). 

These steps generate a solventdata.txt file containing a time series of the density of the solvent, 
which we can use to calculate the number of salt molecules to put inside the simulation box. 
"""
import sys
import molbuilder as mb
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
    print("Charges:",charges)
    print("Number of mols", natoms)
    print("Total charge:",sum(np.array(charges)*np.array(natoms)))
    x = []
    for i in range(len(charges)):
        if natoms[i]-tol <= 0:
            lowbound= 1
        else:
            lowbound = natoms[i]-tol
        x.append(LpVariable(f"x{i}", lowbound, natoms[i]+tol, cat='Integer'))
    #Solve linear algebra problem of charge neutrality
    prob += lpSum(coeff * var for coeff, var in zip(charges, x)) == 0
    prob.solve()
    sorted_variables = sorted(prob.variables()[1:], key=lambda v: int(v.name[1:]))
    return [int(v.varValue) for v in sorted_variables]

# Read which system # to simulate from command line argument
row_idx  = int(sys.argv[1]) 

# Load the CSV file containing systems to simulate
with open("elytes.csv", "r") as f:
    systems = list(csv.reader(f))
comments = systems[0]
# Extract indices of columns specifying the cation, anion,and solvent
index_cat, index_cat_conc = mb.get_indices(comments, "cation")
index_an, index_an_conc = mb.get_indices(comments, "anion")
index_solv, index_solv_ratio = mb.get_indices(comments, "solvent")

# Extract salt species and their concentrations
cat = mb.get_species_and_conc(systems, row_idx, index_cat)
cat_conc = mb.get_species_and_conc(systems, row_idx, index_cat_conc)#.astype(float)
an = mb.get_species_and_conc(systems, row_idx, index_an)
an_conc = mb.get_species_and_conc(systems, row_idx, index_an_conc)#.astype(float)

salt_molfrac = np.array(cat_conc+an_conc).astype(float)
salt_molfrac /= np.sum(salt_molfrac)


# Extract solvent species name and their molar ratios
solv = mb.get_species_and_conc(systems, row_idx, index_solv)
solv_ratio = mb.get_species_and_conc(systems, row_idx, index_solv_ratio)
solv_molfrac = np.array(solv_ratio).astype(float)
solv_molfrac /= np.sum(solv_molfrac)

soltorsolv = len(cat+an)*['A']+len(solv)*['B']
   
species = cat+an+solv#$system[2:Ncomp+2]
molfrac = salt_molfrac.tolist()+solv_molfrac.tolist()

# Collect rows corresponding to the first match for each known entry
#Load the CSV file
cations_file = 'cations.csv'
anions_file = 'anions.csv'
cations = load_csv(cations_file)
anions = load_csv(anions_file)

charges = []
for cat_sp in cat:# in known_entries:
    # Find the first row where column 'B' has the known entry
    matching_row = cations[cations['formula'] == cat_sp].iloc[0]
    charges.append(matching_row['charge'])
for an_sp in an:# in known_entries:
    # Find the first row where column 'B' has the known entry
    matching_row = anions[anions['formula'] == an_sp].iloc[0]
    charges.append(matching_row['charge'])

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

minmol = 2 #We want the smallest concentration to be 2 species
numsalt = 0
natoms_max = 5000 
#num_solv = 1e6#485#5000
numsalt = 0

salt_conc = np.array(cat_conc+an_conc).astype(float)
solv_mwweight = sum(mb.calculate_mw(solv)*solv_frac for solv, solv_frac in zip(solv, solv_molfrac))

if 'volume' == units:
    minboxsize = 4 #nm
    # Solvent density in g/ml, obtained from averaging short MD run
    data = np.loadtxt(f'{row_idx}/solventdata.txt', skiprows=1, usecols=3,delimiter=',')
    rho = np.mean(data[-10:])#:w
    print(rho,"g/mL")
    print(solv_mwweight,"g/mol")
    rho *= 1000 #in g/L
    molrho = rho/solv_mwweight #in mol/L

    numsalt = salt_conc/min(salt_conc)*minmol
    totalmol = np.sum(numsalt)
    volume = totalmol/(sum(salt_conc)*Avog) # L
    boxsize = volume**(1/3)/10*1e9
    newminmol = minmol
    conc = 0.6022*sum(salt_conc) #number per nm3
    while minboxsize > boxsize:
        newminmol += 1
        numsalt = salt_molfrac/min(salt_molfrac)*newminmol
        salt_conc = salt_molfrac*conc/Avog*1e24 #number per nm3
        totalmol = np.sum(numsalt) #total number 
        boxsize = (totalmol/conc)**(1/3) #nm
    volume  = boxsize**3*1e-24 #nm3
    num_solv = rho/solv_mwweight*volume*Avog
elif 'mass' == units:
    #No need to look at solvent density
    numsalt = salt_conc/min(salt_conc)*minmol
    mass = numsalt[0]/Avog/salt_conc[0] #kg
    num_solv = 1000*mass/solv_mwweight*Avog  
    
    numsolv = np.round(num_solv*solv_molfrac).astype(int)
    numspec = np.append(numsalt,numsolv)
    spec = cat+an+solv
    natoms = 0
    for j in range(len(spec)):
        elements, counts = mb.extract_elements_and_counts(spec[j])
        natoms += sum(counts)*numspec[j]
    
    scale_factor = natoms_max/natoms
    if scale_factor > 1:
       numsolv *= int(scale_factor) 
       numsalt *= int(scale_factor)
    num_solv = sum(numsolv)
elif 'number' == units or 'Number' == units:
    #We cannot use minmol to initiate this. Everything is salt. But we know
    #We want to limit the number of atoms
    spec_conc = np.array(cat_conc + an_conc + solv_ratio) 
    spec = cat+an+solv
    numsalt = salt_conc/min(salt_conc)*minmol
    numsolv = solv_molfrac*minmol
    numspec = np.append(numsalt,numsolv)
    natoms = 0
    for j in range(len(spec)):
        elements, counts = mb.extract_elements_and_counts(spec[j])
        natoms += sum(counts)*numspec[j]
    scale_factor = natoms_max/natoms
    if scale_factor > 1:
        numsolv *= int(scale_factor) 
        numsalt *= int(scale_factor)
    num_solv = sum(numsolv)
numsolv = np.round(num_solv*solv_molfrac).astype(int)
Nmols = np.concatenate((numsalt,numsolv)).astype(int)
print(cat,an,solv)
print(Nmols)
totalcharge = np.round(sum(np.array(charges)*np.array(Nmols[:len(cat+an)])))
if totalcharge > 0.0 or any(x == 0 for x in Nmols[:len(cat+an)]):
    print(f"Charge neutrality is not satisfied for system {row_idx}")
    print(cat+an)
    print("Previous number of cation/anion molecules: ",Nmols[:len(cat+an)])
    Nmols[:len(cat+an)] = get_nmols(charges, Nmols[:len(cat+an)], tol=1)
    print("New number of cation/anion molecules: ",Nmols[:len(cat+an)])

#Initial boxsize is set so that density is 0.5 g/mL
rho = 0.5 #g/mL
if units == 'number' or units == 'Number':
    salt_mwweight = sum(mb.calculate_mw(salt)*salt_frac for salt, salt_frac in zip(cat+an, salt_conc/min(salt_conc)))
    rho *= 1000 #in g/L
    molrho = rho/salt_mwweight #in mol/L
    Avog = 6.022e23
    volume = sum(numsalt)/molrho*1e27/Avog #A3
    boxsize = volume**(1/3)
else:
    solv_mwweight = sum(mb.calculate_mw(solv)*solv_frac for solv, solv_frac in zip(solv, solv_molfrac))
    rho *= 1000 #in g/L
    molrho = rho/solv_mwweight #in mol/L
    Avog = 6.022e23
    volume = num_solv/molrho*1e27/Avog #A3
    boxsize = volume**(1/3)
mb.run_system_builder(cat,an, solv,Nmols,'system',str(row_idx),boxsize=boxsize,mdengine='openmm')
lmm.prep_openmm_md("system",cat,an,solv,Nmols.tolist(),str(row_idx))#-1))
