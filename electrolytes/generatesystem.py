import sys
from data2lammps import *
import os
import csv 
import numpy as np

#Load systems CSV file and choose a system
f = open("systems.csv", "r")
i = sys.argv[1]
systems = list(csv.reader(f))#ile)
system = systems[int(i)]

# Get number of components not including the salt(Ncomp), number of ionic species in the salt (Nions), all molecular species in the system (species), stoichiometry for atomic species (stoichio))

Ncomp = int(system[0])  # Total number of species
Nions = int(system[1])  # Total number of ions in the salt
conc = float(system[-2])# Salt concentration

species = system[2:Ncomp+2]
stoichio = np.array(system[Ncomp+2:2*Ncomp+2]).astype(int)

#Initial boxsize is always 5 nm
boxsize = 50 #In Angstrom

#Calculate how many salt species to add in the system
#Given the concentration and density of the solvent
Avog = 6.023*10**23
Nmols = []
num_solv = 500

data = list(csv.reader(open(f'{i}/solventdata.txt', 'r')))
rho = np.array([float(row[3]) for row in data[1:]])
rho = np.mean(rho[int(len(rho)/2):]) # solvent density in g/ml, obtained from averaging short MD run
rho *= 1000 #in g/L

solvspecies = species[Nions:]
solvstoichio = stoichio[Nions:]
solv_mwweight = 0
for j, solv in enumerate(solvspecies):
    solv_mwweight += calculate_mw(solv)*solvstoichio[j]/np.sum(solvstoichio)
molrho = rho/solv_mwweight #in mol/L
volume = num_solv/(Avog*molrho) #in L
numsalt = int(np.round(conc*volume*Avog))

for j in range(Ncomp):
    if j < Nions:
        Nmols.append(numsalt*stoichio[j])
    else:
        Nmols.append(int(num_solv*stoichio[j]/np.sum(stoichio[Nions:])))
print(species,boxsize,Nmols)
run_packmol_moltemplate(species,boxsize,Nmols,'system',i)
