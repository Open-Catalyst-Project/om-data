import sys
from data2lammps import *
import os
import csv 
import numpy as np


# Load the CSV file containing systems to simulate
f = open("systems.csv", "r")
systems = list(csv.reader(f))

# Choose the system to simulate
i = sys.argv[1]
system = systems[int(i)]

# Get number of components not including the salt(Ncomp), number of ionic species in the salt (Nions), all molecular species in the system (species), stoichiometry for atomic species (stoichio))
Ncomp = int(system[0])
Nions = int(system[1])
Ncomp -= Nions

species = system[2:Ncomp+2]
species = species[Nions:]

stoichio = np.array(system[Ncomp+2:2*Ncomp+2]).astype(int)
stoichio = stoichio[Nions:]

#Initial boxsize is always 5 nm. 
boxsize = 50 

num_solv = 500
Nmols = []
for j in range(Ncomp):
    Nmols.append(num_solv*stoichio[j]/np.sum(stoichio))

#Run Packmol, followed up by moltemplate 
run_packmol_moltemplate(species,boxsize,Nmols,'solvent',i)
