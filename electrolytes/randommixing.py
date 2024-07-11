import pandas as pd
import sys
import data2lammps as d2l
import lammps2omm as lmm
import os
import csv 
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

def solve_single_equation(coefficients):
    prob = LpProblem("Integer_Solution_Problem", LpMinimize)
    x = [LpVariable(f"x{i}", 1, 4, cat='Integer') for i in range(len(coefficients))]
    prob += lpSum(coeff * var for coeff, var in zip(coefficients, x)) == 0
    prob.solve()
    return [int(v.varValue) for v in prob.variables()[1:]]

#Function to Randomly choose cations, anions, and solvents
def choose_species(species,solvent=False):
    indices = np.random.choice(len(species), size=np.random.randint(1,5), replace=False)
    if solvent:
        return list(species["formula"][indices]), list(species["charge"][indices]), list(species["min_temperature"][indices]), list(species["max_temperature"][indices])
    else:
        return list(species["formula"][indices]), list(species["charge"][indices])
def load_csv(filename):
    return pd.read_csv(filename)

#Load the CSV file
cations_file = 'cations.csv'
anions_file = 'anions.csv'
solvents_file = 'solvent.csv'

cations = load_csv(cations_file)
anions = load_csv(anions_file)
solvents = load_csv(solvents_file)
elytes= load_csv('elytes.csv')

#Select cations and anions randomly, then solve stoichiometry
#that preserves charge neutrality
Nrandom = 10#00
for i in range(Nrandom):
    cat, catcharges = choose_species(cations)
    an, ancharges = choose_species(anions)

    charges = catcharges + ancharges
    stoich = solve_single_equation(charges)

    #Select neutral species randomly
    neut, neutcharges, minT, maxT= choose_species(solvents,solvent=True)
    stoich_solv = np.random.randint(1, 3, size=len(neut))
        
    soltorsolv = len(cat+an)*['A']+len(neut)*['B']
       
    species = cat+an+neut
    salt_molfrac = np.array(stoich)/sum(stoich)
    solv_molfrac = np.array(stoich_solv)/sum(stoich_solv)
    molfrac = salt_molfrac.tolist()+solv_molfrac.tolist()
    
    minT = np.sum(np.array(minT)*solv_molfrac)
    maxT = np.sum(np.array(maxT)*solv_molfrac)


    # Calculate how much 
    boxsize = 50 #In Angstrom
    Avog = 6.023*10**23
    concs = [0.05, 1.0]
    for conc in concs:
        for temperature  in [minT, maxT]:
            salt_conc = conc*np.array(stoich)
            
            #Add this to the new array
            newspecies = dict()
            newspecies['category'] = 'random'
            newspecies['comment/name'] = f'Rand-{i+1}'
            newspecies['DOI'] = ''
            newspecies['units'] = 'mass'
            newspecies['temperature'] = temperature
            for j in range(4):
                if j < len(cat):
                    newspecies[f'cation{j+1}'] = cat[j]
                    newspecies[f'cation{j+1}_conc'] = salt_conc[j]
                else:
                    newspecies[f'cation{j+1}'] = ''
                    newspecies[f'cation{j+1}_conc'] = ''
                if j < len(an):
                    newspecies[f'anion{j+1}'] = an[j]
                    newspecies[f'anion{j+1}_conc'] = salt_conc[j+len(cat)]
                else:
                    newspecies[f'anion{j+1}'] = ''
                    newspecies[f'anion{j+1}_conc'] = ''
                if j < len(neut):
                    newspecies[f'neutral{j+1}'] = neut[j]
                    newspecies[f'neutral{j+1}_ratio'] = stoich_solv[j]
                else:
                    newspecies[f'neutral{j+1}'] = ''
                    newspecies[f'neutral{j+1}_ratio'] = ''
            newspecies[f'neutral5'] = ''#neut[j]
            newspecies[f'neutral5'] = ''#stoich_solv[j]

            # Step 3: Append the new row to the DataFrame
            elytes = pd.concat([elytes,pd.DataFrame([newspecies])],ignore_index=True)
#elytes.append(newspecies, ignore_index=True)
# Step 4: Save the updated DataFrame back to the CSV file
elytes.to_csv('finalelytes.csv', index=False)

"""
Nmols = []
Natoms = []

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
NMax = 3500
count = 0
if sum(Natoms) > NMax:
    fracs = np.array(Natoms)/sum(Natoms)
    print(fracs)
    for j, frac in enumerate(fracs):
        N = frac*NMax
        count += N
        N = int(np.round((N/(Natoms[j]/Nmols[j]))))
        Nmols[j] = N
#d2l.run_packmol_moltemplate(species,boxsize,Nmols,'system',str(row_idx-1))
#lmm.prep_openmm_sim("system",cat,an,neut,str(row_idx-1))
"""
