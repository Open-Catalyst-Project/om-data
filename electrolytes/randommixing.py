"""randommixing.py
Author: Muhammad R. Hasyim

Script to a list of purely random electrolytes based on their classifications. THe only constraint here
is the number of components and charge neutrality. Here, we create two distinct concentrations (0.05 and 1.0 molal) and temperatures. The salt may contain N-many cations and M-many anions, with the number chosen randomly  and stoichiometry satisfying charge neutrality. Solvents can be mixtures, with components chosen randomly as well. For each cation, anion, and solvent we can have up to four components. 

The resulting random electrolytes are appended as new entry to the elytes.csv file, which contain the list of all electrolytes we want to simulate. 
"""
import pandas as pd
import sys
import data2lammps as d2l
import lammps2omm as lmm
import os
import csv 
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

# Solver that can give us stoichiometry that satisfies charge neutrality
def solve_single_equation(coefficients):
    prob = LpProblem("Integer_Solution_Problem", LpMinimize)
    x = [LpVariable(f"x{i}", 1, 4, cat='Integer') for i in range(len(coefficients))]
    prob += lpSum(coeff * var for coeff, var in zip(coefficients, x)) == 0
    prob.solve()
    return [int(v.varValue) for v in prob.variables()[1:]]

#Function to randomly choose cations, anions, and solvents
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

Nrandom = 200
for i in range(Nrandom):
    
    #Randomly select cations and cations
    cat, catcharges = choose_species(cations)
    an, ancharges = choose_species(anions)

    #Solve for stoichiometry that preserves charge neutrality
    charges = catcharges + ancharges
    stoich = solve_single_equation(charges)

    #Randomly select solvents
    neut, neutcharges, minT, maxT= choose_species(solvents,solvent=True)
    stoich_solv = np.random.randint(1, 3, size=len(neut))
        
    soltorsolv = len(cat+an)*['A']+len(neut)*['B']
       
    species = cat+an+neut
    salt_molfrac = np.array(stoich)/sum(stoich)
    solv_molfrac = np.array(stoich_solv)/sum(stoich_solv)
    molfrac = salt_molfrac.tolist()+solv_molfrac.tolist()
    
    #Compute the minimum and maximum temperature to simulate
    #based on an ideal mixing rule. 
    minT = np.sum(np.array(minT)*solv_molfrac)
    maxT = np.sum(np.array(maxT)*solv_molfrac)


    #Start preparing random electrolytes
    concs = [0.05, 1.0]
    for conc in concs:
        for temperature  in [minT, maxT]:
            salt_conc = conc*np.array(stoich)
            
            newelectrolyte = dict()
            newelectrolyte['category'] = 'random'
            newelectrolyte['comment/name'] = f'Rand-{i+1}'
            newelectrolyte['DOI'] = ''
            newelectrolyte['units'] = 'mass'
            newelectrolyte['temperature'] = temperature
            for j in range(10):
                if j < len(cat):
                    newelectrolyte[f'cation{j+1}'] = cat[j]
                    newelectrolyte[f'cation{j+1}_conc'] = salt_conc[j]
                else:
                    newelectrolyte[f'cation{j+1}'] = ''
                    newelectrolyte[f'cation{j+1}_conc'] = ''
                if j < len(an):
                    newelectrolyte[f'anion{j+1}'] = an[j]
                    newelectrolyte[f'anion{j+1}_conc'] = salt_conc[j+len(cat)]
                else:
                    newelectrolyte[f'anion{j+1}'] = ''
                    newelectrolyte[f'anion{j+1}_conc'] = ''
                if j < len(neut):
                    newelectrolyte[f'neutral{j+1}'] = neut[j]
                    newelectrolyte[f'neutral{j+1}_ratio'] = stoich_solv[j]
                else:
                    newelectrolyte[f'neutral{j+1}'] = ''
                    newelectrolyte[f'neutral{j+1}_ratio'] = ''
            newelectrolyte[f'neutral5'] = ''
            newelectrolyte[f'neutral5'] = ''
            
            elytes = pd.concat([elytes,pd.DataFrame([newelectrolyte])],ignore_index=True)
elytes.to_csv('elytes.csv', index=False)
