"""randommixing.py
Author: Muhammad R. Hasyim

Script to a list of purely random electrolytes based on their classifications. THe only constraint here
is the number of components and charge neutrality. Here, we create two distinct concentrations (0.05 and 0.5 molal) and temperatures. The salt may contain N-many cations and M-many anions, with the number chosen randomly  and stoichiometry satisfying charge neutrality. Solvents can be mixtures, with components chosen randomly as well. For each cation, anion, and solvent we can have up to four components. 

The resulting random electrolytes are appended as new entry to the elytes.csv file, which contain the list of all electrolytes we want to simulate. 
"""
import pandas as pd
import re
import sys
import data2lammps as d2l
import lammps2omm as lmm
import os
import csv 
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

# Remove species that are duplicate, regardless of charge
def remove_dup_species(formulas, ids):
    # Use a dictionary to keep track of unique formulas and corresponding ids
    unique_formulas_with_ids = {}
    
    for formula, id in zip(formulas, ids):
        cleaned_formula = re.split(r'[+-]', formula)[0]
        if cleaned_formula not in unique_formulas_with_ids:
            unique_formulas_with_ids[cleaned_formula] = id

    # Extract the cleaned formulas and corresponding ids
    cleaned_formulas = list(unique_formulas_with_ids.keys())
    corresponding_ids = list(unique_formulas_with_ids.values())

    return cleaned_formulas, corresponding_ids

# Solver that can give us stoichiometry that satisfies charge neutrality
def solve_single_equation(coefficients):
    prob = LpProblem("Integer_Solution_Problem", LpMinimize)
    x = [LpVariable(f"x{i}", 1, 5, cat='Integer') for i in range(len(coefficients))]
    prob += lpSum(coeff * var for coeff, var in zip(coefficients, x)) == 0
    prob.solve()
    return [int(v.varValue) for v in prob.variables()[1:]]

lanthanides = [
    "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu"
]
#Function to randomly choose cations, anions, and solvents
def contains_lanthanide(strings):
    for string in strings:
        for lanthanide in lanthanides:
            if lanthanide in string:
                return True
    return False

def choose_species(species,solvent=False,cations=False):
    
    formulas = lanthanides
    while contains_lanthanide(formulas):
        indices = np.random.choice(len(species), size=np.random.randint(1,5), replace=False)
        #We want to make sure that we have no "identical" species, disregarding their charges
        formulas = np.array(species["formula"])[indices].tolist() 
        formulas, indices = remove_dup_species(formulas,indices)
    if solvent:
        return np.array(species["formula"])[indices].tolist(), np.array(species["charge"])[indices].tolist(), np.array(species["min_temperature"])[indices].tolist(), np.array(species["max_temperature"])[indices].tolist()
    else:
        return np.array(species["formula"])[indices].tolist(), np.array(species["charge"])[indices].tolist()

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

Nrandom = 100
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
    concs = [0.025, 0.25]
    for conc in concs:
        for temperature  in [minT, maxT]:
            salt_conc = conc*np.array(stoich)
            
            newelectrolyte = dict()
            newelectrolyte['category'] = 'random'
            newelectrolyte['comment/name'] = f'Rand-{i+1}'
            newelectrolyte['DOI'] = ''
            newelectrolyte['units'] = 'mass'
            newelectrolyte['temperature'] = temperature
            for j in range(5):
                if j < len(cat):
                    newelectrolyte[f'cation{j+1}'] = cat[j]
                    newelectrolyte[f'cation{j+1}_conc'] = salt_conc[j]/len(cat+an)
                else:
                    newelectrolyte[f'cation{j+1}'] = ''
                    newelectrolyte[f'cation{j+1}_conc'] = ''
                if j < len(an):
                    newelectrolyte[f'anion{j+1}'] = an[j]
                    newelectrolyte[f'anion{j+1}_conc'] = salt_conc[j+len(cat)]/len(cat+an)
                else:
                    newelectrolyte[f'anion{j+1}'] = ''
                    newelectrolyte[f'anion{j+1}_conc'] = ''
                if j < len(neut):
                    newelectrolyte[f'neutral{j+1}'] = neut[j]
                    newelectrolyte[f'neutral{j+1}_ratio'] = stoich_solv[j]
                else:
                    newelectrolyte[f'neutral{j+1}'] = ''
                    newelectrolyte[f'neutral{j+1}_ratio'] = ''
            
            elytes = pd.concat([elytes,pd.DataFrame([newelectrolyte])],ignore_index=True)
elytes.to_csv('elytes.csv', index=False)
