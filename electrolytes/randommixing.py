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

def remove_dup_species(formulas, ids):
    # Use a dictionary to keep track of unique formulas and corresponding ids
    unique_formulas_with_ids = {}

    for formula, id in zip(formulas, ids):
        cleaned_formula = re.split(r'[+-]', formula)[0]
        if cleaned_formula not in unique_formulas_with_ids:
            unique_formulas_with_ids[cleaned_formula] = (formula, id)

    # Extract the original formulas and corresponding ids
    cleaned_formulas = [item[0] for item in unique_formulas_with_ids.values()]
    corresponding_ids = [item[1] for item in unique_formulas_with_ids.values()]

    return cleaned_formulas, corresponding_ids

def remove_duplicates(lists_of_formulas):
    # Initialize lists to collect all formulas and their respective ids
    all_formulas = []
    all_ids = []

    # Remove duplicates within each list and collect all formulas with their original list ids
    for i, formulas in enumerate(lists_of_formulas):
        cleaned_formulas, _ = remove_dup_species(formulas, list(range(len(formulas))))
        all_formulas.extend(cleaned_formulas)
        all_ids.extend([i] * len(cleaned_formulas))  # Use the list index as the id

    # Remove duplicates across lists
    unique_formulas_with_ids = {}

    for formula, id in zip(all_formulas, all_ids):
        cleaned_formula = re.split(r'[+-]', formula)[0]
        if cleaned_formula not in unique_formulas_with_ids:
            unique_formulas_with_ids[cleaned_formula] = (formula, id)

    # Extract the original formulas and corresponding ids
    unique_formulas = [item[0] for item in unique_formulas_with_ids.values()]
    corresponding_ids = [item[1] for item in unique_formulas_with_ids.values()]

    # Create a dictionary to store the unique items for each original list
    cleaned_lists = {i: [] for i in range(len(lists_of_formulas))}

    # Distribute the unique items back into their respective original lists
    for formula, id in zip(unique_formulas, corresponding_ids):
        cleaned_lists[id].append(formula)

    # Convert the dictionary back to a list of lists
    result_lists = [cleaned_lists[i] for i in range(len(lists_of_formulas))]

    return result_lists


# Solver that can give us stoichiometry that satisfies charge neutrality
def solve_single_equation(coefficients):
    prob = LpProblem("Integer_Solution_Problem", LpMinimize)
    x = [LpVariable(f"x{i}", 1, 5, cat='Integer') for i in range(len(coefficients))]
    prob += lpSum(coeff * var for coeff, var in zip(coefficients, x)) == 0
    prob.solve()
    sorted_variables = sorted(prob.variables()[1:], key=lambda v: int(v.name[1:]))
    print(sorted_variables)
    return [int(v.varValue) for v in sorted_variables]

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

def choose_species(species,max_comp,solvent=False,cations=False):
    
    formulas = lanthanides
    while contains_lanthanide(formulas):
        indices = np.random.choice(len(species), size=max_comp, replace=False)
        #indices = np.random.choice(len(species), size=np.random.randint(1,max_comp), replace=False)
        #We want to make sure that we have no "identical" species, disregarding their charges
        formulas = np.array(species["formula"])[indices].tolist() 
        #formulas, indices = remove_dup_species(formulas,indices)
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

Nrandom = 200
fac = 0.05
for i in range(Nrandom):
    max_comp = 8

    #Randomly select cations and cations
    cat, catcharges = choose_species(cations,max_comp)
    an, ancharges = choose_species(anions,max_comp)

    #Solve for stoichiometry that preserves charge neutrality
    charges = catcharges + ancharges
    stoich = solve_single_equation(charges)

    #Randomly select solvents
    solv, solvcharges, minT, maxT= choose_species(solvents,max_comp,solvent=True)
    stoich_solv = np.random.randint(1, 3, size=len(solv))
        
    formulas = remove_duplicates([cat,an,solv])
    cat = formulas[0]
    an = formulas[1]
    solv = formulas[2]
    soltorsolv = len(cat+an)*['A']+len(solv)*['B']
       
    species = cat+an+solv
    salt_molfrac = np.array(stoich)/sum(stoich)
    solv_molfrac = np.array(stoich_solv)/sum(stoich_solv)
    molfrac = salt_molfrac.tolist()+solv_molfrac.tolist()
    
    #Compute the minimum and maximum temperature to simulate
    #based on an ideal mixing rule. 
    minT = (1+fac)*np.sum(np.array(minT)*solv_molfrac)
    maxT =(1-fac)*np.sum(np.array(maxT)*solv_molfrac)


    #Start preparing random electrolytes
    concs = [1.0, 10.0]
    for conc in concs:
        for temperature  in [minT, maxT]:
            salt_conc = conc*np.array(stoich)
            
            newelectrolyte = dict()
            newelectrolyte['category'] = 'random'
            newelectrolyte['comment/name'] = f'Rand-{i+1}'
            newelectrolyte['DOI'] = ''
            newelectrolyte['units'] = 'mass'
            newelectrolyte['temperature'] = temperature
            for j in range(max_comp):
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
                if j < len(solv):
                    newelectrolyte[f'solvent{j+1}'] = solv[j]
                    newelectrolyte[f'solvent{j+1}_ratio'] = stoich_solv[j]
                else:
                    newelectrolyte[f'solvent{j+1}'] = ''
                    newelectrolyte[f'solvent{j+1}_ratio'] = ''
            
            elytes = pd.concat([elytes,pd.DataFrame([newelectrolyte])],ignore_index=True)
elytes.to_csv('testelytes.csv', index=False)
