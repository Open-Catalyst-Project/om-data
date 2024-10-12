"""randommixing.py
Author: Muhammad R. Hasyim

Script to a list of purely random electrolytes based on their classifications. THe only constraint here
is the number of components and charge neutrality. Here, we create two distinct concentrations (0.05 and 0.5 molal) and temperatures. The salt may contain N-many cations and M-many anions, with the number chosen randomly  and stoichiometry satisfying charge neutrality. Solvents can be mixtures, with components chosen randomly as well. For each cation, anion, and solvent we can have up to four components. 

The resulting random electrolytes are appended as new entry to the elytes.csv file, which contain the list of all electrolytes we want to simulate. 
"""
import pandas as pd
import re
import sys
import molbuilder as mb
import lammps2omm as lmm
import os
import csv 
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

Avog = 6.023*10**23
def write_elyte_entry(clas, name, temperature, cat, an, solv, salt_conc, stoich_solv):
    newelectrolyte = dict()
    newelectrolyte['category'] = 'random'
    newelectrolyte['comment/name'] = name 
    newelectrolyte['DOI'] = ''
    if clas == 'MS':
        newelectrolyte['units'] = 'number'
    else:
        newelectrolyte['units'] = 'volume'
    newelectrolyte['temperature'] = temperature
    for j in range(max_comp):
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
        if j < len(solv):
            newelectrolyte[f'solvent{j+1}'] = solv[j]
            newelectrolyte[f'solvent{j+1}_ratio'] = stoich_solv[j]
        else:
            newelectrolyte[f'solvent{j+1}'] = ''
            newelectrolyte[f'solvent{j+1}_ratio'] = ''
    return newelectrolyte

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

# Function to check if selected species exist in the existing list
def check_for_duplicates(existing_species, selected_species):
    existing_cleaned = [re.split(r'[+-]', formula)[0] for formula in existing_species]
    selected_cleaned = [re.split(r'[+-]', formula)[0] for formula in selected_species]

    # Check if any selected species are already in the existing list
    for selected in selected_cleaned:
        if selected in existing_cleaned:
            return True
    return False

def choose_species(species,max_comp,existing_species,solvent=False,cations=False):
    formulas = lanthanides
    while contains_lanthanide(formulas) or check_for_duplicates(existing_species, formulas):
        # Keep randomly choosing until there are no duplicates
        indices = np.random.choice(len(species), size=np.random.randint(1, max_comp), replace=False)
        formulas = np.array(species["formula"])[indices].tolist()

        # Check for duplicates, loop continues until no duplicates are found
        if not contains_lanthanide(formulas) and not check_for_duplicates(existing_species, formulas):
            break
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

Nrandom = 60
fac = 0.05
for i in range(Nrandom):
    max_comp = 3

    #Randomly select cations and cations
    cat, catcharges = choose_species(cations,max_comp,[])
    an, ancharges = choose_species(anions,max_comp,cat)

    #Solve for stoichiometry that preserves charge neutrality
    charges = catcharges + ancharges
    stoich = solve_single_equation(charges)

    #Randomly select solvents
    solv, solvcharges, minT, maxT= choose_species(solvents,max_comp,cat+ansolvent=True)
    stoich_solv = np.random.randint(1, 3, size=len(solv))
        
    #formulas = remove_duplicates([cat,an,solv])
    #cat = formulas[0]
    #an = formulas[1]
    #solv = formulas[2]
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
    distances = [2.25, 1.75] #nm 
    minmol = 1
    for dis in distances:
        for temperature  in [minT, maxT]:
            conc = 0.62035049089/dis**3 # number per nm3, concentration
            mols = salt_molfrac/min(salt_molfrac)*minmol 
            salt_conc = salt_molfrac*conc/Avog*1e24 #number per nm3
            clas = 'Rand'            
            name = f'{clas}-{i+1}'
            if temperature == minT:
                name += '-minT'
            else:
                name += '-maxT'
            if dis == 2.25:
                name += '-lowconc'
            else:
                name += '-highconc'
            newelectrolyte = write_elyte_entry(clas, name, temperature, cat, an, solv, salt_conc, stoich_solv)
            elytes = pd.concat([elytes,pd.DataFrame([newelectrolyte])],ignore_index=True)
elytes.to_csv('elytes.csv', index=False)
