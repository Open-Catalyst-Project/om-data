"""classmixing.py
Author: Muhammad R. Hasyim

Script to a list of random electrolytes based on their classifications. In particular, there are
five classes to choose from:
(1) 40% Salt in protic solvents
(2) 45% Salt in polar aprotic solvents
(3) 10% Salt in ionic liquids
(4) 5% Molten salt 
(5) 5% Aqueous electrolytes
For each class, we create two distinct concentrations (0.05 and 0.5 molal) and temperatures. The salt
may contain N-many cations and M-many anions, with the number chosen randomly  and stoichiometry
satisfying charge neutrality. Solvents can be mixtures, with components chosen randomly as well. 
For each cation, anion, and solvent we can have up to four components. 

The resulting random electrolytes are appended as new entry to the elytes.csv file, which contain the list of all electrolytes we want to simulate. 
"""
import re
import pandas as pd
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
    x = [LpVariable(f"x{i}", 1, 4, cat='Integer') for i in range(len(coefficients))]
    prob += lpSum(coeff * var for coeff, var in zip(coefficients, x)) == 0
    prob.solve()
    solutions = [int(v.varValue) for v in prob.variables()[1:]]
    return solutions

#Function to randomly choose cations, anions, and solvents
def choose_species(species,solvent=False,cations=False):
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

elytes= load_csv('elytes.csv')

Nrandom = 700
for i in range(Nrandom):
    cations = load_csv(cations_file)
    anions = load_csv(anions_file)
    solvents = load_csv(solvents_file)

    #Randomly select which class we want to create
    classes = ['protic','aprotic','IL','MS','aq']
    clas = np.random.choice(classes,p=[0.4,0.4,0.1,0.05,0.05])
    
    #(1) Aqueous electrolytes
    if clas == 'aq':
        aq_idx = list(cations['in_aq'])
        cations = cations.iloc[aq_idx]
        cat, catcharges = choose_species(cations)#,cations=True)
         
        aq_idx = list(anions['in_aq'])
        anions = anions.iloc[aq_idx]
        an, ancharges = choose_species(anions)
        
        charges = catcharges + ancharges
        stoich = solve_single_equation(charges)
        
        neut = ['H2O']
        stoich_solv = [1]
        salt_molfrac = np.array(stoich)/sum(stoich)
        solv_molfrac = np.array(stoich_solv)/sum(stoich_solv)
        minT = 273
        maxT = 373
        soltorsolv = len(cat+an)*['A']+len(neut)*['B']
    #(2) Protic solvents
    if clas == 'protic':
        protic_idx = list(cations['in_protic'])
        cations = cations.iloc[protic_idx]
        cat, catcharges = choose_species(cations)#,cations=True)
        
        protic_idx = list(anions['in_protic'])
        anions = anions.iloc[protic_idx]
        an, ancharges = choose_species(anions)
        
        charges = list(catcharges)+list(ancharges)
        stoich = solve_single_equation(charges)

        protic_idx = list(solvents['protic'])
        solvents = solvents.iloc[protic_idx]
        neut, neutcharges, minT, maxT= choose_species(solvents,solvent=True)
        stoich_solv = np.random.randint(1, 4, size=len(neut))
        
        salt_molfrac = np.array(stoich)/sum(stoich)
        solv_molfrac = np.array(stoich_solv)/sum(stoich_solv)
        minT = np.sum(np.array(minT)*solv_molfrac)
        maxT = np.sum(np.array(maxT)*solv_molfrac)

        soltorsolv = len(cat+an)*['A']+len(neut)*['B']
    #(3) Polar aprotic solvents
    elif clas == 'aprotic':
        aprotic_idx = list(cations['in_aprotic'])
        cations = cations.iloc[aprotic_idx]
        cat, catcharges = choose_species(cations)#,cations=True)
        
        aprotic_idx = list(anions['in_aprotic'])
        anions = anions.iloc[aprotic_idx]
        an, ancharges = choose_species(anions)

        charges = list(catcharges)+list(ancharges)
        stoich = solve_single_equation(charges)

        aprotic_idx = list(solvents['polar_aprotic'])
        solvents = solvents.iloc[aprotic_idx]
        neut, neutcharges, minT, maxT= choose_species(solvents,solvent=True)
        stoich_solv = np.random.randint(1, 4, size=len(neut))
        
        salt_molfrac = np.array(stoich)/sum(stoich)
        solv_molfrac = np.array(stoich_solv)/sum(stoich_solv)
        minT = np.sum(np.array(minT)*solv_molfrac)
        maxT = np.sum(np.array(maxT)*solv_molfrac)
        
        soltorsolv = len(cat+an)*['A']+len(neut)*['B']
    #(3) Ionic liquids
    elif clas == 'IL': 
        IL_idx = list(cations['in_IL'])
        cations_il = cations.iloc[IL_idx]
        cat, catcharges = choose_species(cations_il)#,cations=True)
        
        IL_idx = list(anions['in_IL'])
        anions_il = anions.iloc[IL_idx]
        an, ancharges = choose_species(anions_il)

        charges = list(catcharges)+list(ancharges)
        stoich = solve_single_equation(charges)

        IL_idx = list(cations['IL_comp'])
        cations = cations.iloc[IL_idx]
        neut, solvcharges = choose_species(cations)#,cations=True) 
        
        IL_idx = list(anions['IL_comp'])
        anions = anions.iloc[IL_idx]
        neut1, solvcharges1 = choose_species(anions) 
        neut += neut1
        solvcharges += solvcharges1

        minT = 300
        maxT = 400
        stoich_solv = solve_single_equation(solvcharges)
        
        salt_molfrac = np.array(stoich)/sum(stoich)
        solv_molfrac = np.array(stoich_solv)/sum(stoich_solv)
        soltorsolv = len(cat+an)*['A']+len(neut)*['A']
    #(4) Molten salt
    elif clas == 'MS':
        MS_idx = list(cations['MS_comp'])
        cations_il = cations.iloc[MS_idx]
        cat, catcharges = choose_species(cations_il)#,cations=True)
        
        MS_idx = list(anions['MS_comp'])
        anions_il = anions.iloc[MS_idx]
        an, ancharges = choose_species(anions_il)

        charges = list(catcharges)+list(ancharges)
        stoich = solve_single_equation(charges)
        
        minT = 1000
        maxT = 1300
        stoich_solv = []
        neut = []
        salt_molfrac = np.array(stoich)/sum(stoich)
        solv_molfrac = []

    species = cat+an+neut
    #Start preparing random electrolytes
    concs = [0.05, 0.5]
    for conc in concs:
        for temperature  in [minT, maxT]:
            salt_conc = conc*np.array(stoich)
            #Add this to the new array
            newspecies = dict()
            newspecies['category'] = f'random-{clas}'
            newspecies['comment/name'] = f'{clas}-{i+1}'
            newspecies['DOI'] = ''
            if clas == 'MS':
                newspecies['units'] = 'number'
            else:
                newspecies['units'] = 'mass'
            newspecies['temperature'] = temperature
            for j in range(4):
                if j < len(cat):
                    newspecies[f'cation{j+1}'] = cat[j]
                    if clas == 'MS':
                        newspecies[f'cation{j+1}_conc'] = stoich[j]
                    else:
                        newspecies[f'cation{j+1}_conc'] = salt_conc[j]
                else:
                    newspecies[f'cation{j+1}'] = ''
                    newspecies[f'cation{j+1}_conc'] = ''
                if j < len(an):
                    newspecies[f'anion{j+1}'] = an[j]
                    if clas == 'MS':
                        newspecies[f'anion{j+1}_conc'] = stoich[j+len(cat)]
                    else:
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
            newspecies[f'neutral5'] = ''
            newspecies[f'neutral5'] = ''
            elytes = pd.concat([elytes,pd.DataFrame([newspecies])],ignore_index=True)
elytes.to_csv('elytes.csv', index=False)
