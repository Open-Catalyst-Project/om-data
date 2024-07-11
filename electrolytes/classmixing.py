import pandas as pd
import sys
import data2lammps as d2l
import lammps2omm as lmm
import os
import csv 
import numpy as np

from pulp import LpProblem, LpVariable, lpSum, LpMinimize

# Read which system # to simulate from command line argument
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
        return np.array(species["formula"])[indices].tolist(), np.array(species["charge"])[indices].tolist(), np.array(species["min_temperature"])[indices].tolist(), np.array(species["max_temperature"])[indices].tolist()
    else:
        return np.array(species["formula"])[indices].tolist(), np.array(species["charge"])[indices].tolist()

def load_csv(filename):
    return pd.read_csv(filename)

cations_file = 'cations.csv'
anions_file = 'anions.csv'
solvents_file = 'solvent.csv'

elytes= load_csv('elytes.csv')
Nrandom = 100#00
for i in range(Nrandom):
    cations = load_csv(cations_file)
    anions = load_csv(anions_file)
    solvents = load_csv(solvents_file)

    #Let's write the completely random electrolyte script
    classes = ['protic','aprotic','IL','MS','aq']
    clas = np.random.choice(classes,p=[0.4,0.4,0.1,0.05,0.05])
    if clas == 'aq':
        aq_idx = list(cations['in_aq'])
        cations = cations.iloc[aq_idx]
        cat, catcharges = choose_species(cations)
         
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
    if clas == 'protic':
        protic_idx = list(cations['in_protic'])
        cations = cations.iloc[protic_idx]
        cat, catcharges = choose_species(cations)
        
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
    elif clas == 'aprotic':
        aprotic_idx = list(cations['in_aprotic'])
        cations = cations.iloc[aprotic_idx]
        cat, catcharges = choose_species(cations)
        
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
    elif clas == 'IL': 
        IL_idx = list(cations['in_IL'])
        cations_il = cations.iloc[IL_idx]
        cat, catcharges = choose_species(cations_il)
        
        IL_idx = list(anions['in_IL'])
        anions_il = anions.iloc[IL_idx]
        an, ancharges = choose_species(anions_il)

        charges = list(catcharges)+list(ancharges)
        stoich = solve_single_equation(charges)

        IL_idx = list(cations['IL_comp'])
        cations = cations.iloc[IL_idx]
        neut, solvcharges = choose_species(cations) 
        
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
    elif clas == 'MS':
        MS_idx = list(cations['MS_comp'])
        cations_il = cations.iloc[MS_idx]
        cat, catcharges = choose_species(cations_il)
        
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
        solv_molfrac = []#np.array(stoich_solv)/sum(stoich_solv)

    species = cat+an+neut

    # Calculate how much 
    boxsize = 50 #In Angstrom
    Avog = 6.023*10**23
    concs = [0.05, 1.0]
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
                        newspecies[f'cation{j+1}_conc'] = salt_conc[j]
                    else:
                        newspecies[f'cation{j+1}_conc'] = stoich[j]
                else:
                    newspecies[f'cation{j+1}'] = ''
                    newspecies[f'cation{j+1}_conc'] = ''
                if j < len(an):
                    newspecies[f'anion{j+1}'] = an[j]
                    if clas == 'MS':
                        newspecies[f'anion{j+1}_conc'] = salt_conc[j]
                    else:
                        newspecies[f'anion{j+1}_conc'] = stoich[j]
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
