"""data2lampmps.py
Author: Muhammad R. Hasyim

Module containing routines to pack molecules at different concentrations 
with Packmol and generate LAMMPS DATA and force field files with Moltemplate

This module is sufficient to run a LAMMPS simulation. A second script (lammps2omm.py)
is used to convert from LAMMPS to OpenMM
"""

from pathlib import Path
import contextlib
import subprocess
import shutil
from collections import Counter
from rdkit.Chem import GetPeriodicTable
import re 
import numpy as np
import sys
import os
import csv 
import string
import json

PT = GetPeriodicTable()


def get_indices(labels, keyword):
    """ Grab indices of labeled columns given a specific keyword. 
        Args: 
            labels (list): a list of strings, each of which is a label for a specific column
            keyword (string): specific keyword we want to match with labels.

        This helper function is specific to reading the elytes.csv file.
    """
    indices = [i for i, string in enumerate(labels) if keyword in string]
    return indices[::2], indices[1::2]
    
def get_species_and_conc(systems, i, indices):
    """ Grab list of species and their corresponding concentrations 
        Args: 
            systems (list): a list of lists, containing the full spreadhseet information from the CSV file. 
            i (int): index of the row of the spreadsheet we want to access. Numbering starts from one. 
            indices (list): index of columns that we want to access. 

        This helper function is specific to reading the elytes.xls file.
    """
    species = np.array(systems[i])[indices]
    return [name for name in species if name]

def run_packmol_moltemplate(species,boxsize,Nmols,filename,directory):
    """ Run Packmol and Moltemplate to generate system configuration (in LAMMPS data format) 
        as well as files to run a LAMMPS simulation.

        Args:
            species (list): list of molecular species name (string) that we want to simulate
            boxsize (float): size of the simulation box, using Angstrom units
            Nmols (list): list of number of molecules we want to pack inside the simulation box
            filename (string): a prefix name for all files we want to generate, including LAMMPS DATA files and PDB files.
            directory (string): directory to generate all the output files
    """
    Path(directory).mkdir(parents=True,exist_ok=True)
    
    # Copy-paste the LT file, which has the OPLS settings
    general_ff = "oplscm1a.lt"
    shutil.copy(f'{general_ff}', f'{directory}')

    # Prepare the Packmol script
    packmolstring = '\n'.join([f"tolerance 2.0",
                    f"filetype pdb",
                    f"output {filename}.pdb \n"])
    
    # Prepare the LT file for the system we want to simulate 
    systemlt = f'import "{general_ff}"\n'
    
    # Go through every species and copy their PDB and LT files to the target directory
    # And add the species to the Packmol script and the system LT file. 
    for j in range(len(Nmols)):
        Nmol = int(Nmols[j])
        #Copy PDB files from the ff directory
        #shutil.copy(f"./ff/{species[j]}.pdb", f'./{directory}')
        #shutil.copy(f"./ff/{species[j]}.lt", f'./{directory}')
        spec_name = f'{species[j]}'
        for suffix in ('.pdb', '.lt'):
            shutil.copy(os.path.join('ff', spec_name + suffix), os.path.join(directory, spec_name + suffix))
        
        packmolstring += '\n'.join([f"structure {species[j]}.pdb",
                        f"  number {Nmol}", "  resnumbers 0",
                        f"  inside box {-boxsize/2.0} {-boxsize/2.0} {-boxsize/2.0} {boxsize/2.0} {boxsize/2.0} {boxsize/2.0}",
                        "end structure \n"])

        systemlt += '\n'.join([f'import "{species[j]}.lt"',
                f'mol{j} = new {species[j]}[{Nmol}]\n'])
    
    
    # Now, work inside the directory 
    with contextlib.chdir(directory):
        
        # Write the Packmol script and run it 
        with open(f"{filename}.inp", "w") as f:
            f.write(packmolstring)
        subprocess.run(f'packmol < {filename}.inp',shell=True)

        # Write the system LT file
        systemlt += '\n'.join(['write_once("Data Boundary") {',
            f'  {-boxsize/2} {boxsize/2} xlo xhi',
            f'  {-boxsize/2} {boxsize/2} ylo yhi',
            f'  {-boxsize/2} {boxsize/2} zlo zhi', '}'])

        with open(f"{filename}.lt", "w") as f:
            f.write(systemlt)
        
        # Given the system LT and PDB file, which is generated from Packmol, run Moltemplate
        subprocess.run(['moltemplate.sh', '-pdb', f'{filename}.pdb', f'{filename}.lt'])
    
        # Cleanup the files generated from moltemplate
        subprocess.run(['cleanup_moltemplate.sh', '-base', f'{filename}'])
    
        # Next check the settings file and see if LAMMPS hybrid style is even necessary
        # Begin by reading the entire contents of the *in.settings file
        with open(f'{filename}.in.settings', 'r') as file:
            lines = file.readlines()
        string = ""
        for i, line in enumerate(lines):
            if "dihedral_coeff" in line and ('opls' in line or 'fourier' in line):
                string += line.split()[2]
    
        if 'opls' in string and 'fourier' in string:
            print("No modification necessary. Hybrid style is present")
        else:
            print("Modification necessary. Only either opls/fourier style present")
        
            #First we modify the lines containing dihedral_coeff and delete the keyword for opls and fourier
            for i, line in enumerate(lines):
                if "dihedral_coeff" in line and ('opls' in line or 'fourier' in line):
                    modified_line = line.replace('opls', '')
                    modified_line = modified_line.replace('fourier', '')
                    lines[i] = modified_line
            with open(f'{filename}.in.settings', 'w') as file:
                file.writelines(lines)
        
            # Next, modify the *in.init file. We need to remove the hybrid keyword from that file.
            with open(f'{filename}.in.init', 'r') as file:
                lines = file.readlines()
            for i, line in enumerate(lines):
                if "dihedral_style" in line and ('opls' in line or 'fourier' in line):
                    modified_line = line.replace('hybrid', '')
                    if 'opls' in string:
                        modified_line = modified_line.replace('fourier', '')
                    if 'fourier' in string:
                        modified_line = modified_line.replace('opls', '')
                    lines[i] = modified_line
            with open(f'{filename}.in.init', 'w') as file:
                file.writelines(lines)

def extract_elements_and_counts(formula):
    """ Return the elements and stoichiometry of a chemical species, given its formula
        
        Args:
            formula (string): a chemical formula, which may include their charge. 
            Example: ethanol C2H6O or hexafluorophosphate F6P-
    """
    # Initialize a dictionary to store stoich for each element
    stoich_dict = Counter()
    # Iterate through the elements to extract stoich
    for symbol, count in re.findall('([A-Z][a-z]?)([0-9]*)', formula):
        # If count is not provided, assume it's 1
        if count == '':
            count = 1
        else:
            count = int(count)
        stoich_dict[symbol] += count
    # Convert the dictionary to lists of elements and stoich
    elements, stoich = zip(*stoich_dict.items())
    return elements, stoich

def calculate_mw(formula):
    """ Calculate molecular weight given the chemical formula
        
        Args:
            formula (string): a chemical formula, which may include their charge. 
            Example: ethanol C2H6O or hexafluorophosphate F6P-
    """
    total_weight = 0.0
    elements, counts = extract_elements_and_counts(formula)
    for element, count in zip(elements, counts):
        total_weight += PT.GetAtomicWeight(element) * count
    return total_weight
