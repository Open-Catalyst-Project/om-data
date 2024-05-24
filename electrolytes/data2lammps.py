"""data2lampmps.py
Author: Muhammad R. Hasyim

Module containing routines to pack molecules at different concentrations 
with Packmol and generate LAMMPS DATA and force field files with Moltemplate

This module is sufficient to run a LAMMPS simulation. A second script (lammps2omm.py)
is used to convert from LAMMPS to OpenMM
"""

import re 
import numpy as np
import sys
import os
import csv 

# Dictionary of atomic masses
atomic_masses = {
    'H': 1.008,
    'He': 4.0026,
    'Li': 6.94,
    'Be': 9.0122,
    'B': 10.81,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'F': 18.998,
    'Ne': 20.180,
    'Na': 22.990,
    'Mg': 24.305,
    'Al': 26.982,
    'Si': 28.085,
    'P': 30.974,
    'S': 32.06,
    'Cl': 35.45,
    'K': 39.098,
    'Ar': 39.948,
    'Ca': 40.078,
    'Sc': 44.956,
    'Ti': 47.867,
    'V': 50.942,
    'Cr': 51.996,
    'Mn': 54.938,
    'Fe': 55.845,
    'Co': 58.933,
    'Ni': 58.693,
    'Cu': 63.546,
    'Zn': 65.38,
    'Ga': 69.723,
    'Ge': 72.63,
    'As': 74.922,
    'Se': 78.971,
    'Br': 79.904,
    'Kr': 83.798,
    'Rb': 85.468,
    'Sr': 87.62,
    'Y': 88.906,
    'Zr': 91.224,
    'Nb': 92.906,
    'Mo': 95.95,
    'Tc': 98.0,
    'Ru': 101.07,
    'Rh': 102.91,
    'Pd': 106.42,
    'Ag': 107.87,
    'Cd': 112.41,
    'In': 114.82,
    'Sn': 118.71,
    'Sb': 121.76,
    'Te': 127.6,
    'I': 126.9,
    'Xe': 131.29,
    'Cs': 132.91,
    'Ba': 137.33,
    'La': 138.91,
    'Ce': 140.12,
    'Pr': 140.91,
    'Nd': 144.24,
    'Pm': 145.0,
    'Sm': 150.36,
    'Eu': 151.96,
    'Gd': 157.25,
    'Tb': 158.93,
    'Dy': 162.5,
    'Ho': 164.93,
    'Er': 167.26,
    'Tm': 168.93,
    'Yb': 173.05,
    'Lu': 174.97,
    'Hf': 178.49,
    'Ta': 180.95,
    'W': 183.84,
    'Re': 186.21,
    'Os': 190.23,
    'Ir': 192.22,
    'Pt': 195.08,
    'Au': 196.97,
    'Hg': 200.59,
    'Tl': 204.38,
    'Pb': 207.2,
    'Bi': 208.98,
    'Po': 209.0,
    'At': 210.0,
    'Rn': 222.0,
    'Fr': 223.0,
    'Ra': 226.0,
    'Ac': 227.0,
    'Th': 232.04,
    'Pa': 231.04,
    'U': 238.03,
    'Np': 237.0,
    'Pu': 244.0,
    'Am': 243.0,
    'Cm': 247.0,
    'Bk': 247.0,
    'Cf': 251.0,
    'Es': 252.0,
    'Fm': 257.0,
    'Md': 258.0,
    'No': 259.0,
    'Lr': 266.0,
    'Rf': 267.0,
    'Db': 270.0,
    'Sg': 271.0,
    'Bh': 270.0,
    'Hs': 277.0,
    'Mt': 276.0,
}


def get_indices(comments, keyword):
    """ Grab indices of labeled columns given a specific keyword. 
        Args: 
            comments (list): a list of strings, each of which is a label for a specific column
            keyword (string): specific keyword we want to match with comments.

        This helper function is specific to reading the elytes.csv file.
    """
    indices = [i for i, string in enumerate(comments) if keyword in string]
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
    try:
        os.mkdir(directory)
    except Exception as e:
        print(e)
    # Copy-paste the LT file, which has the OPLS settings
    general_ff = "oplscm1a.lt"
    bashCommand = f'cp {general_ff} ./{directory}'
    os.system(bashCommand)
   

    # Prepare the Packmol script
    packmolstring =f"""
    tolerance 2.0
    filetype pdb
    output {filename}.pdb
    """
    
    # Prepare the LT file for the system we want to simulate 
    systemlt = f"""import "{general_ff}"
    """
    
    # Go through every species and copy their PDB and LT files to the target directory
    # And add the species to the Packmol script and the system LT file. 
    for j in range(len(Nmols)):
        Nmol = int(Nmols[j])
        #Copy PDB files from the ff directory
        bashCommand = f'cp "./ff/{species[j]}.pdb" ./{directory}'
        os.system(bashCommand)

        bashCommand = f'cp "./ff/{species[j]}.lt" ./{directory}'
        os.system(bashCommand)

        packmolstring += f"""structure {species[j]}.pdb
  number {Nmol}
  resnumbers 0
  inside box {-boxsize/2.0} {-boxsize/2.0} {-boxsize/2.0} {boxsize/2.0} {boxsize/2.0} {boxsize/2.0}
end structure
        """
        systemlt += f"""import "{species[j]}.lt"
mol{j} = new {species[j]}[{Nmol}]
        """
    # Write the Packmol script and run it 
    f = open(directory+f"/{filename}.inp","w")
    f.write(packmolstring)
    f.close()
    bashCommand = f"cd {directory}; packmol < {filename}.inp; cd ..;"
    os.system(bashCommand)

    # Write the system LT file
    systemlt += f"""write_once("Data Boundary") {{
  {-boxsize/2} {boxsize/2} xlo xhi
  {-boxsize/2} {boxsize/2} ylo yhi
  {-boxsize/2} {boxsize/2} zlo zhi
}}
    """
    f = open(directory+f"/{filename}.lt","w")
    f.write(systemlt)
    f.close()

    # Given the system LT and PDB file, which is generated from Packmol, run Moltemplate
    bashCommand = f"cd {directory}; moltemplate.sh -pdb {filename}.pdb {filename}.lt; cd ..;"
    os.system(bashCommand)
    
    # Cleanup the files generated from moltemplate
    bashCommand = f"cd {directory}; cleanup_moltemplate.sh -base {filename}; cd ..;"
    os.system(bashCommand)
    
    # Next check the settings file and see if LAMMPS hybrid style is even necessarry
    # Begin by reading the entire contents of the *in.settings file
    file = open(directory+f'/{filename}.in.settings', 'r')
    lines = file.readlines()
    string = ""
    for i, line in enumerate(lines):
        if "dihedral_coeff" in line:
            print(line)
            if 'opls' or 'fourier' in line:
                string += line.split()[2]
    file.close()
    
    if 'opls' in string and 'fourier' in string:
        print("No modification necesssarry. Hybrid style is present")
    else:
        print("Modification necesssarry. Only either opls/fourier style present")
        
        #First we modify the lines containing dihedral_coeff and delete the keyword for opls and fourier
        for i, line in enumerate(lines):
            if "dihedral_coeff" in line:
                if 'opls' or 'fourier' in line:
                    modified_line = line.replace('opls', '')
                    modified_line = modified_line.replace('fourier', '')
                    lines[i] = modified_line
        with open(directory+f'/{filename}.in.settings', 'w') as file:
            file.writelines(lines)
        
        # Next, modify the *in.init file. We need to remove the hybrid keyword from that file.
        with open(directory+f'/{filename}.in.init', 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if "dihedral_style" in line:
                if 'opls' or 'fourier' in line:
                    modified_line = line.replace('hybrid', '')
                    if 'opls' in string:
                        modified_line = modified_line.replace('fourier', '')
                    if 'fourier' in string:
                        modified_line = modified_line.replace('opls', '')
                    lines[i] = modified_line
        with open(directory+f'/{filename}.in.init', 'w') as file:
            file.writelines(lines)

def extract_elements_and_counts(formula):
    """ Return the elements and stoichiometry of a chemical species, given its formula
        
        Args:
            formula (string): a chemical formula, which may include their charge. 
            Example: ethanol C2H6O or hexafluorophosphate F6P-
    """
    # Split the formula based on + or - signs and take only the part before it
    formula = formula.split('+')[0].split('-')[0]
    
    # Use regular expression to find elements and stoich
    elements = re.findall('[A-Z][a-z]*[0-9]*', formula)
    
    # Initialize a dictionary to store stoich for each element
    stoich_dict = {}

    # Iterate through the elements to extract stoich
    for element in elements:
        # Extract the element symbol and count (if present)
        match = re.match('([A-Za-z]+)([0-9]*)', element)
        symbol = match.group(1)
        count = match.group(2)
        
        # If count is not provided, assume it's 1
        if count == '':
            count = '1'
        
        # Add the count to the dictionary
        if symbol in stoich_dict:
            stoich_dict[symbol] += int(count)
        else:
            stoich_dict[symbol] = int(count)
    
    # Convert the dictionary to lists of elements and stoich
    elements = list(stoich_dict.keys())
    stoich = [str(stoich_dict[element]) for element in elements]

    return elements, stoich

def calculate_mw(formula):
    """ Calculate molecular weight given the chemical formula
        
        Args:
            formula (string): a chemical formula, which may include their charge. 
            Example: ethanol C2H6O or hexafluorophosphate F6P-
    """
    total_weight = 0.0
    elements, counts = extract_elements_and_counts(formula)
    print("Elements:", elements)
    print("Counts:", counts)
    for i, element in enumerate(elements):
        total_weight += atomic_masses[element] * int(counts[i])
    return total_weight
