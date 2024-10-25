"""molbuilder.py
Author: Muhammad R. Hasyim

Module containing routines to pack molecules at different concentrations. 
It can configure simulations for LAMMPS, in which case the system is prepped
using Packmol which then generate LAMMPS DATA and force field files with Moltemplate

This module is sufficient to run a LAMMPS simulation. A second script (lammps2omm.py)
is used to convert from LAMMPS to OpenMM

It can also configure simulations for Desmond, in which case the system is prepped
using the disordered systems builder and then simulations are initiated using the MultiSim
utility. 
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
import pandas as pd
try:
    from createmonomers import write_monomers
except ImportError:
    print("Error in importing createmonomers. Possible that Schrodinger module is not available. Make sure Schrodinger virtual environment is activated")

def load_csv(filename):
    return pd.read_csv(filename)

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

def compute_density_desmond(directory):
    my_file = Path(f"{directory}/final_density.cms")
    if my_file.is_file():
        subprocess.run(f"$SCHRODINGER/run python3 computedensity.py {directory}",shell=True)

def prep_desmond_md(filename, job_dir, temperature):
    with open(os.path.join(job_dir, f"{filename}_multisim.msj"),"w") as msj:
        time = 1000 #by default the runtime is 1 ns, unless we're running the elyte sim
        frames = 100
        if filename == 'elyte':
            time = 250*1000 #we set elyte MD time to 250 ns
        interval = time/frames
        multisim = f"""task {{ task = "desmond:auto"  }}

### Start relaxation protocol: Compressive

simulate {{
  annealing = false
  ensemble = {{
     brownie = {{
        delta_max = 0.1
     }}
     class = NVT
     method = Brownie
  }}
  temperature = 10.0
  time = 100.0
  timestep = [0.01 0.01 0.03 ]
  title = "Brownian Dynamics 0.1 ns/NVT/10.0 K"
}}

simulate {{
  checkpt = {{
     write_last_step = true
  }}
  ensemble = NVT
  jobname = "$MASTERJOBNAME"
  time = 24.0
  timestep = [0.001 0.001 0.003 ]
  title = "Molecular Dynamics 0.0 ns/NVT/300.0 K/1 fs"
  trajectory = {{
     interval = 4.8
  }}
}}

simulate {{
  checkpt = {{
     write_last_step = true
  }}
  ensemble = NVT
  jobname = "$MASTERJOBNAME"
  temperature = 700.0
  time = 240.0
  timestep = [0.001 0.001 0.003 ]
  title = "Molecular Dynamics 0.2 ns/NVT/700.0 K/1 fs"
  trajectory = {{
     interval = 4.8
  }}
}}

simulate {{
  checkpt = {{
     write_last_step = true
  }}
  jobname = "$MASTERJOBNAME"
  time = 24.0
  timestep = [0.001 0.001 0.003 ]
  title = "Molecular Dynamics 0.0 ns/NPT/1.01325 bar/300.0 K/1 fs"
  trajectory = {{
     interval = 4.8
  }}
}}

simulate {{
  checkpt = {{
     write_last_step = true
  }}
  jobname = "$MASTERJOBNAME"
  time = 240.0
  title = "Molecular Dynamics 0.2 ns/NPT/1.01325 bar/300.0 K/2 fs"
  trajectory = {{
     interval = 24.0
  }}
}}

simulate {{
  checkpt = {{
     write_last_step = true
  }}
  eneseq = {{
     interval = 100.0
  }}
  ensemble = {{
     class = NPT
     method = MTK 
  }}
  jobname = "$MASTERJOBNAME"
  time = {time}
  timestep = [0.002 0.002 0.006 ]
  temperature = {temperature} 
  title = "Molecular Dynamics {time/1000:.1f} ns/NPT/1.01325 bar/{temperature:.2f} K"
  trajectory = {{
     interval = {interval}
     write_velocity = true
  }}

   dir = "."
   compress = ""
}}
"""
        msj.write(multisim)

def run_system_builder(cat,an, solv,Nmols,filename,directory,boxsize=40,mdengine='openmm'):
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
    species = cat+an+solv
    if mdengine == 'desmond':
        #Create a metadata file for the disordered systems builder
        metadata = {}
        metadata["species"] = list(species)
        metadata["composition"] = Nmols
        
        # Collect rows corresponding to the first match for each known entry
        # Load the CSV file
        cations_file = 'cations.csv'
        anions_file = 'anions.csv'
        cations = load_csv(cations_file)
        anions = load_csv(anions_file)

        charges = []
        for sp in species:
            #Check if the species is a cation
            matching_row = cations[cations['formula'] == sp]
            if matching_row.empty != True:
                matching_row = matching_row.iloc[0]
                charges.append(int(matching_row['charge']))
                continue
            #Check if the species is an anion
            matching_row = anions[anions['formula'] == sp]
            if matching_row.empty != True:
                matching_row = matching_row.iloc[0]
                charges.append(int(matching_row['charge']))
                continue

            #If not of this, we set charges to zero
            charges.append(0)
        metadata["charges"] = list(charges)
        print(metadata)
        write_monomers(cat, an, solv, charges, directory)
        
        general_ff = 'S-OPLS'
        #Run the disordered system builder
        command = [
            "$SCHRODINGER/run",
            "disordered_system_builder_gui_dir/disordered_system_builder_driver.py",
            "-molecules", str(int(sum(Nmols))),
            "-composition", str(':'.join(map(str, Nmols))),
            "-pbc", "new_cubic",
            "-density", "0.500",
            "-scale_vdw", "0.50",
            "-obey", "density",
            "-tries_per_mol", "50",
            "-tries_per_dc", "20",
            "-seed", "1234",
            "-no_recolor",
            "-water_fftype", "TIP3P",
            "-split_components",
            "-forcefield", general_ff,
            "monomers.maegz", f"{filename}",
            "-preserve_res_info",
            "-JOBNAME", f"{filename}",
            "-HOST", "localhost:1"
        ]
        print(' '.join(command))
        return command, directory

    elif mdengine == 'openmm':
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
