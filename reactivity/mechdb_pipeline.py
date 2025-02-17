from mace.calculators import mace_mp
from ase import build

from architector.io_molecule import convert_io_molecule
from architector.io_calc import CalcExecutor
from architector.io_align_mol import simple_rmsd, align_rmsd
import architector.io_ptable as io_ptable

from ase.constraints import ExternalForce
from ase.io import read,write # Read in the initial and final molecules.

import shutil
import os
import sys
import pathlib
import copy
from tqdm import tqdm
import argparse
import multiprocessing as mp
from datetime import datetime
import numpy as np

import openbabel as ob

def min_non_hh_distance(molecule):
    """Find minimum pairwise distance excluding H-H pairs
    
    Parameters
    ----------
    molecule : architector.io_molecule.Molecule
        Input molecule
        
    Returns
    -------
    float
        Minimum distance in Angstroms between non H-H atom pairs
    """
    # Get atomic symbols and positions
    symbols = molecule.ase_atoms.get_chemical_symbols()
    positions = molecule.ase_atoms.get_positions()
    
    min_dist = float('inf')
    
    # Loop through unique pairs
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            # Skip if both atoms are hydrogen
            if symbols[i] == 'H' and symbols[j] == 'H':
                continue
                
            # Calculate distance
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = min(min_dist, dist)
            
    return min_dist

def check_bonds(mol,
                bonds_breaking,
                bonds_forming,
                breaking_cutoff,
                forming_cutoff): 
    """
    Convergence check function
    Check that bonds are broken/formed that are desired.
    """
    dists = mol.ase_atoms.get_all_distances()
    anums = mol.ase_atoms.get_atomic_numbers()
    goods = []
    for inds in bonds_breaking:
        cutoff_dist = (io_ptable.rcov1[anums[inds[0]]] + io_ptable.rcov1[anums[inds[1]]])*breaking_cutoff
        actual_dist = dists[inds[0]][inds[1]]
        if actual_dist > cutoff_dist:
            goods.append(True)
        else:
            goods.append(False)
    for inds in bonds_forming:
        cutoff_dist = (io_ptable.rcov1[anums[inds[0]]] + io_ptable.rcov1[anums[inds[1]]])*forming_cutoff
        actual_dist = dists[inds[0]][inds[1]]
        if actual_dist < cutoff_dist:
            goods.append(True)
        else:
            goods.append(False)
    return np.all(goods)

class AFIRPushConstraint():

    def __init__(self, a1, a2, f_ext, max_dist = None):
        self.indices = [a1, a2]
        self.external_force = f_ext
        self.max_dist = max_dist

    def get_removed_dof(self, atoms):
        return 0

    def adjust_positions(self, atoms, new):
        pass

    def adjust_forces(self, atoms, forces):
        dist = np.subtract.reduce(atoms.positions[self.indices])
        if self.max_dist is not None and np.linalg.norm(dist) < self.max_dist:
            force = self.external_force * dist / np.linalg.norm(dist)
            forces[self.indices] += (force, -force)

    def adjust_potential_energy(self, atoms):
        dist = np.subtract.reduce(atoms.positions[self.indices])
        if self.max_dist is not None and np.linalg.norm(dist) < self.max_dist:
            return -np.linalg.norm(dist) * self.external_force
        else:
            return 0

def find_min_distance(atoms):
    distances=atoms.get_all_distances()
    np.fill_diagonal(distances, np.inf)
    return np.min(distances)

def run_afir(mol1, mol2, calc, logfile):
    breaking_cutoff=1.5 # When a bond is breaking, what the distance should be
    forming_cutoff=1.2 # When a bond is forming, what the distance should be (Angstroms)
    start_force_constant=0.1 # eV/angstrom
    force_increment=0.2 # How fast to ramp up the force constant
    max_steps=50 # Steps/opimization iteration
    fmax_opt=0.15 # Cutoff for the maximum force.

    method='custom'
    
    bonds_forming = [(int(x[0]), int(x[1])) for x in zip(*np.where((mol2.graph - mol1.graph) == 1)) if x[0] < x[1]]
    # Find the broken bonds
    bonds_breaking = [(int(x[0]), int(x[1])) for x in zip(*np.where((mol2.graph - mol1.graph) == -1)) if x[0] < x[1]]

    with open(logfile, 'a') as file1:
        file1.write(f"Bonds forming: {bonds_forming}\n")
        file1.write(f"Bonds breaking: {bonds_breaking}\n")
    
    fconst = start_force_constant
    save_trajectory = [] # Full output trajectory
    opt_mol = copy.deepcopy(mol1)
    
    keep_going = True # Exit flag for loop
    nstep = 0 
    failure_number=10

    traj_list=[]
    
    start = datetime.now()
    
    failed = False
    
    with open(logfile, 'a') as file1:
        file1.write('Started at: {}\n'.format(str(start)))

    
    while keep_going and fconst < 4.0:
    
        with open(logfile,'a') as file1:
            file1.write('Running Fconst = {}\n'.format(fconst))
        
        opt_mol.ase_atoms.set_constraint()
        constraints = []
        for inds in bonds_forming: # Add linear force
            constraint = ExternalForce(inds[0], inds[1], -fconst)
            constraints.append(constraint)
        for inds in bonds_breaking:
            constraint = AFIRPushConstraint(inds[0], inds[1], fconst, 5.0)
            constraints.append(constraint)
        opt_mol.ase_atoms.set_constraint(constraints)

        tmpopt = CalcExecutor(opt_mol,
                              method=method,
                              calculator=calc,
                              fmax=fmax_opt,
                              maxsteps=max_steps,
                              relax=True,
                              save_trajectories=True,
                              use_constraints=True,
                              debug=False,
                             )
        if tmpopt.successful:
            min_distance=np.min([find_min_distance(atoms) for atoms in tmpopt.trajectory])
            if min_distance < 0.8:
                with open(logfile, 'a') as file1:
                    file1.write(f"Min distance {min_distance} < 0.8. Stopping optimization.\n")
                break
            failure_number = 10 # Reset failures tracking.
            save_trajectory += tmpopt.trajectory
            traj_list.append(tmpopt.trajectory)
            opt_mol = tmpopt.mol
            good = check_bonds(opt_mol, bonds_breaking, bonds_forming,
                                breaking_cutoff, forming_cutoff)
            if good:
                keep_going = False
            else:
                fconst += force_increment
                nstep += 1
        else:
            with open(logfile,'a') as file1:
                file1.write('FAILED At Fconst: {}, Step: {}\n'.format(fconst,nstep))
            if len(tmpopt.trajectory) > 0: # Save trajectory
                save_trajectory += tmpopt.trajectory
            keep_going = False

    return save_trajectory, traj_list


def check_isolated_o2(molecule):
    """Check if molecule contains an isolated O2 molecule
    
    Parameters
    ----------
    molecule : architector.io_molecule.Molecule
        Input molecule
        
    Returns
    -------
    bool
        True if isolated O2 is found, False otherwise
    """
    # Ensure molecular graph exists
    if len(molecule.graph) < 1:
        molecule.create_mol_graph()
        
    # Get atomic symbols and connectivity
    symbols = molecule.ase_atoms.get_chemical_symbols()
    
    # Find oxygen indices
    oxygen_indices = [i for i, sym in enumerate(symbols) if sym == 'O']
    
    # Check each oxygen pair
    for i in oxygen_indices:
        for j in oxygen_indices:
            if i < j:  # Avoid checking same pair twice
                # Check if these oxygens are bonded
                if molecule.graph[i][j] > 0:
                    # Count total bonds for each oxygen
                    bonds_i = sum(molecule.graph[i])
                    bonds_j = sum(molecule.graph[j])
                    
                    # If both oxygens only have one bond (to each other)
                    if bonds_i == 1 and bonds_j == 1:
                        return True
                        
    return False


def filter_unique_structures(atoms_list, rmsd_cutoff):
    """Filter structures to keep only those with RMSD > cutoff from all others
    
    Parameters
    ----------
    atoms_list : list[ase.atoms.Atoms]
        List of structures to filter
    rmsd_cutoff : float, optional
        RMSD threshold for considering structures different, default 0.5 Å
        
    Returns
    -------
    list[ase.atoms.Atoms]
        Filtered list containing only sufficiently different structures
    """

    unique_structures = [atoms_list[0]]  # Start with first structure
    
    # Loop through remaining structures
    for atoms in atoms_list[1:]:
        is_unique = True
        
        # Compare against all previously accepted structures
        for ref_atoms in unique_structures:
            rmsd = simple_rmsd(atoms, ref_atoms)
            if rmsd < rmsd_cutoff:
                is_unique = False
                break
                
        if is_unique:
            unique_structures.append(atoms)
            
    return unique_structures

def mechdb_pipeline(input_path, file_name, output_path):
    input_file = os.path.join(input_path, file_name)
    reaction_name = file_name.split(".")[0]
    os.makedirs(os.path.join(output_path, reaction_name), exist_ok=False)

    logfile = os.path.join(output_path, reaction_name, "logfile.txt")

    # Create OpenBabel conversion objects
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "mol2")

    # Create a molecule object
    mol = ob.OBMol()

    mol_list = []
    not_at_end = obConversion.ReadFile(mol, input_file)
    while not_at_end:
        mol_copy = ob.OBMol(mol)
        mol_list.append(mol_copy)
        mol.Clear()
        not_at_end = obConversion.Read(mol)

    ob_reactant_charge=mol_list[0].GetTotalCharge()
    ob_product_charge=mol_list[1].GetTotalCharge()
    
    reactant_file = os.path.join(output_path, reaction_name, "reactant.mol2")
    obConversion.WriteFile(mol_list[0], reactant_file)

    product_file = os.path.join(output_path, reaction_name, "product.mol2")
    obConversion.WriteFile(mol_list[1], product_file)

    reactant = convert_io_molecule(reactant_file)
    reactant.charge=ob_reactant_charge
    reactant.xtb_charge=ob_reactant_charge
    product = convert_io_molecule(product_file)
    product.charge=ob_product_charge
    product.xtb_charge=ob_product_charge

    bad_structs = min_non_hh_distance(reactant) < 0.88 or min_non_hh_distance(product) < 0.88

    if bad_structs:
        with open(logfile, 'a') as file1:
            file1.write("Endpoint too poor! Discarding reaction\n")
        return
    
    if len(reactant.find_metals()) == 0:
        reactant_uff_calc = CalcExecutor(
            reactant,
            method="UFF", 
            relax=True,
            maxsteps=50,
            fmax=0.1
        )
        with open(logfile, 'a') as file1:
            file1.write(f"Reactant UFF relaxation RMSD = {reactant_uff_calc.rmsd}\n")
            file1.write(f"Reactant post-UFF min distance = {min_non_hh_distance(reactant_uff_calc.mol)}\n")
        if reactant_uff_calc.rmsd < 100:
            reactant = reactant_uff_calc.mol
        else:
            with open(logfile, 'a') as file1:
                file1.write("Reactant UFF relaxation went crazy. Using initial structure for GFN-FF relaxation\n")

        product_uff_calc = CalcExecutor(
            product,
            method="UFF", 
            relax=True,
            maxsteps=50,
            fmax=0.1
        )
        with open(logfile, 'a') as file1:
            file1.write(f"Product UFF relaxation RMSD = {product_uff_calc.rmsd}\n")
            file1.write(f"Product post-UFF min distance = {min_non_hh_distance(product_uff_calc.mol)}\n")
        if product_uff_calc.rmsd < 100:
            product = product_uff_calc.mol
        else:
            with open(logfile, 'a') as file1:
                file1.write("Product UFF relaxation went crazy. Using initial structure for GFN-FF relaxation\n")

        
    reactant_gff_calc = CalcExecutor(
        reactant,
        method="GFN-FF", 
        relax=True,
        maxsteps=50,
        fmax=0.1
    )
    with open(logfile, 'a') as file1:
        file1.write(f"Reactant GFN-FF relaxation RMSD = {reactant_gff_calc.rmsd}\n")
        file1.write(f"Reactant post-GFN-FF min distance = {min_non_hh_distance(reactant_gff_calc.mol)}\n")

    product_gff_calc = CalcExecutor(
        product,
        method="GFN-FF", 
        relax=True,
        maxsteps=50,
        fmax=0.1
    )
    with open(logfile, 'a') as file1:
        file1.write(f"Product GFN-FF relaxation RMSD = {product_gff_calc.rmsd}\n")
        file1.write(f"Product post-GFN-FF min distance = {min_non_hh_distance(product_gff_calc.mol)}\n")

   
    with open(logfile, 'a') as file1:
        file1.write(f"Old reactant charge and spin: {reactant_gff_calc.mol.charge}, {reactant_gff_calc.mol.uhf+1}\n")
    reactant_gff_calc.mol.detect_charge_spin()
    with open(logfile, 'a') as file1:
        file1.write(f"New reactant charge and spin: {reactant_gff_calc.mol.charge}, {reactant_gff_calc.mol.uhf+1}\n")
    
    num_metals = len(reactant_gff_calc.mol.find_metals())
    has_isolated_o2 = check_isolated_o2(reactant_gff_calc.mol)
    if num_metals > 1:
        reactant_gff_calc.mol.uhf = reactant_gff_calc.mol.uhf%2
        with open(logfile, 'a') as file1:
            file1.write(f"Multiple metals! Switching to low spin. New multiplicity = {reactant_gff_calc.mol.uhf+1}\n")

    if has_isolated_o2:
        reactant_gff_calc.mol.uhf += 2
        with open(logfile, 'a') as file1:
            file1.write(f"Isolated O2 found! Switching to high spin. New multiplicity = {reactant_gff_calc.mol.uhf+1}\n")

    macemp0calc = mace_mp(model="medium", 
               dispersion=True, 
               default_dtype="float64", 
               device='cpu'
              )

    save_trajectory, traj_list=run_afir(reactant_gff_calc.mol,product_gff_calc.mol,macemp0calc, logfile)

    starting_cutoff = 0.12
    unique_structs = filter_unique_structures(save_trajectory,starting_cutoff)

    while len(unique_structs) < 10 and starting_cutoff > 0.04:
        starting_cutoff -= 0.01
        unique_structs = filter_unique_structures(save_trajectory,starting_cutoff)

    with open(logfile, 'a') as file1:
        file1.write(f"Found {len(unique_structs)} unique structures\n")

    for i, atoms in enumerate(unique_structs):
        write(os.path.join(output_path, reaction_name, f"afir_{i}_{reactant_gff_calc.mol.charge}_{reactant_gff_calc.mol.uhf+1}.xyz"), atoms, format="xyz")




def main(args):
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)
    if not os.path.exists(args.mechdb_sdfs_path):
        raise ValueError(f"Path to MechDB SDFs not found at {args.mechdb_sdfs_path}")

    for i in range(args.start_index, args.end_index):
        file_name = f"rmechdb_{i}.sdf"
        if not os.path.exists(os.path.join(args.mechdb_sdfs_path, file_name)):
            file_name = f"pmechdb_{i}.sdf"
        if not os.path.exists(os.path.join(args.mechdb_sdfs_path, file_name)):
            print(f"File {i} not found")
            continue
        mechdb_pipeline(args.mechdb_sdfs_path, file_name, args.output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechdb_sdfs_path", default=".")
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--start_index", type=int)
    parser.add_argument("--end_index", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)