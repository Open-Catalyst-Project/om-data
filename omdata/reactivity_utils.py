
from ase import build

from architector.io_molecule import convert_io_molecule
from architector.io_calc import CalcExecutor
from architector.io_align_mol import simple_rmsd, align_rmsd
import architector.io_ptable as io_ptable

from ase.constraints import ExternalForce, FixAtoms
from ase.io import read,write # Read in the initial and final molecules.
from ase.optimize import BFGSLineSearch
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

def run_afir(mol1, mol2, calc, logfile, 
             bonds_forming=[], bonds_breaking=[],
             force_step=0.2,
             maxforce=4.0,
             is_polymer=False):
    breaking_cutoff=5.0 # When a bond is breaking, what the distance should be
    forming_cutoff=1.2 # When a bond is forming, what the distance should be (Angstroms)
    start_force_constant=0.1 # eV/angstrom
    force_increment=force_step # How fast to ramp up the force constant
    max_steps=50 # Steps/opimization iteration
    fmax_opt=0.15 # Cutoff for the maximum force.

    method='custom'

    if mol2: # if product structure is specified, will override bonds_forming and bonds_breaking
        # Instead of specifying from reactant to product, get the formed bonds
        bonds_forming = [(int(x[0]), int(x[1])) for x in zip(*np.where((mol2.graph - mol1.graph) == 1)) if x[0] < x[1]]
        # Find the broken bonds
        bonds_breaking = [(int(x[0]), int(x[1])) for x in zip(*np.where((mol2.graph - mol1.graph) == -1)) if x[0] < x[1]]

    with open(logfile, 'a') as file1:
        if bonds_forming: file1.write(f"Bonds forming: {bonds_forming}\n")
        if bonds_breaking: file1.write(f"Bonds breaking: {bonds_breaking}\n")
    
    fconst = start_force_constant
    save_trajectory = [] # Full output trajectory
    opt_mol = mol1
    
    keep_going = True # Exit flag for loop
    nstep = 0 
    failure_number=10

    traj_list=[]
    
    start = datetime.now()
    
    failed = False
    
    with open(logfile, 'a') as file1:
        file1.write('Started at: {}\n'.format(str(start)))

    
    while keep_going and fconst < maxforce: 
    
        with open(logfile,'a') as file1:
            file1.write('Running Fconst = {}\n'.format(fconst))
        
        opt_mol.ase_atoms.set_constraint()
        constraints = []
        for mol in (mol1, mol2):
            if is_polymer:
                ends = mol.ends
                constraint = FixAtoms(ends)
                constraints.append(constraint)
        
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

        if not tmpopt.successful:
            tmpopt = CalcExecutor(opt_mol,
                              method=method,
                              calculator=calc,
                              fmax=fmax_opt,
                              maxsteps=max_steps,
                              relax=True,
                              ase_opt_method=BFGSLineSearch,
                              save_trajectories=True,
                              use_constraints=True,
                              debug=False,
                             )

        if tmpopt.successful:
            min_distance=np.min([find_min_distance(atoms) for atoms in tmpopt.trajectory])
            if min_distance < 0.7:
                with open(logfile, 'a') as file1:
                    file1.write(f"Min distance {min_distance} < 0.7. Stopping optimization.\n")
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


def check_isolated_s2(molecule):
    """Check if molecule contains an isolated S2 molecule
    
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
    
    # Find sulfur indices
    sulfur_indices = [i for i, sym in enumerate(symbols) if sym == 'S']
    
    # Check each oxygen pair
    for i in sulfur_indices:
        for j in sulfur_indices:
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


def filter_unique_structures_simple(atoms_list, rmsd_cutoff):
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


def filter_unique_structures(atoms_list):
    max_energy=-np.inf
    max_energy_structure=None
    for ii, atoms in enumerate(atoms_list):
        if atoms.get_potential_energy() > max_energy:
            max_energy = atoms.get_potential_energy()
            max_energy_structure = atoms

    # Note that this dict must be ordered from largest to smallest keys
    rmsd_cutoff_dict = {0.6: 0.03, 0.4: 0.04, 0.3: 0.07, 0.18: 0.1, 0.0: 0.14}

    unique_structures = [max_energy_structure]  # Start with highest energy
    
    # Loop through structures
    for ii, atoms in enumerate(atoms_list):
        is_unique = True
        avg_force = np.mean(np.linalg.norm(atoms.get_forces(), axis=1))
        if ii == 0:
            deltaE = deltaE = abs(atoms.get_potential_energy()-atoms_list[ii+1].get_potential_energy())
        elif ii == len(atoms_list)-1:
            deltaE = abs(atoms.get_potential_energy()-atoms_list[ii-1].get_potential_energy())
        else:
            left_deltaE = abs(atoms.get_potential_energy()-atoms_list[ii-1].get_potential_energy())
            right_deltaE = abs(atoms.get_potential_energy()-atoms_list[ii+1].get_potential_energy())
            deltaE = max(left_deltaE, right_deltaE)

        avg_force = max(avg_force, deltaE)

        for cutoff, rmsd_cutoff in rmsd_cutoff_dict.items():
            if avg_force > cutoff:
                break
        
        # Compare against all previously accepted structures
        for ref_atoms in unique_structures:
            rmsd = simple_rmsd(atoms, ref_atoms)
            if rmsd < rmsd_cutoff:
                is_unique = False
                break
                
        if is_unique:
            unique_structures.append(atoms)
            
    return unique_structures
