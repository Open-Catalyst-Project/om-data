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
from datetime import datetime
import numpy as np

import openbabel as ob

from omdata.reactivity_utils import min_non_hh_distance, check_bonds, check_isolated_o2, filter_unique_structures, AFIRPushConstraint, run_afir, find_min_distance

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

    unique_structs = filter_unique_structures(save_trajectory)

    with open(logfile, 'a') as file1:
        file1.write(f"Found {len(unique_structs)} unique structures\n")

    for i, atoms in enumerate(unique_structs):
        write(os.path.join(output_path, reaction_name, f"afir_{i}_{reactant_gff_calc.mol.charge}_{reactant_gff_calc.mol.uhf+1}.xyz"), atoms, format="xyz")




def main(args):
#    if os.path.exists(args.output_path):
#        shutil.rmtree(args.output_path)
#    os.makedirs(args.output_path)
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
