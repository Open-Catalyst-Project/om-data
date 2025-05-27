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
import pandas as pd
from monty.serialization import loadfn

import openbabel as ob

from omdata.reactivity_utils import min_non_hh_distance, check_bonds, check_isolated_o2, filter_unique_structures, AFIRPushConstraint, run_afir, find_min_distance

def metal_complex_pipeline(name, sdf, charge, uhf, output_path):
    os.makedirs(os.path.join(output_path, name), exist_ok=False)

    logfile = os.path.join(output_path, name, "logfile.txt")

    mols = convert_io_molecule(sdf)

    macemp0calc = mace_mp(model="medium", 
               dispersion=True, 
               default_dtype="float64", 
               device='cpu'
              )

    save_trajectory, traj_list=run_afir(mols[0],mols[1],macemp0calc, logfile)

    unique_structs = filter_unique_structures(save_trajectory)

    with open(logfile, 'a') as file1:
        file1.write(f"Found {len(unique_structs)} unique structures\n")

    for i, atoms in enumerate(unique_structs):
        write(os.path.join(output_path, name, f"afir_{i}_{charge}_{uhf+1}.xyz"), atoms, format="xyz")


def main(args):
#    if os.path.exists(args.output_path):
#        shutil.rmtree(args.output_path)
#    os.makedirs(args.output_path)
    num_ood_reactions = 0
    if not os.path.exists(os.path.join(args.input_path, "all_combined.pkl")):
        raise ValueError(f"all_combined.pkl not found at {args.input_path}")
    if not os.path.exists(os.path.join(args.input_path, "charge_spin_dict.json")):
        raise ValueError(f"charge_spin_dict.json not found at {args.input_path}")
    if not os.path.exists(os.path.join(args.input_path, "fix_charge_spin_dict.json")):
        raise ValueError(f"fix_charge_spin_dict.json not found at {args.input_path}")

    df = pd.read_pickle(os.path.join(args.input_path, "all_combined.pkl"))
    good_df = df[~df.sdf_file.isna()]
    old_charge_spin_dict = loadfn(os.path.join(args.input_path, "charge_spin_dict.json"))
    fix_charge_spin_dict = loadfn(os.path.join(args.input_path, "fix_charge_spin_dict.json"))
    num_none = len(good_df[good_df.orig_name.str.contains("None")])
    print("Maximum index: ", len(good_df))
    print("Number of None: ", num_none)
    for ii in tqdm(range(args.start_index, args.end_index)):
        try:
            name = good_df.orig_name.iloc[ii][:-4]
            new_name = good_df.new_name.iloc[ii][:-4]
        except:
            print(f"Error at index {ii}")
            continue
        if "None" in name: # Reactions without any ligand swaps - OOD
            orig_charge = old_charge_spin_dict[name]['total_charge']
            orig_uhf = old_charge_spin_dict[name]['total_uhf']
            if name in fix_charge_spin_dict.keys():
                print("Fixing charge and spin for ", name)
                charge = fix_charge_spin_dict[name]['total_charge']
                uhf = fix_charge_spin_dict[name]['total_spin']
                assert orig_charge == charge
                assert orig_uhf != uhf
            else:
                charge = orig_charge
                uhf = orig_uhf
            sdf = good_df.sdf_file.iloc[ii]
            metal_complex_pipeline(new_name, sdf, charge, uhf, args.output_path)
            num_ood_reactions += 1
    print(f"Number of OOD reactions: {num_ood_reactions}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=".")
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--start_index", type=int)
    parser.add_argument("--end_index", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
