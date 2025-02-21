from ase import build

from architector.io_molecule import convert_io_molecule
from architector.io_calc import CalcExecutor
from architector.io_align_mol import simple_rmsd, align_rmsd
import architector.io_ptable as io_ptable
from quacc.recipes.orca._base import prep_calculator

from ase.constraints import ExternalForce
from ase.io import read,write # Read in the initial and final molecules.

import shutil
import os
import sys
import pathlib
import copy
from tqdm import tqdm
import argparse
import psutil
import multiprocessing as mp
from datetime import datetime
import numpy as np

import openbabel as ob

from omdata.reactivity_utils import min_non_hh_distance, check_bonds, check_isolated_o2, filter_unique_structures, AFIRPushConstraint, run_afir, find_min_distance

def main(args):
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)
    if not os.path.exists(args.reactant_xyz_path):
        raise ValueError(f"Reactant path not found at {args.reactant_xyz_path}")
    if not os.path.exists(args.product_xyz_path):
        raise ValueError(f"Product path not found at {args.product_xyz_path}")

    reactant = convert_io_molecule(args.reactant_xyz_path)
    product = convert_io_molecule(args.product_xyz_path)
    reactant.create_mol_graph()
    product.create_mol_graph()

    nprocs = psutil.cpu_count(logical=False)

    orca_calc = prep_calculator(
        charge=int(args.charge),
        spin_multiplicity=int(args.spin_multiplicity),
        default_inputs=['B97-3c def2-SVP EnGrad'],
        default_blocks=['%pal nprocs {} end'.format(nprocs)]
    )

    logfile = os.path.join(args.output_path, "afir.log")

    save_trajectory, traj_list=run_afir(reactant,product,orca_calc, logfile)

    # Save the trajectory
    with open(os.path.join(args.output_path, "flat_trajectory.xyz"), "w") as f:
        for atoms in save_trajectory:
            write(f, atoms, format="xyz")

    for ii, entry in enumerate(traj_list):
        with open(os.path.join(args.output_path, f"afir_trajectory_{ii}.xyz"), "w") as f:
            for atoms in entry:
                write(f, atoms, format="xyz")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactant_xyz_path", default=".")
    parser.add_argument("--product_xyz_path", default=".")
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--charge", default=0)
    parser.add_argument("--spin_multiplicity", default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)