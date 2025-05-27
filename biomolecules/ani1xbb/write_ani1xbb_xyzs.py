import argparse
import multiprocessing as mp
import os
import tarfile
from urllib.request import urlretrieve
from architector.io_molecule import convert_io_molecule
import ase.io
import h5py
from ase import Atoms
from ase.data import covalent_radii
from tqdm import tqdm
import numpy as np
import random
from omdata.reactivity_utils import check_isolated_o2, check_isolated_s2

random.seed(42)
bad_lonesome = [5, 6, 7, 8, 14, 15, 16, 34]


def check_lonesome(atoms):

    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()

    for i in range(len(atoms)):
        if numbers[i] not in bad_lonesome:
            continue
        atom_i_radius = covalent_radii[numbers[i]]
        
        for j in range(len(atoms)):
            if i == j:
                continue
                
            dist = np.linalg.norm(positions[i] - positions[j])
            atom_j_radius = covalent_radii[numbers[j]]
            cutoff = (atom_i_radius + atom_j_radius) * 1.8

            if dist < cutoff:
                break
        else:
            # No atoms within cutoff distance found
            return True
            
    return False


def check_and_write_xyz(args):
    atomic_numbers, positions, output_path = args
    atoms = Atoms(atomic_numbers, positions=positions.reshape(-1, 3))
    o2_present = False
    s2_present = False
    bad_lonesome = False
    if 8 in atomic_numbers:
        o2_present = check_isolated_o2(convert_io_molecule(atoms))
    if 16 in atomic_numbers and not o2_present:
        s2_present = check_isolated_s2(convert_io_molecule(atoms))
    if not o2_present and not s2_present:
        bad_lonesome = check_lonesome(atoms)
    if not bad_lonesome and not o2_present and not s2_present:
        ase.io.write(output_path, atoms, "xyz")


def main(args):
    hf_name = "pc_bb_scan_13el_rel.h5"
    if not os.path.exists(hf_name):
        raise ValueError(f"{hf_name} not found")
    with h5py.File(hf_name) as h5:
        for num_atoms, properties in h5.items():
            coordinates = properties["coord"]
            numbers = properties["numbers"]
            charges = properties["charge"]

            mp_args = []
            for nid, (atomic_numbers, positions, charge) in tqdm(
                enumerate(zip(numbers, coordinates, charges))
            ):
                # Save highly charged systems for OOD
                if abs(charge) > 2:
                    continue
                output_path = os.path.join(
                    args.output_path, f"aniBB_{num_atoms}_{nid}_{charge}_1.xyz"
                )
                total_electrons = sum(atomic_numbers) - charge
                if total_electrons % 2 != 0:
                    raise ValueError(f"Total number of electrons is not even: {total_electrons}")
                mp_args.append((atomic_numbers, positions, output_path))

                if random.random() < 0.1:
                    output_path = os.path.join(
                        args.output_path, f"aniBB_{num_atoms}_{nid}_{charge}_3.xyz"
                    )
                    mp_args.append((atomic_numbers, positions, output_path))

            with mp.Pool(60) as pool:
                list(tqdm(pool.imap(check_and_write_xyz, mp_args), total=len(mp_args)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
