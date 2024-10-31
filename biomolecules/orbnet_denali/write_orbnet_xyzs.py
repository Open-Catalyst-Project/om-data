import argparse
import multiprocessing as mp
import os
import tarfile
from urllib.request import urlretrieve

import ase.io
import h5py
from ase import Atoms
from tqdm import tqdm


def write_xyz(args):
    atomic_numbers, positions, output_path = args
    atoms = Atoms(atomic_numbers, positions=positions)
    ase.io.write(output_path, atoms, "xyz")


def main(args):
    tar_name = "denali_xyz_files.tar.gz"
    xyz_dir_name = "denali_xyz_files"
    if not os.path.exists(tar_name):
        urlretrieve(
            "https://figshare.com/ndownloader/files/28672287",
            "denali_xyz_files.tar.gz",
        )
    if not os.path.exists(xyz_dir_name):
        with tarfile.open(tar_name, "r:gz") as tar:
            tar.extractall()
    with h5py.File(hf_name) as h5:
        for formula, grp in h5["data"].items():
            for rxn, subgrp in grp.items():
                coordinates = list(subgrp["positions"])
                species = [
                    [int(val) for val in list(subgrp["atomic_numbers"])]
                    for entry in coordinates
                ]
                num_atoms = len(species)

                mp_args = []
                for nid, (atomic_numbers, positions) in tqdm(
                    enumerate(zip(species, coordinates))
                ):
                    output_path = os.path.join(
                        args.output_path, f"t1x_{rxn}_{num_atoms}_{nid}_0_1.xyz"
                    )
                    mp_args.append((atomic_numbers, positions, output_path))

                pool = mp.Pool(60)
                list(tqdm(pool.imap(write_xyz, mp_args), total=len(mp_args)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
