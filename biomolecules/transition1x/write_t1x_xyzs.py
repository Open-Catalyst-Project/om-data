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
    hf_name = "transition1x.h5"
    if not os.path.exists(hf_name):
        urlretrieve(
            "https://figshare.com/ndownloader/files/36035789",
            "transition1x.h5",
        )
    pool = mp.Pool(60)
    with h5py.File(hf_name) as h5:
        for formula, grp in tqdm(h5["data"].items()):
            mp_args = []
            for rxn, subgrp in grp.items():
                coordinates = list(subgrp["positions"])
                species = [
                    [int(val) for val in list(subgrp["atomic_numbers"])]
                    for entry in coordinates
                ]
                num_atoms = len(species)

                for nid, (atomic_numbers, positions) in enumerate(
                    zip(species, coordinates)
                ):
                    output_path = os.path.join(
                        args.output_path, f"t1x_{rxn}_{num_atoms}_{nid}_0_1.xyz"
                    )
                    if not os.path.exists(output_path):
                        mp_args.append((atomic_numbers, positions, output_path))
            list(tqdm(pool.imap(write_xyz, mp_args), total=len(mp_args)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
