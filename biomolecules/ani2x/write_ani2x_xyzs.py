import argparse
import h5py
import numpy as np
from ase import Atoms
import ase.io
from tqdm import tqdm
from urllib.request import urlretrieve
import os
import tarfile

def main(args):
    tar_name = "ANI-2x-wB97X-631Gd.tar.gz"
    hf_name = "final_h5/ANI-2x-wB97X-631Gd.h5"
    if not os.path.exists(hf_name):
        if not os.path.exists(tar_name):
            urlretrieve(
                "https://zenodo.org/records/10108942/files/ANI-2x-wB97X-631Gd.tar.gz",
                "ANI-2x-wB97X-631Gd.tar.gz",
            )
        with tarfile.open(tar_name, "r:gz") as tar:
            tar.extractall()
    with h5py.File(hf_name) as h5:
        for num_atoms, properties in h5.items():
            coordinates = properties['coordinates']
            species = properties['species']
            nid = 0
            for atomic_numbers, positions in tqdm(zip(species, coordinates)):
                atoms = Atoms(atomic_numbers, positions=positions)
                ase.io.write(os.path.join(args.output_path, f"ani2x_{num_atoms}_{nid}_0_1.xyz"), atoms, "xyz")
                nid += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
