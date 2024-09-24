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
            coordinates = properties["coordinates"]
            species = properties["species"]

            mp_args = []
            for nid, (atomic_numbers, positions) in tqdm(
                enumerate(zip(species, coordinates))
            ):
                output_path = os.path.join(
                    args.output_path, f"ani2x_{num_atoms}_{nid}_0_1.xyz"
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
