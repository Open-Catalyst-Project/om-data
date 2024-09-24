import argparse
import multiprocessing as mp
import os
import random
from urllib.request import urlretrieve

import numpy as np
from pymatgen.core.sites import Site
from pymatgen.core.structure import Molecule
from tqdm import tqdm


def sample(idx):
    natoms = len_list[idx]
    charge = int(charge_list[idx])
    multiplicity = 1
    site_list = []
    for jj in range(natoms):
        site_list.append(Site(elem_list[idx][jj], coords_list[idx][jj]))
    tmp_mol = Molecule.from_sites(site_list)
    rand = random.random()
    if rand < 0.1:
        charge -= 1
        multiplicity = 2
    elif rand < 0.2:
        charge += 1
        multiplicity = 2
    tmp_mol.to(
        os.path.join(args.output_path, f"spf_{idx}_{charge}_{multiplicity}.xyz"),
        "xyz",
    )


def main(args):
    random.seed(823)
    db_name = args.db_path
    if not os.path.exists(db_name):
        urlretrieve(
            "https://zenodo.org/records/2605372/files/solvated_protein_fragments.npz",
            db_name,
        )
    data = np.load(db_name)

    global len_list, charge_list, coords_list, elem_list
    len_list = data["N"]
    charge_list = data["Q"]
    coords_list = data["R"]
    elem_list = data["Z"]

    indices = range(len(len_list))
    pool = mp.Pool(60)
    list(tqdm(pool.imap(sample, indices)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default=".")
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
