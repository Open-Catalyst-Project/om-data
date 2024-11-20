import argparse
import multiprocessing as mp
import os
import tarfile
from urllib.request import urlretrieve

import ase.io
import h5py
from rdkit import Chem
from functools import partial
from ase import Atoms
from tqdm import tqdm


def write_xyz(args):
    atomic_numbers, positions, output_path = args
    atoms = Atoms(atomic_numbers, positions=positions)
    ase.io.write(output_path, atoms, "xyz")

def work(group, hf_name):
    with h5py.File(hf_name, swmr=True) as h5:
        properties = h5[group]
        subset = list(properties["subset"])[0].decode("utf-8").replace(" ", "_").replace('.', '_')
        fixed_group = group.replace(" ", "_").replace('[','').replace(']', '').replace('.','_').replace(':','')
        coordinates = [coords*0.529177 for coords in list(properties["conformations"])]
        species = [[int(val) for val in list(properties["atomic_numbers"])] for entry in coordinates]
        mol = Chem.MolFromSmiles(list(properties["smiles"])[0])
        charge = Chem.GetFormalCharge(mol)

        mp_args = []
        for nid, (atomic_numbers, positions) in enumerate(zip(species, coordinates)):
            output_path = os.path.join(
                args.output_path, f"{subset}_spice_{fixed_group}_{nid}_{charge}_1.xyz"
            )
            if not os.path.exists(output_path):
                write_xyz((atomic_numbers, positions, output_path))


def main(args):
    hf_name = "SPICE-2.0.1.hdf5"
    if not os.path.exists(hf_name):
        urlretrieve(
            "https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5",
            "SPICE-2.0.1.hdf5",
        )
    pool = mp.Pool(60)
    with h5py.File(hf_name) as h5:
        groups = list(h5.keys())
    work_fxn = partial(work, hf_name=hf_name)
    list(tqdm(pool.imap(work_fxn, groups), total=len(groups)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
