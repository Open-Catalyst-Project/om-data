import argparse
import os
import pathlib

import pandas as pd
from architector import convert_io_molecule
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path")
    return parser.parse_args()

def parallel_work(fname, xyz_path):
    name = xyz_path / os.path.splitext(os.path.basename(fname))[0]
    df = pd.read_pickle(fname)
    for i, row in df.iterrows():
        mol = convert_io_molecule(row["mol2string"])
        # mol.uhf is the number of unpaired electrons in the molecule
        # multiplicity = n_unpaired electrons + 1
        label = "{}_{}_{}_{}.xyz".format(
            name, i, str(int(mol.charge)), str(int(mol.uhf + 1))
        )
        xyz = mol.write_xyz(label, writestring=False)


def main():
    args = parse_args()
    path = pathlib.Path(args.output_path)
    xyz_path = path / "xyzs"
    xyz_path.mkdir(exist_ok=True)
    xyz_names = {
        os.path.join(name.parent, str(name.name).split("_")[0])
        for name in xyz_path.glob("*.xyz")
    }
    print(len(xyz_names))
    pkl_glob = list(path.glob("*.pkl"))
    print(len(pkl_glob))
    pkl_glob = [f for f in pkl_glob if str(xyz_path / os.path.splitext(os.path.basename(f))[0]) not in xyz_names]
    fxn = partial(parallel_work, xyz_path=xyz_path)
    with mp.Pool(60) as pool:
        list(tqdm(pool.imap(fxn, pkl_glob), total=len(pkl_glob)))

if __name__ == "__main__":
    main()
