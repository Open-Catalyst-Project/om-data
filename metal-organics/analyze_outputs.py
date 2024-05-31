import argparse
import os
import pathlib

import pandas as pd
from architector import convert_io_molecule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path")
    return parser.parse_args()


def main():
    args = parse_args()
    path = pathlib.Path(args.output_path)
    for fname in path.glob("*.pkl"):
        name = os.path.join("xyzs", os.path.splitext(os.path.basename(fname))[0])
        df = pd.read_pickle(fname)
        for i, row in df.iterrows():
            mol = convert_io_molecule(row["mol2string"])
            # mol.uhf is the number of unpaired electrons in the molecule
            # multiplicity = n_unpaired electrons + 1
            label = "{}_{}_{}_{}.xyz".format(
                name, i, str(int(mol.charge)), str(int(mol.uhf + 1))
            )
            xyz = mol.write_xyz(label, writestring=False)


if __name__ == "__main__":
    main()
