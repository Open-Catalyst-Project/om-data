import argparse
import glob
import os
import random

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--sample_df")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(23147)
    if args.sample_df is None:
        file_list = glob.glob(os.path.join(args.output_path, "*.maegz"))
#        file_list = [f for f in file_list if 'ZINC' in f or 'CHEMBL' in f]
    else:
        sample_df = pd.read_pickle(args.sample_df)
        file_list = [os.path.join(args.output_path, fname) for fname in sample_df.index]
    with open("300K_paths_list.txt", "w") as fh_300, open("400K_paths_list.txt", "w") as fh_400:
        for fname in file_list:
            # random split uses 0.15 for pdb_pockets, 0.5 for pdb_fragments
            if random.random() < 0.5:
                fh_400.write(fname + "\n")
            else:
                fh_300.write(fname + "\n")


if __name__ == "__main__":
    main()
