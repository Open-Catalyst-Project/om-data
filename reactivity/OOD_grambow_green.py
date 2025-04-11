import numpy as np
from gpsts.geodesic import construct_geodesic_path
from ase import Atoms
from ase.io import read, write
from tqdm import tqdm
import csv
import os
import argparse


def main(args):
    assert os.path.exists(os.path.join(args.input_path, "xyzs_noT1x_noRGD1"))
    assert os.path.exists(os.path.join(args.input_path, "match_gg_rgd1_t1x.csv"))
    rxn_ids = []
    with open(os.path.join(args.input_path, "match_gg_rgd1_t1x.csv"), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[2] == '' and row[1] == '[]':
                rxn_ids.append(int(row[0]))
    for rxn in tqdm(rxn_ids):
        reactant = read(os.path.join(args.input_path, f'xyzs_noT1x_noRGD1/r{rxn:06}.xyz'))
        ts = read(os.path.join(args.input_path, f'xyzs_noT1x_noRGD1/ts{rxn:06}.xyz'))
        product = read(os.path.join(args.input_path, f'xyzs_noT1x_noRGD1/p{rxn:06}.xyz'))
        path = construct_geodesic_path(reactant, ts, nimages=10)
        path += construct_geodesic_path(ts, product, nimages=10)[1:]
        for image_idx, image in enumerate(path):
            # All these examples are neutral and closed shell - but should be run as open shell singlets
            write(os.path.join(args.output_path, f'{rxn}_{image_idx}_0_1.xyz'), image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=".")
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)