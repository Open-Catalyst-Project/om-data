# pip install popcornn
import argparse
import os

import numpy as np
from ase.io import read, write
from popcornn import optimize_MEP, tools
from tqdm import tqdm


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    indices = np.array_split(range(0, 146961), args.slurm_workers)[args.chunk]

    os.makedirs(output_path, exist_ok=True)

    for i in tqdm(indices):
        try:
            print(f"Processing reaction {i}")
            lines = open(f"{input_path}/{i}_R.xyz").readlines()
            charge = lines[1].split()[0]
            spin_multiplicity = lines[1].split()[1][0]
            if os.path.exists(f"{output_path}/{i}_1_{charge}_{spin_multiplicity}.xyz"):
                print(f"Skipping reaction {i} because it already exists")
                continue
            reactant = read(f"{input_path}/{i}_R.xyz")
            product = read(f"{input_path}/{i}_P.xyz")
            config = tools.import_run_config("popcornn_config.yaml")
            final_images, ts_image = optimize_MEP([reactant, product], **config)
            for j, atoms in enumerate(final_images):
                if j == 0 or j == len(final_images) - 1:
                    continue
                write(f"{output_path}/{i}_{j}_{charge}_{spin_multiplicity}.xyz", atoms)
        except Exception as e:
            print(f"Error in {i}, {str(e)})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--slurm-workers", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
