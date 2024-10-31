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


    with h5py.File(hf_name) as h5:
        for group, properties in h5.items():
            subset = list(properties["subset"])[0].decode("utf-8").replace(" ", "_")
            fixed_group = group.replace(" ", "_")
            group_output_path = os.path.join(args.output_path, subset)
            if not os.path.exists(group_output_path):
                os.mkdir(group_output_path)
            coordinates = [coords*0.529177 for coords in list(properties["conformations"])]
            species = [[int(val) for val in list(properties["atomic_numbers"])] for entry in coordinates]
            charge = int(round(sum(list(properties["mbis_charges"])[0])[0]))

            check_charges = [int(round(sum(charges)[0])) for charges in list(properties["mbis_charges"])]
            for one_charge in check_charges:
                assert one_charge == charge

            mp_args = []
            for nid, (atomic_numbers, positions) in tqdm(
                enumerate(zip(species, coordinates))
            ):
                output_path = os.path.join(
                    group_output_path, f"spice_{fixed_group}_{nid}_{charge}_1.xyz"
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
