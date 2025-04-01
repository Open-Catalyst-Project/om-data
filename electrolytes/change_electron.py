import argparse
import glob
import multiprocessing as mp
import os
import random
import shutil
from functools import partial

from ase.io import read
from tqdm import tqdm

TM_LIST = {*range(21, 31), *range(39, 49), *range(72, 81)}
random.seed(5122)


def n_trans_metals(fname: str) -> int:
    """
    Count the number of transition metals in an .xyz file

    :param fname: .xyz file to check contents of
    """
    atoms = read(fname)
    return sum(1 for at in atoms.get_atomic_numbers() if at in TM_LIST)


def filter_tm_count(fname: str):
    """
    Check if file has too many (5 or greater) transition metals

    :param fname: .xyz file to check contents of
    :return: `fname` if system is acceptably low, None otherwise
    """
    if n_trans_metals(fname) < 5:
        return fname


def redox(system: str, change: int, output_dir: str) -> None:
    """
    Apply a change in the number of electrons to a given .xyz file

    Note that we always opt for changing the number of electrons leading to
    a higher spin configuration. We can always spin flip down later.

    :param system: (potentially absolute) path to an .xyz file
    :param change: the change in charge to apply to the system
    :param output_dir: the output directory to save data to
    """
    *basename, charge, spin = os.path.basename(system).split("_")
    charge = int(charge)
    spin = int(spin.replace(".xyz", ""))
    new_name = "_".join(basename + [str(charge + change), str(spin + 1)]) + ".xyz"
    shutil.copy(system, os.path.join(output_dir, new_name))


def main(input_dir: str, output_dir: str):
    """
    Take our starting electrolyte geometries and randomly reduce half
    and oxidize the other half.

    :param input_dir: directory containing "neutral" electrolytes
    :param output_dir: directory to place redoxed electrolytes
    """
    flist = glob.glob(os.path.join(input_dir, "*.xyz"))
    add = partial(redox, change=1, output_dir=output_dir)
    subtract = partial(redox, change=-1, output_dir=output_dir)
    with mp.Pool(60) as pool:
        flist = set(tqdm(pool.imap(filter_tm_count, flist), total=len(flist)))
    flist -= {None}
    flist = list(flist)
    print(len(flist))
    random.shuffle(flist)

    with mp.Pool(60) as pool:
        list(tqdm(pool.imap(add, flist[: len(flist) // 2]), total=len(flist) // 2))
        list(tqdm(pool.imap(subtract, flist[len(flist) // 2 :]), total=len(flist) // 2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=".")
    parser.add_argument("--output_dir", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input_path, args.output_path)
