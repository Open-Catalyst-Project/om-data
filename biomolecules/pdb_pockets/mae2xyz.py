import argparse
import glob
import multiprocessing as mp
import os
import sys
from functools import partial

from schrodinger.structure import StructureReader
from tqdm import tqdm

ftype = 'maegz'

def write_file(fn):
    st = StructureReader.read(fn)
    xyz_name = os.path.join(os.path.dirname(fn), 'xyz', os.path.basename(fn).replace(f".{ftype}", ".xyz"))
    st.write(xyz_name)


def main(path):
    file_list = glob.glob(os.path.join(path, f"*.{ftype}"))

    pool = mp.Pool(60)
    list(tqdm(pool.imap(write_file, file_list), total=len(file_list)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.output_path)
