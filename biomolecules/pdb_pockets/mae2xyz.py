import argparse
import glob
import multiprocessing as mp
import os
import sys
from functools import partial

from schrodinger.structure import StructureReader
from tqdm import tqdm

ftype = 'mae'

def write_file(fn):
    st = StructureReader.read(fn)
    xyz_name = os.path.join(os.path.dirname(fn), 'xyz', os.path.basename(fn).replace(f".{ftype}", ".xyz"))
    try:
        st.write(xyz_name)
    except:
        print(fn)


def main(path, file_list):
    if path is not None:
        file_list = glob.glob(os.path.join(path, f"*.{ftype}"))
    elif file_list is not None:
        with open(file_list, 'r') as fh:
            file_list = [f.strip() for f in fh.readlines()]

    os.makedirs(os.path.join(os.path.dirname(file_list[0]), 'xyz'), exist_ok=True)
    with mp.Pool(60) as pool:
        list(tqdm(pool.imap(write_file, file_list), total=len(file_list)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path")
    parser.add_argument("--file_list")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.output_path, args.file_list)
