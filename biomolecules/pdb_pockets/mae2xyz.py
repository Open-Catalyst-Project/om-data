import argparse
import glob
import multiprocessing as mp
import os
import sys
from functools import partial

from schrodinger.structure import StructureReader
from tqdm import tqdm

def write_file(fn, old_ext, new_ext):
    try:
        st = StructureReader.read(fn)
    except:
        print(fn)
    xyz_name = os.path.join(os.path.dirname(fn), new_ext, os.path.basename(fn).replace(f".{old_ext}", f".{new_ext}"))
    try:
        st.write(xyz_name)
    except:
        print(fn)


def main(path, file_list, old_ext, new_ext):
    if path is not None:
        file_list = glob.glob(os.path.join(path, f"*.{old_ext}"))
    elif file_list is not None:
        with open(file_list, 'r') as fh:
            file_list = [f.strip() for f in fh.readlines()]

    os.makedirs(os.path.join(os.path.dirname(file_list[0]), new_ext), exist_ok=True)
    parallel_write = partial(write_file, old_ext=old_ext, new_ext=new_ext)
    with mp.Pool(60) as pool:
        list(tqdm(pool.imap(parallel_write, file_list), total=len(file_list)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path")
    parser.add_argument("--file_list")
    parser.add_argument("--old_ext", type=str, default='mae')
    parser.add_argument("--new_ext", type=str, default='xyz')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.output_path, args.file_list, args.old_ext, args.new_ext)
