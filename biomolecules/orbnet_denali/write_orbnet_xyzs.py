import argparse
import multiprocessing as mp
import os
import tarfile
from urllib.request import urlretrieve

from tqdm import tqdm
import shutil

def copy_with_new_name(args):
    orig_filepath, output_path, nid = args
    split_path = orig_filepath.split("/")
    assert len(split_path) == 3
    specific_subdir = split_path[1]
    file_hash = split_path[-1].split(".")[0]
    with open(orig_filepath) as fh:
        fh.readline()
        line = fh.readline()
        charge, spin = line.split()
    new_filepath = os.path.join(output_path, f"orbnet_{specific_subdir}_{file_hash}_{nid}_{charge}_{spin}.xyz")
    shutil.copyfile(orig_filepath, new_filepath)

def main(args):
    tar_name = "denali_xyz_files.tar.gz"
    xyz_dir_name = "xyz_files"
    if not os.path.exists(tar_name):
        urlretrieve(
            "https://figshare.com/ndownloader/files/28672287",
            "denali_xyz_files.tar.gz",
        )
    if not os.path.exists(xyz_dir_name):
        with tarfile.open(tar_name, "r:gz") as tar:
            tar.extractall()

    pool = mp.Pool(60)
    nid = 0
    os_walk = tuple(os.walk(xyz_dir_name))
    for subdir, dirs, files in tqdm(os_walk, total=len(os_walk)):
        mp_args = []
        for file in files:
            mp_args.append((os.path.join(subdir, file), args.output_path, nid))
            nid += 1

        list(tqdm(pool.imap(copy_with_new_name, mp_args), total=len(mp_args)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
