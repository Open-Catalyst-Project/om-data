import argparse
import glob
import multiprocessing as mp
import os
import sys
from functools import partial

from schrodinger.structure import StructureReader
from tqdm import tqdm


def write_file(fn, frame0):
    if not frame0:
        for i in range(9, 0, -1):
            xyz_name = fn.replace("frame0", f"frame{i}")
            mae_name = xyz_name.replace(".xyz", ".mae")
            if os.path.exists(mae_name):
                st = StructureReader.read(mae_name)
                st.write(xyz_name)
                break
    else:
        st = StructureReader.read(fn)
        xyz_name = fn.replace(".mae", ".xyz")
        st.write(xyz_name)


def main(frame0, path):
    if frame0:
        file_list = glob.glob(os.path.join(path, "*frame0*.mae"))
        done_list = [
            os.path.basename(fn)
            for fn in glob.glob(os.path.join(path, "2024_09_26_inputs", "*.xyz"))
            + glob.glob(os.path.join(path, "2024_09_27_inputs", "*.xyz"))
        ]
        done_list = {fn.replace(".xyz", ".mae") for fn in done_list}
        file_list = [fn for fn in file_list if os.path.basename(fn) not in done_list]
    else:
        file_list = glob.glob(os.path.join(path, "*.xyz"))

    pool = mp.Pool(60)
    write_frame = partial(write_file, frame0=frame0)
    list(tqdm(pool.imap(write_frame, file_list), total=len(file_list)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(True, args.output_path)
    main(False, args.output_path)
