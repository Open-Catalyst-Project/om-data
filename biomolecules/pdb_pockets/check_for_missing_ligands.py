import argparse
import glob
import os

from schrodinger.structure import StructureReader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    file_list = glob.glob(os.path.join(args.output_path, f"{args.prefix}*.mae"))
    if args.batch is not None:
        file_list = file_list[1000 * args.batch : 1000 * (args.batch + 1)]
    for fname in tqdm(file_list):
        if not os.path.exists(fname):
            continue
        try:
            st = StructureReader.read(fname)
        except:
            print("Cant read", fname)
            continue
        if "l" not in (ch.name for ch in st.chain):
            print("no ligand", fname)
            continue


if __name__ == "__main__":
    main()
