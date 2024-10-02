import argparse
import glob
import os
from tqdm import tqdm

from schrodinger.structure import Structure, StructureReader, StructureWriter
from schrodinger.structutils import measure, analyze

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--batch", type=int) 
    return parser.parse_args()


def main():
    args = parse_args()
    suspect_structures = []
    file_list = glob.glob(os.path.join(args.output_path, '*.pdb'))
    if args.batch is not None:
        file_list = file_list[10000*args.batch:10000*(args.batch+1)]
    for fname in tqdm(file_list):
        try:
            st = StructureReader.read(fname)
        except:
            print(fname)
            raise
        for at1, at2 in measure.get_close_atoms(st, dist=0.5):
            print(at1, at2)
            suspect_structures.append(fname)
    print(suspect_structures)

if __name__ == "__main__":
    main()
