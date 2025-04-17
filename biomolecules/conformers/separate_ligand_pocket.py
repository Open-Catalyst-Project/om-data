import argparse
import glob
import os

from schrodinger.structure import StructureReader
from tqdm import tqdm


def main(output_path):
    dest_path = os.path.join(os.path.dirname(output_path), "lig_pock_eval")
    os.makedirs(dest_path, exist_ok=True)
    for fname in tqdm(glob.glob(f"{output_path}/*frame0*mae")):
        st = StructureReader.read(fname)
        lig = st.chain["l"].extractStructure()
        lig_ats = st.chain["l"].getAtomList()
        pocket = st.copy()
        pocket.deleteAtoms(lig_ats)
        basename = os.path.splitext(os.path.basename(fname))[0]
        lig.write(
            os.path.join(dest_path, f"{basename}_ligand_{lig.formal_charge}_1.mae")
        )
        pocket.write(
            os.path.join(dest_path, f"{basename}_pocket_{pocket.formal_charge}_1.mae")
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.output_path)
