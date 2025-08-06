import argparse
import glob
import multiprocessing as mp

import numpy as np
from schrodinger.application.matsci import clusterstruct
from schrodinger.application.matsci.nano.xtal import connect_atoms
from schrodinger.structure import StructureReader
from tqdm import tqdm


def fix_mol(fname):
    st = StructureReader.read(fname)
    connect_atoms(st)
    clusterstruct.contract_structure(st)
    st.write(fname.replace(".cif", ".xyz"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=1)
    return parser.parse_args()


def main(n_chunks, chunk_idx, n_workers):
    cif_list = glob.glob("/checkpoint/levineds/omc_extracts/molecules/val/*.cif")
    chunks_to_process = np.array_split(cif_list, n_chunks)
    chunk = chunks_to_process[chunk_idx]
    with mp.Pool(n_workers) as pool:
        list(tqdm(pool.imap(fix_mol, chunk), total=len(chunk)))


if __name__ == "__main__":
    args = parse_args()
    main(args.n_chunks, args.chunk_idx, args.n_workers)
