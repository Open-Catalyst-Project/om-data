import argparse
import csv
import glob
import os
import multiprocessing as mp
from collections import Counter
from functools import partial
from tqdm import tqdm

from schrodinger.application.jaguar.autots_bonding import clean_st
from schrodinger.application.jaguar.packages.shared import read_cartesians

ION_SPIN = {
    "Ag+2": 1,
    "Co+2": 3,
    "Cr+2": 4,
    "Cr+3": 3,
    "Cu+2": 1,
    "Fe+2": 4,
    "Fe+3": 5,
    "Mn+2": 5,
    "Ni+2": 2,
    "Pd+2": 2,
    "Pt+2": 2,
    "Ti+": 3,
    "OV+2": 1,
    "V+2": 3,
    "V+3": 2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


def count_elements(fname):
    with open(fname, "r") as fh:
        data = fh.readlines()[2:]
    elts = Counter(line.strip().split(" ")[0] for line in data)
    return elts

def rename_tempo(fname):
    carts = read_cartesians(fname)[0]
    st = carts.getStructure()
    st = clean_st(st)
    bonds_to_del = []
    for bond in st.bond:
        if bond.order == 0:
            bonds_to_del.append(bond.atom)
    for bond in bonds_to_del:
        st.deleteBond(*bond)
    cnt = Counter(
        frozenset(Counter(at.atomic_number for at in mol.atom).items())
        for mol in st.molecule
    )
    n_tempo = cnt[frozenset(((1, 18), (6, 9), (7, 1), (8, 1)))]
    new_name = fname.replace("_1.xyz", f"_{n_tempo+1}.xyz")
    os.rename(fname, new_name)

def rename_job(spec, path):
    job_idx, system = spec
    ion_contents = {k.split("+")[0].replace('O',''): v for k, v in ION_SPIN.items() if k in system[5:]}
    for fname in glob.glob(f"{os.path.join(path, str(job_idx))}*"):
        elt_counts = count_elements(fname)
        curr_spin = int(os.path.splitext(os.path.basename(fname))[0].split("_")[-1]) - 1
        n_unpaired_spins = curr_spin
        for elt, count in elt_counts.items():
            n_unpaired_spins += ion_contents.get(elt, 0) * count
        if curr_spin != n_unpaired_spins:
            new_fname = fname.replace(
                f"_{curr_spin + 1}.xyz", f"_{n_unpaired_spins + 1}.xyz"
            )
            print(fname, new_fname)
            os.rename(fname, new_fname)

def main():
    args = parse_args()
    pool = mp.Pool(60)

    # Do TEMPO jobs
    for i in [659, 660, 661, 662, 767, 768, 769, 770, 3681, 3682, 3683, 3684]:
        lst = glob.glob(f"{os.path.join(args.output_path, str(i))}_*xyz")
        list(tqdm(pool.imap(rename_tempo, lst), total=len(lst)))

    # Do metal jobs
    with open("elytes.csv", "r") as f:
        systems = list(csv.reader(f))[1:]
    rename_fxn = partial(rename_job, path=args.output_path)
    list(tqdm(pool.imap(rename_fxn, enumerate(systems, start=1)), total=len(systems)))


if __name__ == "__main__":
    main()
