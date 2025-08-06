"""
Improve heavy-main group organic coverage by randomly replacing atoms of 1st row main-group

Choose a few patterns to substitute
randomly select a number from 1 to 5 to select
attempt to apply the changes
relax the atoms that have changed with ff for 10 steps (enough to take the edge off)
randomly stretch and shrink a subset of the heavy-mg bonds to ensure we cover both too long and too short
"""
import argparse
import os
import random
from collections import Counter
from functools import partial
from itertools import product
from tqdm import tqdm
import multiprocessing as mp

import numpy as np
from schrodinger.adapter import evaluate_smarts
from schrodinger.application.jaguar.packages.shared import read_cartesians
from schrodinger.forcefield import OPLSVersion
from schrodinger.forcefield.ffld_options import ForceFieldOptions, MinimizationOptions
from schrodinger.forcefield.minimizer import minimize_substructure
from schrodinger.structure import StructureReader
from schrodinger.structutils.rings import find_ring_bonds

chalcogens_smarts = ("[#8X2H1]", "[#8X2H0]", "[#8X1]")
chalcogens_elts = ("Se", "Te")
pnictogens_smarts = ("[#7X3H2]", "[#7X3H1]", "[#7X3H0]", "[#7X2]", "[#7X1]")
pnictogens_elts = ("P", "As", "Sb")
carbon_smarts = (
    "[#6X4H3]",
    "[#6X4H2]",
    "[#6X4H1]",
    "[#6X4H0]",
    "[#6X3H2]",
    "[#6X3H1]",
    "[#6X3H0]",
    "[#6X2H1]",
    "[#6X2H0]",
)
carbon_elts = ("Si", "Ge")
smarts_patterns = [
    list(product(chalcogens_smarts, chalcogens_elts)),
    list(product(pnictogens_smarts, pnictogens_elts)),
    list(product(carbon_smarts, carbon_elts)),
]
MIN_OPTS = MinimizationOptions(max_step=10)
FF_OPTS = ForceFieldOptions(version=OPLSVersion.F14)


def get_random_structure(source="ani2x"):
    st = StructureReader.read(fname)
    st.title = fname
    return st


def select_patterns(st):
    n_patterns = random.choices(range(1, 5), weights=[4,4,2,1])[0]
    sel_patts = {}
    tries = 0
    max_tries = 50
    while len(sel_patts) < n_patterns and tries < max_tries:
        tries += 1
        elt_group = random.choice(smarts_patterns)  # Have an equal prob of replacing C, O, N
        weights = [1 / len(elt_group)] * len(elt_group)
        weights[0] /= 4  # Down weight the terminal case by a factor of 4
        cand_pattern = random.choices(elt_group, weights=weights)[0]
        match_at = [
            at[0]
            for at in evaluate_smarts(st, cand_pattern[0])
            if at[0] not in sel_patts
        ]
        if not match_at:
            continue
        sel_at = random.choice(match_at)
        sel_patts[sel_at] = cand_pattern[1]
    return sel_patts


def make_changes(st, patterns):
    ats_to_opt = []
    for at, new_elt in patterns.items():
        st.atom[at].element = new_elt
        ats_to_opt.append(at)
        # ats_to_opt.extend([b_at.index for b_at in st.atom[at].bonded_atoms])
    minimize_substructure(st, ats_to_opt, MIN_OPTS, FF_OPTS)
    return st


def get_new_fname(old_fname, patterns):
    old_fname = os.path.splitext(old_fname)[0]
    *parts, charge, spin = old_fname.split("_")
    new_fname = (
        "_".join(
            parts + [str(s) for it in patterns.items() for s in it] + [charge, spin]
        )
        + ".mae"
    )
    return new_fname


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, help="Source of system to modify", required=True
    )
    parser.add_argument("--output_path", type=str, help="Where to save files to")
    parser.add_argument(
        "--n_structs", type=int, help="How many structure to sample", default=1
    )
    return parser.parse_args()


def dilate_bond(st, patterns) -> bool:
    """
    Choose a random non-ring bond in the structure that has been changed
    and stetch/shrink it by a scale factor of between 10% and 20%.

    :param st: Structure to alter
    :param patterns: (atom index of changed atoms, new element)
    :return: True if dilation was successful
    """
    ring_bonds = find_ring_bonds(st)
    stretch_at = random.choice(list(patterns))
    try:
        rand_bond = random.choice(
            [bond for bond in st.atom[stretch_at].bond if frozenset((bond.atom1.index, bond.atom2.index)) not in ring_bonds]
        )
    except IndexError:  # No non-ring bonds
        return False
    dilate_amt = random.uniform(0.1, 0.2)
    sign = random.choice((1, -1))
    new_len = rand_bond.length * abs(dilate_amt + sign)
    st.adjust(new_len, *rand_bond.atom)
    return True


def read_xyz(fname):
    cart = read_cartesians(fname)[0]
    *_, charge, _ = fname.split('_')
    cart.charge = int(charge)
    st = cart.getStructure()
    return st


def main(fname, output_path):
    st = read_xyz(fname)
    st.title = os.path.basename(fname)
    patterns = select_patterns(st)
    if not patterns:
        return
    try:
        st = make_changes(st, patterns)
    except:
        return
    if random.random() < 0.15:
        dilated = dilate_bond(st, patterns)
    else:
        dilated = False
    new_fname = get_new_fname(st.title, patterns)
    st.write(os.path.join(output_path, new_fname))
    return dilated


if __name__ == "__main__":
    args = parse_args()
    with open(args.source, "r") as fh:
        fname_list = [f.strip() for f in fh.readlines()]
    random.shuffle(fname_list)
    fname_list = fname_list[: args.n_structs]
    fxn = partial(main, output_path=args.output_path)
    with mp.Pool(60) as pool:
        dilated = list(tqdm(pool.imap(fxn, fname_list), total=args.n_structs))
    print(Counter(dilated))
