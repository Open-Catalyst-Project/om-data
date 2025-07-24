import argparse
import itertools
#import multiprocessing as mp
import numpy as np
import os
import random
import tempfile
from contextlib import chdir
from functools import partial

from schrodinger.adapter import evaluate_smarts, to_structure
from schrodinger.application.jaguar.packages.shared import (
    read_cartesians, uniquify_with_comparison)
from schrodinger.application.jaguar.packages.tautomer_enumerator import (
    NoTautomersError, get_tautomers)
from schrodinger.comparison import are_conformers
from schrodinger.structutils import build
from tqdm import tqdm


def powerset(iterable):
    """
    Get the powerset of an iterable (the set of all subsets) as
    generator, ignoring the empty set.

    e.g. powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    :type iterable: iterable
    :param iterable: iterable from which to construct powerset
    :rtype: generator
    :return: the set of all subsets of the iterable
    """
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def deprotonate_carbons(st):
    rejects = set()
    patt_list = ["[CH3]"] + ["[CH2]"] * 2 + ["[CH]"] * 4 + ["[cH]"] * 3 +["[Si!H0]"] + ["[Ge!H0]"]
    random.shuffle(patt_list)
    for patt in patt_list:
        if patt in rejects:
            continue
        deprot_site = evaluate_smarts(st, patt)
        if deprot_site:
            break
        else:
            rejects.add(patt)
    site = random.choice(deprot_site)
    st_copy = st.copy()
    at = st_copy.atom[site[0]]
    at.formal_charge -= 1
    h_at = next((b_at for b_at in at.bonded_atoms if b_at.element == "H"), None)
    st_copy.deleteAtoms([h_at])
    return st_copy

def protonate_sts(st_list, pattern):
    prot_sites = evaluate_smarts(st_list[0], pattern)
    addl_sts = []
    for st in st_list:
        for site_set in powerset(prot_sites):
            st_copy = st.copy()
            sites = []
            for site in site_set:
                st_copy.atom[site[0]].formal_charge = 1
                sites.append(site[0])
            build.add_hydrogens(st_copy, atom_list=sites)
            addl_sts.append(st_copy)
    return addl_sts

def deprotonate_sts(st_list, pattern):
    deprot_sites = set.intersection(
        *[set(evaluate_smarts(st, pattern)) for st in st_list]
    )
    addl_sts = []
    for st in st_list:
        for site_set in powerset(deprot_sites):
            st_copy = st.copy()
            h_ats = []
            skip = False
            for site in site_set:
                at = st_copy.atom[site[0]]
                if at.formal_charge > 0:
                    skip = True
                    break
                at.formal_charge -= 1
                h_ats.append(
                    next(
                        (b_at for b_at in at.bonded_atoms if b_at.element == "H"), None
                    )
                )
            if skip:
                continue
            st_copy.deleteAtoms(h_ats)
            addl_sts.append(st_copy)
    return addl_sts

def read_xyz(fname):
    cart = read_cartesians(fname)[0]
    *_, charge, _ = fname.split('_')
    cart.charge = int(charge)
    st = cart.getStructure()
    return st

def main(fname, output_path):
    st = read_xyz(fname)
#    st = to_structure('CCCCNC=O')
#    st.generate3dConformation()

    # Maybe FF optimize 90% of the time?
    # Let's run 3M of these, 2M from ani-2X and 1M from heavy atom stuff
    # is there a way to get more extreme deprotonations?
    neg_st_list = []
    for charge in range(-4, 0):
        ff_min = random.random() < 0.85
        try:
            lst = get_tautomers(
                st,
                charge=charge,
                use_epikx=True,
                use_fast_pka=False,
                add_stereoisomers=True,
                do_FF_minimization=ff_min,
                suppress_outfiles=True,
                logfile="null",
            )
        except NoTautomersError:
            continue
        neg_st_list.extend(lst)
    pos_st_list = []
    for charge in range(1,5):
        ff_min = random.random() < 0.85
        try:
            lst = get_tautomers(
                st,
                charge=charge,
                use_epikx=True,
                use_fast_pka=False,
                add_stereoisomers=True,
                do_FF_minimization=ff_min,
                suppress_outfiles=True,
                logfile="null",
            )
        except NoTautomersError:
            continue
        pos_st_list.extend(lst)

    ff_min = random.random() < 0.85
    taut_lst = get_tautomers(
        st,
        charge=0,
        use_epikx=True,
        use_fast_pka=False,
        add_stereoisomers=True,
        do_FF_minimization=ff_min,
        suppress_outfiles=True,
        logfile="null",
    )
    if len(taut_lst) > 1:
        neut_st_list = [new_st for new_st in taut_lst if not are_conformers(new_st, st)]
        neut_st_list = uniquify_with_comparison(neut_st_list, are_conformers, use_stereo=True)
    else:
        neut_st_list = []

    neg_st_list = uniquify_with_comparison(neg_st_list, are_conformers, use_stereo=True)
    pos_st_list = uniquify_with_comparison(pos_st_list, are_conformers, use_stereo=True)
    addl_sts = []
    deprot_patt = [
        "[NX3!H0!-,N+X4!H1!H0]",
        "[PX3!H0!-,P+X4!H1!H0]",
        "[AsX3!H0!-,As+X4!H1!H0]",
        "[SbX3!H0!-,Sb+X4!H1!H0]",
        "[SeX2!H0!-]",
        "[TeX2!H0!-]",
    ]
    prot_patt = ["[PX3]", "[AsX3]", "[SbX3]"]
    for patt in deprot_patt:
        addl_sts.extend(deprotonate_sts(neg_st_list + neut_st_list + [st], patt))
    for patt in prot_patt:
        addl_sts.extend(protonate_sts(pos_st_list + neut_st_list + [st], patt))
    if random.random() < 0.15:
        addl_sts.append(deprotonate_carbons(st))
    new_st_list = neg_st_list + pos_st_list + neut_st_list + addl_sts
    new_st_list = uniquify_with_comparison(new_st_list, are_conformers, use_stereo=True)
    print('n_st', len(new_st_list))
    for idx, st in enumerate(new_st_list):
        print(os.path.join(output_path, f'{get_fname_prefix(fname)}_{idx}_{st.formal_charge}_1.mae'))
        st.write(os.path.join(output_path, f'{get_fname_prefix(fname)}_{idx}_{st.formal_charge}_1.mae'))

def get_fname_prefix(old_fname):
    old_fname = os.path.splitext(os.path.basename(old_fname))[0]
    *parts, charge, spin = old_fname.split("_")
    new_fname = "_".join(parts)
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
    parser.add_argument(
        "--n_chunks", type=int, help="How many chunks to split into", default=1
    )
    parser.add_argument(
        "--chunk_idx", type=int, help="which chunk to run", default=0
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(31)
    with open(args.source, "r") as fh:
        fname_list = [f.strip() for f in fh.readlines()]
    random.shuffle(fname_list)
    fname_list = fname_list[: args.n_structs]
    chunks_to_process = np.array_split(fname_list, args.n_chunks)
    chunk = chunks_to_process[args.chunk_idx]
    with tempfile.TemporaryDirectory() as temp_dir:
        with chdir(temp_dir):
            for fname in tqdm(chunk):
                main(fname, args.output_path)
