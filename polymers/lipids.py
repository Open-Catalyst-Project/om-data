"""
For each PDB:

Preprocessing:
1) determine charge by grouping residues together and adding up temperature_factor and rounding
2) extract a representative residue of each type
3) determine Lewis structure
4) copy Lewis structure back to each residue

Grabbing:
1) Take within 3A of a given res.num and chain (and not water)
2) fillres if water
3) fillres if number of atoms 
3) capping of phoshpate with OMe like we did for nucleic acids



"""
import argparse
import glob
import multiprocessing as mp
import os
import random
from collections import Counter, defaultdict

import numpy as np
from schrodinger.application.jaguar.utils import mmjag_update_lewis
from schrodinger.structure import StructureReader
from schrodinger.structutils.analyze import evaluate_asl
from tqdm import tqdm

from subsample_md_boxes import extract_cluster, get_cut_ends


def get_residue_charges(st):
    """
    Determine the total charge on all residues in the system.

    """
    charges = {}

    for mol in st.molecule:
        res = next(iter(mol.residue))
        if res.pdbres in charges:
            continue
        charges[res.pdbres] = round(sum(at.temperature_factor for at in mol.atom))

    return charges


def determine_lewis_structure(st, charges):
    """
    Extract one complete molecule of each
    """
    assigned_molecules = {}
    for mol in st.molecule:
        res = next(iter(mol.residue))
        if res.pdbres in assigned_molecules:
            continue
        mol_st = mol.extractStructure()
        mol_st.property["i_m_Molecular_charge"] = charges[res.pdbres]
        mmjag_update_lewis(mol_st)
        assigned_molecules[res.pdbres] = mol_st
    return assigned_molecules


def copy_bonding(mol_st, mol):
    at_dict = dict(zip(mol_st.atom, mol.atom))
    for at in mol_st.atom:
        at_dict[at].formal_charge = at.formal_charge
    for bond in mol_st.bond:
        mapped_bond = mol.structure.getBond(at_dict[bond.atom1], at_dict[bond.atom2])
        mapped_bond.order = bond.order


def assign_lewis_structure(st):
    charges = get_residue_charges(st)
    assigned_mols = determine_lewis_structure(st, charges)
    for mol in st.molecule:
        res = next(iter(mol.residue))
        repr_mol = assigned_mols[res.pdbres]
        copy_bonding(repr_mol, mol)


def append_phosphate_O(st, at_list):
    P_atoms = evaluate_asl(
        st, f'atom.num {",".join([str(i) for i in at_list])} and at.ele P'
    )
    for at in P_atoms:
        for b_at in st.atom[at].bonded_atoms:
            if b_at.index not in at_list:
                at_list.append(b_at.index)
                at_list.sort()

def get_chain_dups(st):
    chain_dups = defaultdict(dict)
    for mol in st.molecule:
        # Skip waters and ions
        if len(mol.atom) <= 3:
            continue
        res = next(iter(mol.residue))
        if res.chain in chain_dups[res.pdbres]:
            continue
        for res in mol.residue:
            chain_dups[res.pdbres][res.chain] = Counter([at.element for at in res.atom])

    # We do this in two loops so we don't do work for every molecule in the system
    dup_dict = defaultdict(lambda: defaultdict(list))
    for res_name, chain_dict in chain_dups.items():
        for chain_name, counter in chain_dict.items():
            dup_dict[res_name][frozenset(counter.items())].append(chain_name)
    dups_by_res = {res_name:list(counter_dict.values()) for res_name, counter_dict in dup_dict.items()}
    dups_by_res_chain = defaultdict(dict)
    for res_name, partitioning in dups_by_res.items():
        for chain_name in partitioning:
            dups_by_res_chain[res_name][chain_name] = len(partitioning)

def weighted_random_shuffle(items, weights):
    """
    Shuffle items so that higher-weighted items are more likely to appear earlier.
    Each item appears exactly once.
    """


def get_cluster_centers(st):
    res_dict = defaultdict(list)
    for mol in st.molecule:
        # Skip waters and ions
        if len(mol.atom) <= 3:
            continue
        for res in mol.residue:
            res_dict[res.pdbres].append((res.resnum, res.chain))
    for val in res_dict.values():
        random.shuffle(val)
    return res_dict


def add_missing_double_connectors(st, at_list):
    at_set = set(at_list)
    missing_bonded_atoms = Counter()
    for at in at_set:
        bonded_atoms = set(b_at.index for b_at in st.atom[at].bonded_atoms)
        missing_bonded_atoms.update(bonded_atoms - at_set)

    for key, val in missing_bonded_atoms.items():
        if val > 1:
            at_list.append(key)


def get_cluster(st, center_res, max_atoms):

    for size in (10, 5, 4):
        raw_cluster_ats = evaluate_asl(
            st,
            f"(fillres within 3 (res.num {center_res[0]} and chain. {center_res[1]} and not water)) and within {size} (res.num {center_res[0]} and chain {center_res[1]} and not water)",
        )
        append_phosphate_O(st, raw_cluster_ats)
        add_missing_double_connectors(st, raw_cluster_ats)
        cut_ends = get_cut_ends(st, raw_cluster_ats)
        if len(raw_cluster_ats) + len(cut_ends) < max_atoms:
            break
    else:
        return
    cluster = extract_cluster(st, raw_cluster_ats)
    return cluster


def center_iterator(data):
    """
    Iterate through centers.

    We cycle through the keys and we don't go on to the next key until
    we find at least one working value.
    """
    keys = list(data.keys())
    indices = {k: 0 for k in keys}  # indices for each key
    exhausted_keys = set()
    key_idx = 0  # which key we are on

    while len(exhausted_keys) < len(keys):
        curr_key = keys[key_idx]
        if curr_key in exhausted_keys:
            key_idx = (key_idx + 1) % len(keys)
            continue

        idx = indices[curr_key]
        action = yield data[curr_key][idx]

        # Increment indices and heck if we now exhausted that key
        indices[curr_key] += 1
        if indices[curr_key] >= len(data[curr_key]):
            exhausted_keys.add(curr_key)

        if action != "try_again":
            key_idx = (key_idx + 1) % len(keys)
    yield None  # Sentinel to indicate completion


def main(fname):
    fname = "3_0.0ps.pdb"
    #    st = StructureReader.read(fname)
    #    print('assigning lew')
    #    assign_lewis_structure(st)
    #    print('getting cluster')
    st = StructureReader.read("assigned.mae")
    cluster_centers = get_cluster_centers(st)
    counter = 0
    n_structs = 10
    print("beginning loop")
    gen = center_iterator(cluster_centers)
    center = next(gen)
    # Try centers for a given species until it works
    while center is not None and counter < n_structs:
        cluster = get_cluster(st, center, 300)
        if cluster is None:
            print(center)
            center = gen.send("try_again")
        else:
            counter += 1
            new_fname = (
                os.path.splitext(fname)[0]
                + f"_{center[0]}_{center[1]}_{cluster.formal_charge}_1.mae"
            )
            cluster.write(new_fname)
            center = gen.send("success")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_chains_dir", default=".")
    parser.add_argument("--csv_dir", default=".")
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    random.seed(29)
    args = parse_args()
    pdb_list = glob.glob(os.path.join(args.input_dir, "*/*.pdb"))
    chunks_to_process = np.array_split(pdb_list, args.n_chunks)
    chunk = chunks_to_process[args.chunk_idx]
    with mp.Pool(args.n_workers) as pool:
        list(tqdm(pool.imap(main, chunk), total=len(chunk)))
