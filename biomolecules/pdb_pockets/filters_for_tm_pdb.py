"""
Loop through list of source, get the .maegz version in large_experiments, use the structure from the lmdb
make a copy impose the atom positions, loop over metal bonds and compare bond length, if more than 1.5x the start then raise flag
"""
import json

# from fairchem.core.datasets.ase_datasets import AseDBDataset
# dataset = AseDBDataset(
#    {"src": "/large_experiments/opencatalyst/foundation_models/data/omol/processed/lmdbs/250303/splits/data/pdb_ood_metal"}
# )
#
# dct = {}
# for idx in range(len(dataset)):
#    atoms = dataset.get_atoms(idx)
#    source = atoms.info["source"]
#    dct[source] = atoms.get_positions().tolist()
#
# with open('data.json', 'w') as fh:
#    json.dump(dct, fh)

import os
import re
import numpy as np
from collections import Counter
from schrodinger.structure import StructureReader, create_new_structure
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.structutils.build import delete_bonds
from schrodinger.structutils.measure import get_shortest_distance
from schrodinger.application.jaguar.autots_rmsd import reform_barely_broken_bonds
from schrodinger.application.matsci.nano.xtal import connect_atoms
from tqdm import tqdm
import multiprocessing as mp
from cleanup import TM_LIST

MAX_VALENCIES = { "H": 4, "He": 4, "Li": 8, "Be": 8, "B": 8, "C": 5, "N": 5, "O": 5, "F": 5, "Ne": 8, "Na": 8, "Mg": 8, "Al": 8, "Si": 8, "P": 8, "S": 8, "Cl": 8, "Ar": 8, "K": 8, "Ca": 8, "Sc": 9, "Ti": 9, "V": 9, "Cr": 9, "Mn": 9, "Fe": 9, "Co": 9, "Ni": 9, "Cu": 9, "Zn": 9, "Ga": 9, "Ge": 9, "As": 8, "Se": 8, "Br": 8, "Kr": 8, "Rb": 8, "Sr": 8, "Y": 9, "Zr": 9, "Nb": 9, "Mo": 9, "Tc": 9, "Ru": 9, "Rh": 9, "Pd": 9, "Ag": 9, "Cd": 9, "In": 9, "Sn": 9, "Sb": 9, "Te": 8, "I": 8, "Xe": 8, "Cs": 8, "Ba": 8, "La": 9, "Ce": 9, "Pr": 10, "Nd": 9, "Pm": 9, "Sm": 9, "Eu": 9, "Gd": 9, "Tb": 9, "Dy": 9, "Ho": 9, "Er": 9, "Tm": 9, "Yb": 9, "Lu": 9, "Hf": 9, "Ta": 9, "W": 9, "Re": 9, "Os": 9, "Ir": 9, "Pt": 9, "Au": 9, "Hg": 9, "Tl": 9, "Pb": 9, "Bi": 9, "Po": 9, "At": 8, "Rn": 8, "Fr": 9, "Ra": 9, "Ac": 9, "Th": 9, "Pa": 9, "U": 9, "Np": 9, "Pu": 9, "Am": 9, "Cm": 9, "Bk": 9, "Cf": 9, "Es": 9, "Fm": 9, "Md": 9, "No": 9, "Lr": 9, "Rf": 9, "Db": 9, "Sg": 9, "Bh": 9, "Hs": 9, "Mt": 9, "Ds": 9, "Rg": 9, "Cn": 9, "Nh": 9, "Fl": 9, "Mc": 9, "Lv": 9, "Ts": 1, "Og": 1, "DU": 15, "Lp": 15, "": 15, }


def get_fname(source):
    base = os.path.basename(os.path.dirname(source))
    temp = "300K" if "300K" in source else "400K"
    if "CHEMBL" in source:
        for batch in ("", "_frames_batch2", "_frames_batch3", "_frames_batch4"):
            path = (
                f"/checkpoint/levineds/pdb_fragments{batch}/{temp}/frames/{base}.maegz"
            )
            if os.path.exists(path):
                break
    else:
        path = os.path.join(
            f"/large_experiments/opencatalyst/foundation_models/data/omol/pdb_pockets/md/",
            temp,
            "frames",
            base + ".mae",
        )
    return path


def is_metal_alone(st):
    metals = evaluate_asl(st, "metals")
    for metal in metals:
        if st.atom[metal].bond_total == 0:
            return True
    return False


def bad_ox_state(st):
    metals = evaluate_asl(st, "metals")
    good_ox = True
    if any(
        st.atom[at].element == "Os" and st.atom[at].formal_charge == 0 for at in metals
    ):
        good_ox = False
    return not good_ox


def count_tm(st):
    return sum(1 for at in st.atom if at.atomic_number in TM_LIST)


def system_has_separated(st):
    """
    if molecules are disconnected and those molecules are either containing a metal or have molar mass greater than 20.
    """
    st_copy = st.copy()
    molecule_list = []
    metals = evaluate_asl(st_copy, "metals")
    metal_mols = {st_copy.atom[at].molecule_number for at in metals}
    for mol in st_copy.molecule:
        if mol.number in metal_mols or mol.extractStructure().total_weight > 20:
            molecule_list.append(mol.atom)
    min_dist = []
    for mol_atoms in molecule_list:
        if len(mol_atoms) < st_copy.atom_total:
            at_idxs = [at.index for at in mol_atoms]
            dist, *_ = get_shortest_distance(st_copy, atoms=at_idxs)
            min_dist.append(dist)
    return any(dist > 5.0 for dist in min_dist)


def mo_w_clusters(st):
    """
    These are oxo clusters and they did not have protons assigned to them correctly
    """
    metals = [
        at for at in evaluate_asl(st, "metals") if st.atom[at].element in {"Mo", "W"}
    ]
    return len(metals) > 2


def non_cluster_multimetallics(st):
    metals = set(evaluate_asl(st, "metals"))
    metal_mols = {st.atom[at].molecule_number for at in metals}
    return len(metal_mols) > 1
#    metal_near_metals = False
#    for metal in metals:
#        if set(evaluate_asl(st, f'metals and withinbonds 2 atom.num {metal}')) == metals:
#            metal_near_metals = True
#            break
#    return not metal_near_metals


def parallel_work(data):
    source, (elts, xyzs) = data
    st = StructureReader.read(get_fname(source))
    # st = create_new_structure()
    # for elt, xyz in zip(elts, xyzs):
    #    st.addAtom(elt, *xyz)
    st.title = source
    connect_atoms(st, cov_factor=2.0, max_valencies=MAX_VALENCIES)
    if (
        is_metal_alone(st)
        or bad_ox_state(st)
        or (count_tm(st) > 5)
        or mo_w_clusters(st)
        or non_cluster_multimetallics(st)
        or system_has_separated(st)
    ):
        return source


def main():
    with open("data_pdb_metal_ood.json", "r") as fh:
        data = json.loads(fh.read())
    with mp.Pool(60) as pool:
        lonesome_metals = set(
            tqdm(pool.imap(parallel_work, data.items()), total=len(data.items()))
        )
    lonesome_metals -= {None}
    with open("lonesome.txt", "w") as fh:
        fh.writelines([line + "\n" for line in lonesome_metals])


if __name__ == "__main__":
    main()
