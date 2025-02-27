import argparse
import glob
import itertools
import os
from collections import defaultdict

import pandas as pd
from schrodinger.structure import StructureReader
from schrodinger.structutils import build, measure
from tqdm import tqdm

from biolip_extraction import get_biolip_db


def get_covalently_attached_ligands(biolip_df: pd.DataFrame, path: str, batch):
    """

    :param biolip_df: DataFrame storing BioLiP database
    :param path: Path to extracted pockets
    """
    shared_pockets = biolip_df[
        biolip_df.duplicated(
            subset=["pdb_id", "ligand_chain"],
            keep=False,
        )
    ]
    grouped_pockets = shared_pockets.groupby(
        ["pdb_id", "ligand_chain"],
        group_keys=False,
    )
    file_list = glob.glob(os.path.join(path, "*state0*.mae"))

    grouped_file_list = defaultdict(list)
    for f in file_list:
        start = os.path.basename(f).split("_")[0]
        grouped_file_list[start].append(f)

    combs = []
    if batch is not None:
        grouped_pockets = list(grouped_pockets)[1000 * batch : 1000 * (batch + 1)]
    for key, item in tqdm(grouped_pockets):
        pdb_id = item.iloc[0]["pdb_id"]
        fnames = grouped_file_list[pdb_id]
        combs.extend(determine_combinable_structures(fnames))
    return combs


def determine_combinable_structures(fname_list):
    lig_dict = {}
    for fname in fname_list:
        st = StructureReader.read(fname)
        st.title = fname
        lig_st = st.chain["l"].extractStructure()
        build.delete_hydrogens(lig_st)
        lig_dict[lig_st] = st

    combs = []
    for lig1, lig2 in itertools.combinations(lig_dict, 2):
        bonding_ats = measure.get_atoms_close_to_structure(lig1, lig2, 1.7, False)
        if bonding_ats:
            combs.append((lig_dict[lig1].title, lig_dict[lig2].title))
    return combs


def check_if_coords_overlap(combs):
    reduced_combs = []
    for f1, f2 in tqdm(combs):
        coord_collide = False
        st1 = StructureReader.read(f1)
        st2 = StructureReader.read(f2)
        if "c" in (ch.name for ch in st1.chain):
            coord_st = st1.chain["c"].extractStructure()
            st_copy = st2.copy()
            st_copy.extend(coord_st)
            if measure.get_close_atoms(st_copy, dist=0.06):
                coord_collide = True
        elif "c" in (ch.name for ch in st2.chain):
            coord_st = st2.chain["c"].extractStructure()
            st_copy = st1.copy()
            st_copy.extend(coord_st)
            if measure.get_close_atoms(st_copy, dist=0.06):
                coord_collide = True
        if not coord_collide:
            reduced_combs.append((f1, f2))
    return reduced_combs


def all_state_glob(fname, file_list):
    idx = fname.index("state0")
    glob_str = fname[:idx] + "state"
    for file in file_list:
        if file.startswith(glob_str):
            yield file


def cap_ligand_structures(f1, lig2, file_list):
    for st_name in all_state_glob(f1, file_list):
        st = StructureReader.read(st_name)
        bonding_ats1 = measure.get_atoms_close_to_structure(st, lig2, 1.7, False)
        build.add_hydrogens(st, atom_list=bonding_ats1)
        st.write(st_name)


def replace_missing_lig_with_H(combs, path):
    file_list = glob.glob(os.path.join(path, "*.mae"))
    for f1, f2 in tqdm(combs):
        st1 = StructureReader.read(f1)
        st2 = StructureReader.read(f2)
        lig1 = st1.chain["l"].extractStructure()
        lig2 = st2.chain["l"].extractStructure()
        build.delete_hydrogens(lig1)
        build.delete_hydrogens(lig2)

        cap_ligand_structures(f1, lig2, file_list)
        cap_ligand_structures(f2, lig1, file_list)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".", destination="path")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--read", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.read:
        with open(f"ligands_needing_caps_{args.batch}.txt", "r") as fh:
            combs = eval(fh.read())
    else:
        biolip_df = get_biolip_db(pklpath=args.path)
        combs = get_covalently_attached_ligands(biolip_df, args.path, args.batch)
        combs = check_if_coords_overlap(combs)
        with open(f"ligands_needing_caps_{args.batch}.txt", "w") as fh:
            fh.writelines(str(combs))
    replace_missing_lig_with_H(combs, args.path)


if __name__ == "__main__":
    main()
