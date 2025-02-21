"""
Extract Desmond outputs into frames.maegz for easy processing
"""

import argparse
import glob
import multiprocessing as mp
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from schrodinger.application.desmond.constants import CMS_POINTER, TRJ_POINTER
from schrodinger.application.desmond.packages import topo, traj_util
from schrodinger.application.jaguar.utils import (
    get_stoichiometry_string,
    group_with_comparison,
)
from schrodinger.comparison import are_conformers
from schrodinger.structure import Residue, Structure, StructureWriter
from schrodinger.structutils.color import apply_color_scheme
from tqdm import tqdm


def get_res_dict(st: Structure) -> Dict[Tuple[str, int], List[Residue]]:
    """
    Get map from a species to the Residues in the input Structure that
    are that species.

    :param st: Structure to analyze
    :return: Map from (stoichiometry string, charge) to list of residues
    """
    if len(set(res.pdbres.strip() for res in st.residue) - {"T3P"}) > 1:
        relabel_t3p(st)
        return {}
    res_dict = defaultdict(list)
    for res in st.residue:
        res_dict[
            (
                get_stoichiometry_string([at.element for at in res.atom]),
                sum(at.formal_charge for at in res.atom),
            )
        ].append(res)
    split_res_dict = {}
    for key, vals in res_dict.items():
        if key[1] != 0:
            split_res_dict[key] = vals
            continue
        res_map = {res.extractStructure(): res for res in vals}
        grouped_neutral = group_with_comparison(list(res_map), are_conformers)
        if len(grouped_neutral) == 1:
            split_res_dict[key] = vals
            continue
        for i, group in enumerate(grouped_neutral):
            split_res_dict[key + (i,)] = [res_map[st_res] for st_res in group]
    return split_res_dict


def label_residues(res_dict) -> None:
    """
    Assign a label to each residue of a certain type

    This label consists of three capital letters (e.g. AAA, BBB, etc.)
    where the type is determined by the grouping in the input `res_dict`.

    :param res_dict: the residues group by stoichiometry
    """
    for idx, vals in enumerate(res_dict.values()):
        for res in vals:
            res.pdbres = chr(65 + idx) * 3


def label_chains(res_dict: Dict[Tuple[str, int], List[Residue]]) -> None:
    """
    Label chains as either solvent or solute
    """
    for key, vals in res_dict.items():
        if key[1] != 0:
            for res in vals:
                res.chain = "A"
        else:
            for res in vals:
                res.chain = "B"


def find_missing_res_label(st: Structure) -> None:
    """
    Determine label that is missing for waters

    :param st: Structure to analyze
    """
    res_list = {res.pdbres.strip() for res in st.residue}
    for i in range(len(res_list)):
        if chr(65 + i) * 3 not in res_list:
            return chr(65 + i) * 3


def relabel_t3p(st: Structure) -> None:
    """
    Relabel T3P waters as a regular letter label.

    Schrodinger tools discard the water label for T3P waters so we
    need to put it back.

    :param st: Structure to relabel
    """
    missing_label = find_missing_res_label(st)
    for res in st.residue:
        if res.pdbres.strip() == "T3P":
            res.pdbres = missing_label


def generate_frames(dir_name):
    fname = os.path.join(dir_name, "final-out.cms")
    maegz_name = os.path.join(dir_name, "frames.maegz")
    if not os.path.exists(fname) or os.path.exists(maegz_name):
        return
    msys_model, cms_model, tr = traj_util.read_cms_and_traj(fname)
    apply_color_scheme(cms_model.fsys_ct, "element")
    res_dict = get_res_dict(cms_model.fsys_ct)
    label_residues(res_dict)
    label_chains(res_dict)
    fsys_st = cms_model.fsys_ct.copy()
    for prop in (CMS_POINTER, TRJ_POINTER):
        fsys_st.property.pop(prop, None)
    st_list = []
    for fr in tr:
        topo.update_fsys_ct_from_frame_GF(fsys_st, cms_model, fr)
        st_list.append(fsys_st.copy())
    with StructureWriter(maegz_name) as writer:
        writer.extend(st_list)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


def main():
    pool = mp.Pool(60)
    args = parse_args()
    dir_list = glob.glob(os.path.join(args.output_path, "*"))
    list(tqdm(pool.imap(generate_frames, dir_list), total=len(dir_list)))


if __name__ == "__main__":
    main()
