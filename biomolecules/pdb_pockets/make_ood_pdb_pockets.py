import argparse
import json
import os

import pandas as pd
import requests
#import swifter
from rdkit import Chem
from rdkit.Chem.QED import qed

from biolip_extraction import retrieve_ligand_and_env


def get_rdkit_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol


def wrap_qed(x):
    if x is not None and not (x != x):
        qed_val = qed(x)
    else:
        qed_val = None
    return qed_val


def wrap_rot(x):
    tors_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    if x is not None and not (x != x):
        n_rot = len(x.GetSubstructMatches(tors_smarts))
    else:
        n_rot = None
    return n_rot


def get_ccd_info(ccd_id):
    resp = requests.get(
        f"https://data.rcsb.org/rest/v1/core/chemcomp/{ccd_id}", allow_redirects=True
    )
    parsed = json.loads(resp.content.decode())
    try:
        atom_count = parsed["rcsb_chem_comp_info"]["atom_count"]
    except KeyError:
        atom_count = 0
    try:
        smiles = parsed["rcsb_chem_comp_descriptor"]["smilesstereo"]
    except KeyError:
        smiles = ""
    return atom_count, smiles


def make_time_split_df():
    if os.path.exists('time_split.pkl'):
        return pd.read_pickle('time_split.pkl')

    old_df = pd.read_pickle("biolip_df_05_31_2024.pkl")
    new_df = pd.read_pickle("biolip_df_03_13_2025.pkl")
    match_columns = ["pdb_id"]
    diff_df = new_df[
        ~new_df[match_columns]
        .apply(tuple, axis=1)
        .isin(old_df[match_columns].apply(tuple, axis=1))
    ]
    diff_df = diff_df[~diff_df["ligand_id"].isin(["dna", "rna", "peptide"])]
    diff_df[["lig_atom_count", "lig_SMILES"]] = diff_df["ligand_id"].swifter.apply(
        lambda x: pd.Series(get_ccd_info(x))
    )
    diff_df[["rdmol"]] = diff_df["lig_SMILES"].swifter.apply(
        lambda x: pd.Series(get_rdkit_mol(x))
    )
    diff_df[["qed"]] = diff_df["rdmol"].apply(lambda x: pd.Series(wrap_qed(x)))
    diff_df[["num_rotatable"]] = diff_df["rdmol"].swifter.apply(
        lambda x: pd.Series(wrap_rot(x))
    )

    diff_df_rot = diff_df[diff_df["num_rotatable"] > 2.0]
    diff_df_rot = diff_df_rot[diff_df["num_rotatable"] <= 12.0]
    druglike_df = diff_df_rot.sort_values(by="qed", ascending=False)
    druglike_df = druglike_df[druglike_df['qed'] > 0.5]
    return druglike_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--output_path", default=".")
    parser.add_argument(
        "--no_fill_sidechain", dest="fill_sidechain", action="store_false"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    biolip_df = make_time_split_df()
#    biolip_df.to_pickle('time_split.pkl')
    retrieve_ligand_and_env(
        biolip_df,
        start_pdb=args.start_idx,
        end_pdb=args.end_idx,
        output_path=args.output_path,
        fill_sidechain=args.fill_sidechain,
    )

if __name__ == "__main__":
    main()
