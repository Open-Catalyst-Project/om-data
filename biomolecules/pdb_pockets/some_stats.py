import argparse
import glob
import itertools
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from schrodinger.application.jaguar.autots_bonding import copy_bonding
from schrodinger.application.jaguar.utils import mmjag_update_lewis
from schrodinger.structure import Structure, StructureReader, StructureWriter
from schrodinger.structutils.analyze import (
    evaluate_asl,
    evaluate_smarts,
    has_valid_lewis_structure,
    hydrogens_present,
)
from tqdm import tqdm

from biolip_extraction import get_biolip_db
from cleanup import TM_LIST

# Loop through biolip_df, for each
#   1) is generated
#   2) Number of atoms
#   3) is covalently bound
#   4) is ion
#   5) is transition metal
#   6) has non-standard amino acid
#   7) has non-canonical amino acid
#   8) charge
#   9) has a charged residue
#   10) amino acid counter
#   11) number of chains
#   12) number of amino acids (ACE and NMA count as AA)
#   13) protonatable amines
#   14) deprotonable COOH or phenol
#   15) unpaired spin
#   16) elements


def res_is_cap(res):
    return res.pdbres.strip() in {"ACE", "NMA"}


def is_covalent_ligand(st, zob):
    st_copy = st.copy()
    if not zob:
        for bond in st.bond:
            if bond.order == 0:
                st_copy.deleteBond(*bond.atom)
    for mol in st_copy.molecule:
        mol_chains = {res.chain for res in mol.residue}
        mol_chains -= {"c"}
        if "l" in mol_chains and len(mol_chains) > 1:
            is_covalent = True
            break
    else:
        is_covalent = False
    return is_covalent


def get_row_identifiers(fname):
    row_identifiers = []
    pdb_id, *bs_list = os.path.basename(fname).split("_")[0:-2]
    for site in bs_list:
        if "state" in site or "hemecharge" in site or "atecharge" in site:
            continue
        m = re.search(r"(.+)(BS\d+)", site)
        if m is None:
            print(pdb_id, bs_list, site)
            continue
        row_identifiers.append((pdb_id, m.group(1), m.group(2)))
    return row_identifiers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch", type=int)
    return parser.parse_args()


def lookup_value(result, site_id):
    return result[
        (result["pdb_id"] == site_id[0])
        & (result["receptor_chain"] == site_id[1])
        & (result["binding_site_number_code"] == site_id[2])
    ].iloc[0]["has_binding_info"]


def main(args):
    fnames = sorted(glob.glob(os.path.join(args.output_path, f"{args.prefix}*.mae")))
    if args.batch is not None:
        fnames = fnames[args.batch * 1000 : (args.batch + 1) * 1000]
    biolip_df = get_biolip_db(pklpath=args.output_path)
    output_dict = {
        "n_atoms": [],
        "is_covalently_bound": [],
        "is_coordinated": [],
        "is_ion": [],
        "num_tm": [],
        "has_non_canonical": [],
        "has_non_standard": [],
        "charge": [],
        "has_charged_res": [],
        "aa_counter": [],
        "n_chains": [],
        "n_aas": [],
        "prot_amines": [],
        "deprot_os": [],
        "phenols": [],
        "ligand_is_zwitterion": [],
        "estate_physical": [],
        "biolip_entries": [],
        "ligand_id": [],
        "has_binding_info": [],
        "hydrogens_present": [],
        "good_lewis": [],
        "unpaired_spin": [],
        "elements": [],
    }
    index_labels = []
    for fname in tqdm(fnames):
        st = StructureReader.read(fname)
        index_labels.append(os.path.basename(fname) + "gz")
        n_atoms = st.atom_total
        output_dict["n_atoms"].append(n_atoms)
        is_covalent = is_covalent_ligand(st, zob=False)
        output_dict["is_covalently_bound"].append(is_covalent)
        is_coordinated = is_covalent_ligand(st, zob=True)
        output_dict["is_coordinated"].append(is_coordinated)
        is_ion = len(st.chain["l"].atom) == 1
        output_dict["is_ion"].append(is_ion)
        num_tm = sum(1 for at in st.atom if at.atomic_number in TM_LIST)
        output_dict["num_tm"].append(num_tm)
        has_non_canon = any(
            not res.isStandardResidue() for res in st.residue if res.chain.isupper()
        )
        output_dict["has_non_canonical"].append(has_non_canon)
        has_non_stand = any(
            res.getCode() == "X" or res.getCode().islower()
            for res in st.residue
            if res.chain.isupper() and not res_is_cap(res)
        )
        output_dict["has_non_standard"].append(has_non_stand)
        output_dict["charge"].append(st.formal_charge)
        has_charged_res = any(
            sum(at.formal_charge for at in res.atom) != 0
            for res in st.residue
            if res.chain.isupper()
        )
        output_dict["has_charged_res"].append(has_charged_res)
        aa_counter = Counter(
            res.pdbres.strip() for res in st.residue if res.chain.isupper()
        )
        output_dict["aa_counter"].append(aa_counter)
        n_chains = st.mol_total
        if not is_covalent:
            n_chains -= 1
        output_dict["n_chains"].append(n_chains)
        n_aas = aa_counter.total()
        output_dict["n_aas"].append(n_aas)
        elt_counter = Counter(at.element for at in st.atom)
        output_dict["elements"].append(elt_counter)
        lig_st = st.chain["l"].extractStructure()
        output_dict["ligand_id"].append(next(iter(lig_st.residue)).pdbres.strip())
        prot_amines = bool(evaluate_smarts(lig_st, "[C,N][N+1X4;!H0]"))
        output_dict["prot_amines"].append(prot_amines)
        phenol = bool(evaluate_smarts(lig_st, "[c,n][O+0H]"))
        output_dict["phenols"].append(phenol)
        deprot_os = bool(evaluate_smarts(lig_st, "[C](=[O+0H0])[O-]"))
        output_dict["deprot_os"].append(deprot_os)
        is_zwitter = lig_st.formal_charge == 0 and any(
            at.formal_charge != 0 for at in lig_st.atom
        )
        output_dict["ligand_is_zwitterion"].append(is_zwitter)
        mult = int(os.path.splitext(os.path.basename(fname))[0].split("_")[-1])
        output_dict["unpaired_spin"].append(mult != 1)
        estate_physical = (
            sum(at.atomic_number for at in st.atom) - st.formal_charge
        ) % 2 == (mult - 1) % 2
        output_dict["estate_physical"].append(estate_physical)
        identifiers = get_row_identifiers(fname)
        output_dict["biolip_entries"].append(identifiers)
        output_dict["has_binding_info"].append(
            any(lookup_value(biolip_df, item) for item in identifiers)
        )
        st_copy = st.copy()
        remove_metal_metal_bonds(st)
        st.retype()
        lewis_valid = has_valid_lewis_structure(st)
        if st.formal_charge != st_copy.formal_charge:
            copy_bonding(st_copy, st)
            st.retype()
            # Set this to False since it's basically saying the Lewis structure is invalid at the given charge
            lewis_valid = False
        output_dict["hydrogens_present"].append(hydrogens_present(st))
        output_dict["good_lewis"].append(lewis_valid)
        remove_metal_zob(st)
        st.write(fname.replace(".mae", ".maegz"))
    output_df = pd.DataFrame(output_dict, index=index_labels)
    out_name = f"stats_{args.prefix}"
    if args.batch is not None:
        out_name += f"_{args.batch}"
    out_name += ".pkl"
    #    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #        print(output_df)
    output_df.to_pickle(out_name)


def remove_metal_zob(st):
    metals = evaluate_asl(st, "metals")
    waters = evaluate_asl(st, "water and atom.ele O")
    for at_idx in metals:
        # Skip Be
        if st.atom[at_idx].atomic_number == 4:
            continue
        for bond in list(st.atom[at_idx].bond):
            if bond.otherAtom(at_idx).index in waters:
                continue
            if {at.element for at in bond.atom} in (
                {"C", "Sn"},
                {"S", "Sn"},
                {"C", "Pb"},
                {"S", "Pb"},
            ):
                continue
            if bond.order == 0:
                st.deleteBond(*bond.atom)


def remove_metal_metal_bonds(st):
    metals = evaluate_asl(st, "metals")
    for at1, at2 in itertools.combinations(metals, 2):
        bond = st.getBond(at1, at2)
        if bond is not None:
            st.deleteBond(*bond.atom)


if __name__ == "__main__":
    args = parse_args()
    main(args)
