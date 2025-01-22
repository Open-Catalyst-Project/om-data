import argparse
import glob
import multiprocessing as mp
import os
import re
from collections import Counter, defaultdict

from schrodinger.application.jaguar.autots_bonding import copy_bonding
from schrodinger.application.jaguar.packages.shared import uniquify_with_comparison
from schrodinger.application.jaguar.utils import mmjag_update_lewis
from schrodinger.comparison import are_conformers
from schrodinger.infra import mm
from schrodinger.structure import Structure, StructureReader
from schrodinger.structutils import build, measure
from schrodinger.structutils.analyze import (
    evaluate_asl,
    evaluate_smarts,
    has_valid_lewis_structure,
    hydrogens_present,
)
from tqdm import tqdm

from cleanup import LN, SPIN_PROP, TM_LIST


def fix_heme_charge(st):
    change_made = False
    for res in st.chain["l"].residue:
        if res.pdbres.strip() in {
            "HEM", "HEC", "CLA", "BCL", "DHE", "HEA", "HEO", "HDM", "HP5",
            "NTE", "COH", "HNI", "VOV", "6CO", "6CQ", "CCH", "DDH", "HEB",
            "ISW", "HEV", "WUF", "3ZZ", "BW9", "CV0", "ZNH", "522", "76R",
            "HCO", "HFM", "MD9", "ZND", "CLN", "WRK", "WC5", "ZEM", "NTE",
            "89R", "VER", "UFE", "HIF",
        }:
            hs_to_delete = []
            coord_n = [res.getAtomByPdbName(f" N{i} ") for i in "ABCD"]
            if any(at is None for at in coord_n):
                continue
            for at in coord_n:
                hs_to_delete.extend(
                    [h_at for h_at in at.bonded_atoms if h_at.atomic_number == 1]
                )
            if sum(at.formal_charge for at in coord_n) != -2:
                offset = 1 if sum(bond.order for bond in coord_n[0].bond) == 2 else 0
                for i, at in enumerate(coord_n, offset):
                    at.formal_charge = -1 * (i % 2)
                change_made = True
            st.deleteAtoms(hs_to_delete)
    return change_made


def unpair_spin_for_metals(st):
    change_made = False
    st.property.pop(SPIN_PROP, None)
    metals = [at for at in st.atom if at.atomic_number in TM_LIST]
    ln = [at for at in st.atom if at.atomic_number in LN]
    elec_parity = (sum(at.atomic_number for at in st.atom) - st.formal_charge) % 2
    if metals and elec_parity != 0:
        st.property[SPIN_PROP] = 2
        change_made = True
    elif ln and len(ln) == 1:
        spin = 8 - abs(64 - ln[0].atomic_number)
        st.property[SPIN_PROP] = spin
        change_made = True
    else:
        st.property[SPIN_PROP] = 1
    return change_made


def non_physical_estate(st):
    change_made = False
    if (sum(at.atomic_number for at in st.atom) - st.formal_charge) % 2 != 0:
        ats_before = st.atom_total
        build.add_hydrogens(st)
        if ats_before != st.atom_total:
            change_made = True
        else:
            mmjag_update_lewis(st)
            ats_before = st.atom_total
            build.add_hydrogens(st)
            if ats_before != st.atom_total:
                change_made = True
    return change_made


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch", type=int)
    return parser.parse_args()


def do_work(fname_list):
    heme_adjusted = []
    for fname in fname_list:
        heme_fix = False
        st = StructureReader.read(fname)
        if fix_heme_charge(st):
            unpair_spin_for_metals(st)

            new_fname = get_new_name(fname, st)
            st = build.reorder_protein_atoms_by_sequence(st)
            st.title = new_fname
            heme_adjusted.append(st)
    write_out_new_structures(heme_adjusted, fname_list)


def main():
    args = parse_args()
    file_list = sorted(glob.glob(os.path.join(args.output_path, f"{args.prefix}*.mae")))
    file_dict = defaultdict(list)
    for fn in file_list:
        m = re.match(r"(.*)_(.*)_state", os.path.basename(fn))
        file_dict[(m.group(1), m.group(2))].append(fn)
    if args.batch is not None:
        file_dict_keys = list(file_dict)
        file_dict_keys = file_dict_keys[1000 * args.batch : 1000 * (args.batch + 1)]
        file_dict = {k: file_dict[k] for k in file_dict_keys}

    pool = mp.Pool(60)
    file_list = list(file_dict.values())
    list(tqdm(pool.imap(do_work, file_list), total=len(file_list)))


def write_out_new_structures(heme_adjusted, fname_list):
    heme_adjusted = uniquify_with_comparison(
        heme_adjusted, are_conformers, use_lewis_structure=False
    )
    old_sts = [StructureReader.read(fn) for fn in fname_list]
    for st in heme_adjusted:
        if all(
            not are_conformers(st, st2, use_lewis_structure=False) for st2 in old_sts
        ):
            st.write(st.title)


def get_new_name(fname, st):
    # Try to parse as name, charge, spin
    *new_basename, charge, spin = os.path.splitext(os.path.basename(fname))[0].split(
        "_"
    )

    # If the charge is not actually a charge but something like
    # "state0", then it's actually part of the name and "spin"
    # is actually the charge, but we are discarding that anyway
    try:
        int(charge)
    except ValueError:
        new_basename.append(charge)

    new_fname = os.path.join(os.path.dirname(fname), "_".join(new_basename))
    new_fname += f"_hemecharge_{st.formal_charge}_{st.property[SPIN_PROP]}.mae"
    return new_fname


if __name__ == "__main__":
    main()
