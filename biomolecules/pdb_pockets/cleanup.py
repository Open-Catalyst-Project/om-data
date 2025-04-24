import argparse
import glob
import os
from collections import Counter, defaultdict

from schrodinger.application.jaguar.autots_bonding import copy_bonding
from schrodinger.application.jaguar.utils import mmjag_update_lewis
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

TM_LIST = {*range(21, 31), *range(39, 49), *range(72, 81)}
LN = list(range(58, 71))
SPIN_PROP = "i_m_Spin_multiplicity"


def too_high_Z(st):
    if any(at.atomic_number > 83 for at in st.atom):
        return True
    return False


def has_HF(st):
    return bool(evaluate_smarts(st, "[F][H]"))


def remove_partial_bef(st):
    change_made = False
    bef_res = [res for res in st.residue if res.pdbres.strip() in {"BEF"}]
    grouped_bef = defaultdict(list)
    for res in bef_res:
        grouped_bef[res.resnum].append(res)
    bef_res = [val for val in grouped_bef.values() if len(val) < 4]
    for res in bef_res:
        if "c" in (at_res.chain for at_res in res):
            change_made = True
            ats = [at_res.atom[1] for at_res in res]
            st.deleteAtoms(ats)
    return change_made


def correct_bef(st):
    change_made = False
    bef_res = [
        res
        for res in st.residue
        if res.pdbres.strip() in {"BEF", "MNQ", "DAE", "NMQ", "ONP"}
    ]
    grouped_bef = defaultdict(list)
    for res in bef_res:
        grouped_bef[res.resnum].append(res)
    bef_res = [val for val in grouped_bef.values() if len(val) in {4, 5}]
    for res in bef_res:
        if {at_res.chain for at_res in res}.intersection({"l", "c"}):
            change_made = True
            be_at = []
            f_at = []
            for at_res in res:
                at_res.chain = "l"
                if at_res.atom[1].element == "Be":
                    be_at.append(at_res.atom[1].index)
                    at_res.atom[1].formal_charge = -1
                elif at_res.atom[1].element == "F":
                    at_res.atom[1].formal_charge = 0
                    f_at.append(at_res.atom[1].index)
            for f in f_at:
                st.addBond(be_at[0], f, 1)
            # These BeF3 groups are usually bound to a phosphate.
            # We should add that bond.
            bound_at = [
                at
                for at in measure.get_atoms_close_to_subset(st, be_at, 1.7, False)
                if at not in f_at + be_at
            ]
            if len(bound_at) == 1:
                st.addBond(be_at[0], bound_at[0], 0)

    return change_made


def correct_af3(st):
    change_made = False
    af3_res = [res for res in st.residue if res.pdbres.strip() in {"AF3"}]
    for res in af3_res:
        if res.chain in {"l", "c"}:
            change_made = True
            al_at = []
            f_at = []
            for at_res in res.atom:
                at_res.chain = "l"
                if at_res.element == "Al":
                    al_at.append(at_res.index)
                    at_res.formal_charge = 0
                elif at_res.element == "F":
                    at_res.formal_charge = 0
                    f_at.append(at_res.index)
            for bond in st.atom[al_at[0]].bond:
                bond.order = 1
            bound_at = [
                at
                for at in measure.get_atoms_close_to_subset(st, al_at, 1.7, False)
                if at not in f_at + al_at
            ]
            if len(bound_at) == 1:
                st.addBond(al_at[0], bound_at[0], 0)

    return change_made


def correct_hydrogens_with_mmlewis(st):
    ## Warning has_valid_lewis_structure will actually change the Lewis structure
    st.retype()
    change_made = False
    if not hydrogens_present(st):
        st_copy = st.copy()
        build.add_hydrogens(st)
        change_made = st_copy.atom_total != st.atom_total
    return change_made


def correct_charge_with_mmlewis(st):
    st.retype()
    st_copy = st.copy()
    lewis_valid = has_valid_lewis_structure(st)
    change_made = st.formal_charge != st_copy.formal_charge
    return change_made


def correct_pentavalent_carbonyls(st):
    sites = evaluate_smarts(st, "[CX4]=O")
    change_made = False
    for at1, at2 in sites:
        bond = st.getBond(at1, at2)
        bond.order = 1
        build.add_hydrogens(st, atom_list=[at2])
        change_made = True
    return change_made


def correct_overzealous_guan(st):
    sites = evaluate_smarts(st, "[CX3](=[NH2+])([NH3+]([H])([H])[H])")
    prot_N = [site[2] for site in sites]
    bad_Hs = set()
    for site in sites:
        bad_Hs.update(site[3:])
    if prot_N:
        for at in prot_N:
            st.atom[at].formal_charge = 0
        st.deleteAtoms(list(bad_Hs))
        build.add_hydrogens(st, atom_list=prot_N)
    return bool(prot_N)


def remove_very_long_bonds(st):
    long_bonds = []
    change_made = False
    for bond in st.bond:
        rows = set()
        for at in bond.atom:
            at_num = at.atomic_number
            if at_num in {1, 2}:
                rows.add(1)
            elif at_num <= 10:
                rows.add(2)
            else:
                rows.add(3)
        if rows == {1, 1}:
            long_bonds.append(list(bond.atom))
        elif rows == {1, 2} and bond.length > 1.4:
            long_bonds.append(list(bond.atom))
        elif rows == {2, 2} and bond.length > 1.85:
            long_bonds.append(list(bond.atom))
    ats = set()
    for bond in long_bonds:
        st.deleteBond(*bond)
        change_made = True
        ats.update(bond)
    if ats:
        build.add_hydrogens(st, atom_list=list(ats))
    return change_made


def remove_extra_Hs(st: Structure) -> bool:
    ats_to_delete = []
    for pair in measure.get_close_atoms(st, dist=0.6):
        at1, at2 = sorted(pair, key=lambda x: st.atom[x].atomic_number)
        if (
            st.atom[at1].atomic_number == 1
            and st.atom[at2].atomic_number == 6
            and st.atom[at2].bond_total > 4
            and st.areBound(at1, at2)
        ):
            ats_to_delete.append(at1)
    if ats_to_delete:
        st.deleteAtoms(ats_to_delete)
        return True
    return False


def ion_corrections(st):
    change_made = False
    for at in st.atom:
        if at.element in {"Li", "Na", "K", "Rb", "Cs", "Ag"} and at.formal_charge != 1:
            change_made = True
            at.formal_charge = 1
        elif (
            at.element in {"Mg", "Ca", "Sr", "Zn", "Cd", "Hg"} and at.formal_charge != 2
        ):
            change_made = True
            at.formal_charge = 2
        elif at.element in {"Al", "Eu"} and at.formal_charge != 3:
            change_made = True
            at.formal_charge = 3
        elif at.element in {"Fe"} and at.formal_charge not in {2, 3}:
            change_made = True
            at.formal_charge = 2  # Default to 2
        elif at.element in {"Cu"} and at.formal_charge not in {1, 2}:
            change_made = True
            at.formal_charge = 2  # Default to 2
        elif at.element in {"Os"} and at.formal_charge not in {2, 3}:
            change_made = True
            at.formal_charge = 3  # Default to 3, seems more common than 2
        elif at.element in {"Sn"} and at.formal_charge != 4:
            change_made = True
            at.formal_charge = 4
            for bond in at.bond:
                o_at = bond.otherAtom(at)
                if o_at.element in {"C", "S"}:
                    o_at.formal_charge = 0
                    at.formal_charge -= 1
    return change_made


def assign_name_to_unknown(st):
    def res_is_ala(res):
        ala_names = {"CA", "N", "C", "O", "HN", "HA", "CB", "HB1", "HB2", "HB3"}
        at_names = {at.pdbname.strip() for at in res.atom}
        if ala_names == at_names and res.getAtomByPdbName(" CA ").chirality == "S":
            return True
        else:
            return False

    def res_is_gly(res):
        gly_names = {"CA", "N", "C", "O", "HN", "HA1", "HA2"}
        at_names = {at.pdbname.strip() for at in res.atom}
        if gly_names == at_names:
            return True
        else:
            return False

    change_made = False
    for res in st.residue:
        if res.pdbres.strip() == "UNK":
            if res_is_ala(res):
                res.pdbres = "ALA"
                change_made = True
            elif res_is_gly(res):
                res.pdbres = "GLY"
                change_made = True
    return change_made


def deprotonate_metal_bound_n(st):
    N_to_dep = evaluate_asl(
        st,
        "atom.ele N and atom.att 4 and (withinbonds 1 metals) and (withinbonds 1 atom.ele H)",
    )
    ats_to_del = []
    change_made = False
    if N_to_dep:
        metals = evaluate_asl(st, "metals")
    for at_N in N_to_dep:
        att_H = next(
            b_at for b_at in st.atom[at_N].bonded_atoms if b_at.atomic_number == 1
        )
        att_m = next(
            b_at for b_at in st.atom[at_N].bonded_atoms if b_at.index in metals
        )
        if st.areBound(att_H, att_m) and (
            st.measure(at_N, att_m) > st.measure(att_H, att_m)
        ):
            ats_to_del.append(att_H)
            st.atom[at_N].formal_charge = -1
            change_made = True
    st.deleteAtoms(ats_to_del)
    return change_made


def deprotonate_carbonyls(st):
    carbonyls = evaluate_smarts(st, "[H][C]#[O]")
    ats_to_delete = [carb[0] for carb in carbonyls]
    if ats_to_delete:
        st.deleteAtoms(ats_to_delete)
    return bool(ats_to_delete)


def deprotonate_carboxylic_acids(st: Structure) -> bool:
    """
    At physiological pH, it's a good assumption that any carboxylic acids
    will be deprotonated. In the absence pKa's for ligands, we will make
    this assumption.

    :param st: Structure with carboxylic acid groups that can be deprotonated
    :return: True if structure needed to be deprotonated
    """
    cooh_smarts = "[C](=[O])([O][H])"
    try:
        matched_ats = evaluate_smarts(st, cooh_smarts)
    except ValueError:
        matched_ats = []
    H_ats = {ats[-1] for ats in matched_ats}
    O_ats = {ats[-2] for ats in matched_ats}
    for O_at in O_ats:
        st.atom[O_at].formal_charge -= 1
    st.deleteAtoms(H_ats)
    return bool(H_ats)


def deprotonate_phosphate_esters(st: Structure) -> bool:
    """
    At physiological pH, it's a good assumption that any phosphate esters
    will be deprotonated. In the absence pKa's for ligands, we will make
    this assumption.

    :param st: Structure with phosphate groups that can be deprotonated
    :return: True if structure needed to be deprotonated
    """
    change_made = False
    phos_smarts = "[*;!#1][*][PX4,SX4](-,=[O])(-,=[O])(-,=[O,S][H])"
    try:
        matched_ats = evaluate_smarts(st, phos_smarts)
    except ValueError:
        matched_ats = []
    H_ats = {ats[-1] for ats in matched_ats}
    O_ats = {ats[-2] for ats in matched_ats}
    if H_ats:
        for O_at in O_ats:
            st.atom[O_at].formal_charge = -1
        st.deleteAtoms(H_ats)
        change_made = True
    return change_made


def zob_noble_gas(st):
    change_made = False
    ng = evaluate_smarts(st, "[#2,#10,#18,#36,#54]")
    for [at] in ng:
        for bond in st.atom[at].bond:
            if bond.order != 0:
                change_made = True
                bond.order = 0
    return change_made


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
            coord_n = [res.getAtomByPdbName(f" N{i} ") for i in "ABCD"]
            if any(at is None for at in coord_n):
                continue
            if sum(at.formal_charge for at in coord_n) != -2:
                for i, at in enumerate(coord_n, 1):
                    at.formal_charge = -1 * (i % 2)
                change_made = True
    return change_made


def fix_quartenary_N_charge(st):
    quart_N = evaluate_smarts(st, "[NX4+0]")
    change_made = False
    for at_N in quart_N:
        if all(bond.order == 1 for bond in st.atom[at_N[0]].bond):
            st.atom[at_N[0]].formal_charge = 1
            change_made = True
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


def merge_chain_names(st, at1, at2):
    chains = [at1.chain, at2.chain]
    if len(set(chains)) != 1:
        main_chains = [chain for chain in chains if chain.isupper()]
        if len(main_chains) == 1:
            main_chain = main_chains.pop()
            old_chain = next(chain for chain in chains if chain != main_chain)
            if old_chain == "c":
                old_bonds = []
                for bond in st.bond:
                    if {bond.atom1.chain, bond.atom2.chain} == {"l", "c"}:
                        old_bonds.append([bond.atom1, bond.atom2, bond.order])
                for b_at1, b_at2, _ in old_bonds:
                    st.deleteBond(b_at1, b_at2)
            if at1.chain == main_chain:
                old_mol = at2.molecule_number
            else:
                old_mol = at1.molecule_number
            for at in st.molecule[old_mol].atom:
                if at.chain == old_chain:
                    at.chain = main_chain
            if old_chain == "c":
                st.addBonds(old_bonds)


def remove_total_overlaps(st):
    ats_to_delete = [
        max(at1, at2) for at1, at2 in measure.get_close_atoms(st, dist=0.05)
    ]
    if ats_to_delete:
        st.deleteAtoms(ats_to_delete)
        return True
    return False


def meld_ace_nma(st):
    change_made = False
    at_pairs = measure.get_close_atoms(st, dist=0.6)
    while at_pairs:
        at1, at2 = at_pairs.pop()
        at1, at2 = (st.atom[at1], st.atom[at2])
        if (
            {at1.atomic_number, at2.atomic_number} == {6}
            and {at1.pdbres.strip(), at2.pdbres.strip()} == {"ACE", "NMA"}
            and {at1.pdbname.strip(), at2.pdbname.strip()} == {"CA", "CH3"}
        ):
            # Renumber chains if necessary
            merge_chain_names(st, at1, at2)

            if at1.pdbname.strip() == "CA":
                keep_at = at1
                del_at = at2
            else:
                keep_at = at2
                del_at = at1
            for at in keep_at.getResidue().getAtomList():
                st.atom[at].resnum = del_at.resnum
                st.atom[at].inscode = del_at.inscode
            for at in at1.getResidue().getAtomList() + at2.getResidue().getAtomList():
                st.atom[at].pdbres = "GLY "
            del_Hs = [b_at for b_at in del_at.bonded_atoms if b_at.atomic_number == 1]
            del_Hs.extend(
                [b_at for b_at in keep_at.bonded_atoms if b_at.atomic_number == 1]
            )
            non_H = next(
                b_at for b_at in del_at.bonded_atoms if b_at.atomic_number != 1
            )
            st.addBond(keep_at, non_H, 1)
            st.deleteAtoms(del_Hs + [del_at])
            build.add_hydrogens(st, atom_list=[keep_at])
            at_pairs = measure.get_close_atoms(st, dist=0.6)
            change_made = True
    return change_made


def reconnect_open_chains(st: Structure):
    broken_C = evaluate_asl(
        st,
        "atom.pt C and atom.att 2 and within 1.5 (atom.pt N and atom.att 2 and not chain l) and not chain l",
    )
    broken_N = evaluate_asl(
        st,
        "atom.pt N and atom.att 2 and within 1.5 (atom.pt C and atom.att 2 and not chain l) and not chain l",
    )
    change_made = False
    while broken_C:
        b_C = broken_C.pop()
        for b_N in broken_N:
            if st.measure(b_C, b_N) < 1.5 and not st.areBound(b_C, b_N):
                change_made = True
                st.addBond(b_C, b_N, 1)
                st.atom[b_C].formal_charge = 0
                st.atom[b_N].formal_charge = 0
                merge_chain_names(st, st.atom[b_C], st.atom[b_N])
                broken_C = evaluate_asl(
                    st,
                    "atom.pt C and atom.att 2 and within 1.5 (atom.pt N and atom.att 2 and not chain l) and not chain l",
                )
                broken_N = evaluate_asl(
                    st,
                    "atom.pt N and atom.att 2 and within 1.5 (atom.pt C and atom.att 2 and not chain l) and not chain l",
                )
                break
    return change_made


def fix_disrupted_disulfides(st: Structure) -> bool:
    """
    If a cysteine sulfur is missing a hydrogen, add it.

    This occurs when a cysteine residue which is part of a disulfide
    bond is extracted but its corresponding partner is not, leaving
    a "bare" S.

    :param st: Structure to correct
    :return: If a change was made, return True
    """
    broken_cysteines = evaluate_asl(st, "atom.pt SG and atom.att 1 and res CYS")
    if broken_cysteines:
        build.add_hydrogens(st, atom_list=broken_cysteines)
        return True
    return False


def remove_ligand_ace_cap(st: Structure) -> bool:
    """
    If there is an ACE added to a ligand (and the ACE is not itself the ligand),
    remove it and recover what is likely a protonated N.

    :param st: Structure to correct
    :return: If a change was made, return True
    """
    ats_to_delete = []
    N_to_protonate = []
    for res in st.chain["l"].residue:
        if res.pdbres.strip() == "ACE" and len(st.chain["l"].getAtomList()) != 6:
            capped_N = next(
                (at for at in res.getCarbonylCarbon().bonded_atoms if at.element == "N")
            )
            ats_to_delete.extend(res.getAtomList())
            N_to_protonate.append(capped_N)
    for at in N_to_protonate:
        st.atom[at].formal_charge += 1
    if N_to_protonate:
        st.deleteAtoms(ats_to_delete)
        build.add_hydrogens(st, atom_list=N_to_protonate)
        return True
    return False


def remove_ligand_nma_cap(st: Structure) -> bool:
    """
    If there is an NMA added to a ligand (and the ligand is not NMA),
    remove it and recover what is likely a deprotonated O.

    :param st: Structure to correct
    :return: If a change was made, return True
    """
    ats_to_delete = []
    for res in st.chain["l"].residue:
        if res.pdbres.strip() == "NMA" and len(res.getAtomList()) == 6:
            n_at = res.getBackboneNitrogen()
            if n_at is None:
                raise RuntimeError
            ats_to_delete.extend(res.getAtomList())
            ats_to_delete.remove(n_at.index)
            n_at.element = "O"
            n_at.formal_charge = -1
            for b_at in n_at.bonded_atoms:
                adj_res = b_at.getResidue()
                if adj_res != res:
                    break
            n_at.resnum = adj_res.resnum
            n_at.inscode = adj_res.inscode
            n_at.pdbres = adj_res.pdbres
            n_at.pdbname = ""
    st.deleteAtoms(ats_to_delete)
    return bool(ats_to_delete)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--nuclear_option", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    file_list = sorted(glob.glob(os.path.join(args.output_path, f"{args.prefix}*.mae")))
    if args.batch is not None:
        file_list = file_list[1000 * args.batch : 1000 * (args.batch + 1)]
    change_counter = Counter()
    for fname in tqdm(file_list):
        if not os.path.exists(fname):
            continue
        try:
            st = StructureReader.read(fname)
        except:
            print("can't open structure:", fname)
            continue
        st.title = os.path.basename(os.path.splitext(fname)[0])
        if "l" not in (ch.name for ch in st.chain):
            print("no ligand:", fname)
            continue
        if fix_disrupted_disulfides(st):
            change_counter["disrupted_disulfide"] += 1
        # change_made |= deprotonate_carboxylic_acids(st)
        # change_made |= deprotonate_phosphate_esters(st)
        if remove_extra_Hs(st):
            change_counter["extra_Hs"] += 1
        if meld_ace_nma(st):
            change_counter["meld_ace_nma"] += 1
        if reconnect_open_chains(st):
            change_counter["reconnect_open"] += 1
        if remove_very_long_bonds(st):
            change_counter["long_bond"] += 1
        if correct_pentavalent_carbonyls(st):
            change_counter["pentavalent_carb"] += 1
        if remove_total_overlaps(st):
            change_counter["total_ovelap"] += 1
        if fix_quartenary_N_charge(st):
            change_counter["quart_N"] += 1
        if ion_corrections(st):
            change_counter["ion_corr"] += 1
        if deprotonate_metal_bound_n(st):
            change_counter["metal_bound_N"] += 1
        if deprotonate_carbonyls(st):
            change_counter["deprot_carb"] += 1
        if assign_name_to_unknown(st):
            change_counter["assign_name"] += 1
        if correct_overzealous_guan(st):
            change_counter["overzeal_guan"] += 1
        if zob_noble_gas(st):
            change_counter["noble_gas"] += 1
        if correct_bef(st):
            change_counter["bef_problem"] += 1
        if remove_partial_bef(st):
            change_counter["bef_problem"] += 1
        if correct_af3(st):
            change_counter["bef_problem"] += 1
        if args.nuclear_option:
            if fix_heme_charge(st):
                change_counter["heme_charge"] += 1
            if non_physical_estate(st):
                change_counter["non_phys_changee"] += 1
        if correct_charge_with_mmlewis(st):
            change_counter["mmlewis_charge"] += 1
        if correct_hydrogens_with_mmlewis(st):
            change_counter["mmlewis_hydrogen"] += 1
        if unpair_spin_for_metals(st):
            change_counter["unpair_spins"] += 1

        new_fname = get_new_name(fname, st)
        st = build.reorder_protein_atoms_by_sequence(st)
        os.remove(fname)
        if too_high_Z(st):
            change_counter["high_z"] += 1
        elif has_HF(st):
            change_counter["has_HF"] += 1
        else:
            st.write(new_fname)
    print(change_counter)


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
    new_fname += f"_{st.formal_charge}_{st.property[SPIN_PROP]}.mae"
    return new_fname


if __name__ == "__main__":
    main()
