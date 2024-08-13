import argparse
import glob
import os

from schrodinger.application.jaguar.utils import mmjag_update_lewis
from schrodinger.structure import Structure, StructureReader
from schrodinger.structutils import build, measure
from schrodinger.structutils.analyze import evaluate_asl, evaluate_smarts
from tqdm import tqdm


def remove_extra_Hs(st: Structure) -> bool:
    ats_to_delete = []
    for pair in measure.get_close_atoms(st, dist=0.5):
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


def rename_coord_chain(st):
    chain_names = [ch.name for ch in st.chain]
    if "l" not in chain_names and "c" in chain_names:
        lig = []
        if len(st.chain["c"].getAtomList()) == 1:
            lig = st.chain["c"].getAtomList()
        if not lig:
            lig = evaluate_asl(st, "chain c and metals")
        if not lig:
            lig = evaluate_asl(st, "chain c and res D8U")
        if not lig:
            lig = evaluate_asl(st, "chain c and atom.ato >=18")
        if not lig:
            lig = evaluate_asl(st, "chain c and atom.ele F,Cl,Br,I")
        for at in lig:
            st.atom[at].chain = "l"
        if lig:
            return True


def deprotonate_metal_bound_n(st):
    N_to_dep = evaluate_asl(
        st,
        "atom.ele N and atom.att 4 and (withinbonds 1 metals) and (withinbonds 1 atom.ele H)",
    )
    ats_to_del = []
    for at_N in N_to_dep:
        att_H = next(
            b_at for b_at in st.atom[at_N].bonded_atoms if b_at.atomic_number == 1
        )
        ats_to_del.append(att_H)
        st.atom[at_N].formal_charge = -1
    st.deleteAtoms(ats_to_del)
    return bool(N_to_dep)


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


def fix_heme_charge(st):
    change_made = False
    for res in st.chain["l"].residue:
        if res.pdbres.strip() in {"HEM", "HEC"}:
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
    for at_N in quart_N:
        if all(bond.order == 1 for bond in st.atom[at_N[0]].bond):
            st.atom[at_N[0]].formal_charge = 1
    return bool(quart_N)


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
            if at1.chain == main_chain:
                old_mol = at2.molecule_number
            else:
                old_mol = at1.molecule_number
            old_chain = next(chain for chain in chains if chain != main_chain)
            for at in st.molecule[old_mol].atom:
                if at.chain == old_chain:
                    at.chain = main_chain


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
    at_pairs = measure.get_close_atoms(st, dist=0.5)
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
            at_pairs = measure.get_close_atoms(st, dist=0.5)
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
        change_made = True
        b_C = broken_C.pop()
        for b_N in broken_N:
            if st.measure(b_C, b_N) < 1.5 and not st.areBound(b_C, b_N):
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
    broken_cysteines = evaluate_asl(st, "atom.pt SG and atom.att 1")
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
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--nuclear_option", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    file_list = glob.glob(os.path.join(args.output_path, f"{args.prefix}*.pdb"))
    file_list = file_list[10000 * args.batch : 10000 * (args.batch + 1)]
    for fname in tqdm(file_list):
        if not os.path.exists(fname):
            continue
        st = StructureReader.read(fname)
        change_made = rename_coord_chain(st)
        change_made = fix_disrupted_disulfides(st) or change_made
        change_made = deprotonate_carboxylic_acids(st) or change_made
        change_made = remove_ligand_ace_cap(st) or change_made
        change_made = remove_ligand_nma_cap(st) or change_made
        change_made = remove_extra_Hs(st) or change_made
        change_made = meld_ace_nma(st) or change_made
        change_made = reconnect_open_chains(st) or change_made
        change_made = remove_total_overlaps(st) or change_made
        change_made = fix_quartenary_N_charge(st) or change_made
        if args.nuclear_option:
            change_made = deprotonate_metal_bound_n(st) or change_made
            change_made = fix_heme_charge(st) or change_made
            change_made = non_physical_estate(st) or change_made
        if change_made:
            new_fname = os.path.join(
                os.path.dirname(fname),
                "_".join(os.path.basename(fname).split("_")[:-1]),
            )
            new_fname += f"_{st.formal_charge}.pdb"
            print(f"{fname} -> {new_fname}")
            st = build.reorder_protein_atoms_by_sequence(st)
            os.remove(fname)
            st.write(new_fname)


if __name__ == "__main__":
    main()
