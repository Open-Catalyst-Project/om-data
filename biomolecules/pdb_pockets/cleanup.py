import argparse
import glob
import os

from schrodinger.structure import Structure, StructureReader, StructureWriter
from schrodinger.structutils import analyze, build
from tqdm import tqdm


def fix_disrupted_disulfides(st: Structure) -> bool:
    """
    If a cysteine sulfur is missing a hydrogen, add it.

    This occurs when a cysteine residue which is part of a disulfide
    bond is extracted but its corresponding partner is not, leaving
    a "bare" S.

    :param st: Structure to correct
    :return: If a change was made, return True
    """
    broken_cysteines = analyze.evaluate_asl(st, "atom.pt SG and atom.att 1")
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
    return parser.parse_args()


def main():
    args = parse_args()
    for fname in tqdm(glob.glob(os.path.join(args.output_path, f"{args.prefix}*.pdb"))):
        st = StructureReader.read(fname)
        change_made = fix_disrupted_disulfides(st) or change_made
        try:
            change_made = remove_ligand_ace_cap(st) or change_made
        except:
            print(fname)
            raise
        try:
            change_made = remove_ligand_nma_cap(st) or change_made
        except:
            print(fname)
            raise
        if change_made:
            st = build.reorder_protein_atoms_by_sequence(st)
            st.write(fname)


if __name__ == "__main__":
    main()
