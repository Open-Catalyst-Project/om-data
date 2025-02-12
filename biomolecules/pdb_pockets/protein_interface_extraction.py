import argparse
import glob
import os
import random

import dill
from schrodinger.structure import Residue, Structure
from schrodinger.structutils import analyze, build

import biolip_extraction as blp_ext
import protein_core_extraction as prot_core

MAX_ATOMS = 350


def convert_res_idx_to_res(
    st: Structure, res_idx: List[Tuple[int, int]], dill_info
) -> List[Residue]:
    """
    Convert the indexes of residues derived from DIPS-Plus into
    Schrodinger Residue objects.

    :param st: the structure containing the protein
    :param res_idx: the list of residue identifiers from pos_idx
    :param dill_info: the DIPS-Plus database file for this protein
    :return: the extracted list of residues as Residue objects
    """
    res_list = []
    for res_idx, col in res_idxs:
        res_info = dill_info[1 + col].loc[res_idx]
        res = st.findResidue(f'{res_info["chain"]}:{res_info["residue"]}')
        res_list.append(res)
    return res_list


def sample_interface_residues(
    st: Structure, dill_info, n_iface_res: int
) -> List[Residue]:
    """
    Sample the DIPS-Plus database for this entry for a number of
    interface residues.

    :param st: Structure to obtain residues in
    :param dill_info: the DIPS-Plus database file for this protein
    :param n_iface_res: number of interface residues to sample from
                        the database
    :return: sampled residues from the protein
    """
    interface_res = list({(i, col) for col in (0, 1) for i in dill_info[3][:, col]})
    if interface_res:
        res_list = random.sample(interface_res, min(len(interface_res), n_iface_res))
    else:
        res_list = []
    res_list = convert_res_idx_to_res(st, res_list, dill_info)
    return res_list


def get_interface_neighborhood(
    st: Structure, dill_info, pdb_id: str, n_iface_res: int, done_list: List[str]
) -> List[Structure]:
    """
    Extract random interfaces from proteins according to the DIPS-Plus database

    :param st: Structure of protein of interest
    :param dill_info: the DIPS-Plus database file for this protein
    :param pdb_id: PDB identifier for this protein
    :param n_iface_res: number of interface residues to sample from
                        the database
    :param done_list: list of already extracted interfaces
    :return: List of interfaces (as neighborhoods of amino acids)
    """
    center_res_list = sample_interface_residues(st, dill_info, n_iface_res)
    interfaces = []
    heavy_atom_sidechain = (
        "((not atom.ptype N,CA,C,O and not atom.ele H) or (res. gly and atom.ptype CA))"
    )
    for center_res in center_res_list:
        st_copy = st.copy()
        side_chain = analyze.evaluate_asl(
            st_copy, center_res.getAsl() + f"and {heavy_atom_sidechain}"
        )

        save_name = f"{pdb_id}_iface{center_res.chain}{center_res.resnum}{center_res.inscode.strip()}"
        if save_name in done_list:
            continue
        interchain_ats = analyze.evaluate_asl(
            st_copy,
            f'fillres( not chain {center_res.chain} and (within 6 atom.num {",".join([str(i) for i in side_chain])}) and ({heavy_atom_sidechain} or water)) and not metals',
        )
        intrachain_ats = analyze.evaluate_asl(
            st_copy,
            f'fillres( chain {center_res.chain} and (within 4 atom.num {",".join([str(i) for i in side_chain])}) and ({heavy_atom_sidechain} or water)) and not metals',
        )
        iface_ats = interchain_ats + intrachain_ats
        res_list = list({st_copy.atom[at].getResidue() for at in iface_ats})
        gap_res = prot_core.get_single_gap_residues(st_copy, res_list)
        try:
            blp_ext.make_gaps_gly(st_copy, None, gap_res)
        except:
            print("problem found")
            print(
                f"{pdb_id} {center_res.chain} {center_res.resnum}{center_res.inscode}"
            )
            for res in res_list:
                print(f"{pdb_id} {res.chain} {res.resnum}{res.inscode}")
            for res in gap_res:
                print(f"{pdb_id} {res.chain} {res.resnum}{res.inscode}")
                for atom in res.atom:
                    print(atom.pdbname, atom.index)
            st_copy.write(f"{pdb_id}_problem.maegz")
            raise

        iface_ats_with_gaps = []
        for res in res_list + gap_res:
            iface_ats_with_gaps.extend(res.getAtomIndices())
        # Try to label ligands
        for at in analyze.evaluate_asl(st_copy, "ligand"):
            st_copy.atom[at].chain = "l"
        interface = st_copy.extract(iface_ats_with_gaps)

        # Exclude simple dimers of amino acids as this kind of thing is covered by SPICE
        if len(interface.residue) < 3 or interface.atom_total > MAX_ATOMS:
            continue
        try:
            blp_ext.cap_termini(st_copy, interface, remove_lig_caps=True)
        except Exception as e:
            print("Error: Cannot cap termini")
            print(e)
            raise
        interface = prot_core.prepwizard_core(interface, pdb_id, epik_states=1)
        interface = build.reorder_protein_atoms_by_sequence(interface)
        interface.title = save_name
        interfaces.append(interface)
    return interfaces


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--n_iface_res", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4621)
    return parser.parse_args()


def main():
    """
    Take the DIPS-Plus, extract a random residue from the pos_idx, take inter-chain things at 6A, take intra-chain things at 4A
    """
    args = parse_args()
    random.seed(args.seed)
    dips_plus = sorted(glob.glob("/checkpoint/levineds/dips_plus/raw/*/*"))[
        args.start_idx : args.end_idx
    ]

    os.chdir(args.output_path)
    done_list = {"_".join(f.split("_")[:-2]) for f in glob.glob("*.maegz")}

    for dips_file in dips_plus:
        dill_info = dill.load(open(dips_file, "rb"))
        pdb_id, _ = dill_info[0].split(".")
        st = blp_ext.download_cif(pdb_id)
        try:
            interfaces = get_interface_neighborhood(
                st, dill_info, pdb_id, args.n_iface_res, done_list
            )
        except Exception:
            raise
            continue
        prot_core.write_structures(interfaces, args.output_path)
        prot_core.cleanup(pdb_id)


if __name__ == "__main__":
    main()
