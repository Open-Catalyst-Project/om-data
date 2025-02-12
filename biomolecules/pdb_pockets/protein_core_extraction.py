import argparse
import glob
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List

from schrodinger.structure import (
    Residue,
    Structure,
    StructureReader,
    get_residues_by_connectivity,
)
from schrodinger.structutils import analyze, build

import biolip_extraction as blp_ext
from cleanup import SPIN_PROP, unpair_spin_for_metals

MAX_ATOMS = 350


def get_buried_residues(st: Structure) -> List[Residue]:
    """
    Extract a list of the residues considered buried in this structure

    We say that any residue with a relative SASA less than 0.2 is buried.

    :param st: structure to obtain buried residues in
    :return: buried residues
    """
    # Taken from JMB, 196(3), 641-656
    sasa_norm = {
        "ALA": 113,
        "ARG": 241,
        "ASN": 158,
        "ASP": 151,
        "CYS": 140,
        "GLN": 189,
        "GLU": 183,
        "GLY": 85,
        "HIS": 194,
        "ILE": 182,
        "LEU": 180,
        "LYS": 211,
        "MET": 204,
        "PHE": 218,
        "PRO": 143,
        "SER": 122,
        "THR": 146,
        "TRP": 259,
        "TYR": 229,
        "VAL": 160,
    }
    sasa_data = analyze.calculate_sasa_by_residue(st)
    buried_res = []
    for sasa, res in zip(sasa_data, get_residues_by_connectivity(st)):
        if res.pdbres.strip() in sasa_norm:
            rel_sasa = sasa / sasa_norm[res.pdbres.strip()]
            if rel_sasa < 0.2:
                buried_res.append(res)
    return buried_res


def sample_buried_residues(st: Structure, n_core_res: int) -> List[Residue]:
    """
    Sample the structure for a number of buried residues.

    :param st: the structure to obtain residues from
    :param n_core_res: number of core residues to extract
    :return: sampled residues from the protein
    """
    buried_res = get_buried_residues(st)
    if buried_res:
        res_list = random.sample(buried_res, min(len(buried_res), n_core_res))
    else:
        res_list = []
    return res_list


def group_residues_by_chain(res_list: List[Residue]) -> Dict[str, List[Residue]]:
    """
    Group residues by chain name

    :param res_list: list of residues to group
    :return: residues grouped by chain name
    """
    grouped_res_list = defaultdict(list)
    for res in res_list:
        grouped_res_list[res.chain].append(res)
    return grouped_res_list


def get_single_gap_residues(st: Structure, res_list: List[Residue]) -> List[Residue]:
    """
    Get residues which are between two residues in the given list

    :param st: Structure to get gaps in
    :param res_list: list of residues to look for gaps between
    :return: list of residues which fill a single gap in `res_list`
    """
    grouped_res_list = group_residues_by_chain(res_list)
    gap_res_list = []
    for chained_res_list in grouped_res_list.values():
        gap_res, _ = blp_ext.get_single_gaps(st, "", chained_res_list, ())
        gap_res_list.extend(gap_res)
    return gap_res_list


def get_core_neighborhood(
    st: Structure, pdb_id: str, n_core_res: int, done_list: List[str]
) -> List[Structure]:
    """
    Extract random protein-core interactions

    :param st: Structure of protein of interest
    :param pdb_id: PDB identifier for this protein
    :param n_core_res: number of core residues to sample
    :param done_list: list of already extracted cores
    :return: List of cores (as neighborhoods of amino acids)
    """
    center_res_list = sample_buried_residues(st, n_core_res)
    cores = []
    for center_res in center_res_list:
        st_copy = st.copy()
        heavy_atom_sidechain = "((not atom.ptype N,CA,C,O and not atom.ele H) or (res. gly and atom.ptype CA))"
        side_chain = analyze.evaluate_asl(
            st_copy, center_res.getAsl() + f"and {heavy_atom_sidechain}"
        )
        rand = random.random()
        if rand < 0.8:
            radius = 4
        else:
            radius = 4.5

        save_name = f"{pdb_id}_core{center_res.chain}{center_res.resnum}{center_res.inscode.strip()}"
        if save_name in done_list:
            continue

        core_ats = analyze.evaluate_asl(
            st_copy,
            f'fillres( (within {radius} atom.num {",".join([str(i) for i in side_chain])}) and {heavy_atom_sidechain}) and not metals',
        )
        res_list = list({st_copy.atom[at].getResidue() for at in core_ats})
        gap_res = get_single_gap_residues(st_copy, res_list)
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

        core_ats_with_gaps = []
        for res in res_list + gap_res:
            core_ats_with_gaps.extend(res.getAtomIndices())
        core = st_copy.extract(core_ats_with_gaps)

        # Exclude simple dimers of amino acids as this kind of thing is covered by SPICE
        if len(core.residue) < 3 or core.atom_total > MAX_ATOMS:
            continue
        try:
            blp_ext.cap_termini(st_copy, core, remove_lig_caps=False)
        except Exception as e:
            print("Error: Cannot cap termini")
            print(e)
            raise
        core = prepwizard_core(core, pdb_id)
        core = build.reorder_protein_atoms_by_sequence(core)
        core.title = f"{pdb_id}_core{center_res.chain}{center_res.resnum}{center_res.inscode.strip()}"
        cores.append(core)
    return cores


def prepwizard_core(core: Structure, pdb_id: str, epik_states: int = 0) -> Structure:
    """
    Use Schrodinger PrepWizard to clean up protein structure

    :param core: Structure to cleanup
    :param pdb_id: identifier of protein this core came from
    :param epik_states: number of protonation/tautomer states
                        to consider for ligands present
    :return: Cleaned up structure
    """
    maename = f"{pdb_id}.maegz"
    outname = f"{pdb_id}_prepped.maegz"
    # Remove any dummy atoms, PrepWizard doesn't like them
    dummy_atoms = [at for at in core.atom if at.atomic_number < 1]
    core.deleteAtoms(dummy_atoms)
    core.write(maename)
    # Run PrepWizard
    try:
        blp_ext.run_prepwizard(
            maename, outname, fill_sidechain=False, epik_states=epik_states
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("PrepWizard took longer than 2 hours, skipping")
    if not os.path.exists(outname):
        raise RuntimeError("PrepWizard failed")

    prepped_core = StructureReader.read(outname)
    prepped_core = build.remove_alternate_positions(prepped_core)
    prepped_core = build.reorder_protein_atoms_by_sequence(prepped_core)

    # Cleanup
    for file_to_del in (maename, outname):
        if os.path.exists(file_to_del):
            os.remove(file_to_del)
    for deldir in glob.glob(f"{pdb_id}-???"):
        shutil.rmtree(deldir)
    sys.stdout.flush()
    return prepped_core


def write_structures(cores: List[Structure], output_path: str) -> None:
    """
    Write structures to an maegz file.

    We use this format to keep various metadata, bonding, etc.
    and convert to xyz after the fact.

    :param cores: structures to write to file
    :param output_path: location to store files
    """
    for core in cores:
        unpair_spin_for_metals(core)
        fname = os.path.join(
            output_path,
            f"{core.title}_{core.formal_charge}_{core.property[SPIN_PROP]}.maegz",
        )
        core.write(fname)


def cleanup(pdb_id: str) -> None:
    """
    Cleanup downloaded files

    :param pdb_id: id of PDB which has been downloaded
    """
    for ext in {".pdb", ".cif"}:
        if os.path.exists(f"{pdb_id}{ext}"):
            os.remove(f"{pdb_id}{ext}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--n_core_res", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4621)
    return parser.parse_args()


def main():
    """
    Randomly sample pdb list/
    Download, compute per residue sasa, randomly sample things below threshold, take things within some radius, cap termini, run PPW. done?
    """
    args = parse_args()
    random.seed(args.seed)
    with open("pdb_list.txt", "r") as fh:
        pdb_list = [line.strip() for line in fh.readlines()][
            args.start_idx : args.end_idx
        ]

    os.chdir(args.output_path)
    done_list = {"_".join(f.split("_")[:-2]) for f in glob.glob("*.maegz")}

    for pdb_id in pdb_list:
        st = blp_ext.download_cif(pdb_id)
        try:
            cores = get_core_neighborhood(st, pdb_id, args.n_core_res, done_list)
        except Exception:
            raise
            continue
        write_structures(cores, args.output_path)
        cleanup(pdb_id)


if __name__ == "__main__":
    main()
