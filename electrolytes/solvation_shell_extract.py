import logging
from typing import List

logging.basicConfig(level=logging.INFO)

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
from schrodinger.application.jaguar.utils import group_with_comparison
from schrodinger.application.matsci import clusterstruct
from schrodinger.comparison import are_conformers
from schrodinger.structure import Structure, StructureReader
from schrodinger.structutils import analyze
from tqdm import tqdm

from solvation_shell_utils import (
    expand_shell,
    filter_by_rmsd,
    generate_lognormal_samples,
    renumber_molecules_to_match,
)
from utils import validate_metadata_file


def extract_solvation_shells(
    input_dir: str,
    save_dir: str,
    system_name: str,
    radii: List[float],
    skip_solvent_centered_shells: bool,
    max_frames: int,
    shells_per_frame: int,
    max_shell_size: int,
    top_n: int,
):
    """
    Given a MD trajectory in a PDB file, perform a solvation analysis
    on the specified solute to extract the first solvation shell.

    Args:
        input_dir: Path to 1) the PDB file containing the MD trajectory (system_output.pdb) and 2) a metadata file (system_metadata.json)
        save_dir: Directory in which to save extracted solvation shells.
        system_name: Name of the system - used for naming the save directory.
        radii: List of shell radii to extract around solutes and solvents.
        skip_solvent_centered_shells: Skip extracting solvent-centered shells.
        max_frames: Maximum number of frames to read from the trajectory.
        shells_per_frame: Number of solutes or solvents per MD simulation frame from which to extract candidate shells.
        max_shell_size: Maximum size (in atoms) of saved shells.
        top_n: Number of snapshots to extract per topology.
    """

    # Read a structure and metadata file
    logging.info("Reading structure and metadata files")

    # Read metadata
    with open(os.path.join(input_dir, "metadata_system.json")) as f:
        metadata = json.load(f)

    validate_metadata_file(metadata)

    partial_charges = np.array(metadata["partial_charges"])

    solutes = {}
    solvents = {}
    for res, species, spec_type in zip(
        metadata["residue"], metadata["species"], metadata["solute_or_solvent"]
    ):
        if spec_type == "solute":
            solutes[species] = res
        elif spec_type == "solvent":
            solvents[species] = res
    spec_dicts = {"solute": solutes, "solvent": solvents}

    # Read structures
    structures = list(StructureReader(os.path.join(input_dir, "system_output.pdb")))
    if max_frames > 0:
        structures = random.sample(structures, max_frames)
    # assign partial charges to atoms
    logging.info("Assigning partial charges to atoms")
    for st in tqdm(structures):
        for at, charge in zip(st.atom, partial_charges):
            at.partial_charge = charge

    # For each solute: extract shells around the solute of some heuristic radii and bin by composition/graph hash
    # Choose the N most diverse in each bin
    spec_types = ["solute"]
    if not skip_solvent_centered_shells:
        spec_types.append("solvent")

    for spec_type in spec_types:
        for species, residue in spec_dicts[spec_type].items():
            logging.info(f"Extracting solvation shells around {species}")
            for radius in radii:
                logging.info(f"Radius = {radius} A")
                extracted_shells = []
                for i, st in tqdm(
                    enumerate(structures), total=len(structures)
                ):  # loop over timesteps
                    extracted_shells.extend(
                        extract_residue_from_structure(
                            st,
                            radius,
                            residue,
                            spec_type,
                            shells_per_frame,
                            max_shell_size,
                        )
                    )

                if spec_type == "solvent":
                    # raise a warning and continue to the next radii/species
                    logging.warning("No solute-free shells found for solvent")
                    continue

                grouped_shells = group_shells(extracted_shells, spec_type)

                # Now ensure that topologically related atoms are equivalently numbered (up to molecular symmetry)
                grouped_shells = [
                    renumber_molecules_to_match(items) for items in grouped_shells
                ]

                # Now extract the top N most diverse shells from each group
                logging.info(
                    f"Extracting top {top_n} most diverse shells from each group"
                )
                final_shells = []
                # example grouping - set of structures
                for group_idx, shell_group in tqdm(
                    enumerate(grouped_shells), total=len(grouped_shells)
                ):
                    filtered = filter_by_rmsd(shell_group, n=top_n)
                    filtered = [(group_idx, st) for st in filtered]
                    final_shells.extend(filtered)

                # Save the final shells
                logging.info("Saving final shells")
                save_path = os.path.join(
                    save_dir, system_name, species, f"radius_{radius}"
                )
                os.makedirs(save_path, exist_ok=True)
                for i, (group_idx, st) in enumerate(final_shells):
                    charge = get_structure_charge(st)
                    if spec_type == "solute":
                        fname = os.path.join(
                            save_path, f"group_{group_idx}_shell_{i}_{charge}.xyz"
                        )
                    elif spec_type == "solvent":
                        fname = os.path.join(save_path, f"shell_{i}_{charge}.xyz")

                    st.write(fname)


def extract_residue_from_structure(
    st: Structure,
    radius: float,
    residue: str,
    spec_type: str,
    shells_per_frame: int,
    max_shell_size: int,
) -> Structure:
    """
    Extract around a  given residue type from a structure by a given radius

    :param st: Structure to extract from
    :param radius: distance (in Angstrom) around residue to expand
                   (initially in the case of solutes)
    :param resiude: name of residue to consider
    :param spec_type: type of species being extracted, either 'solute' or 'solvent'
    :param shells_per_frame: number of shells to extract from this structure
    :param max_shell_size: maximum number of atoms in a shell
    :return: extracted shell structure
    """
    # extract all molecules of interest
    molecules = [res for res in st.residue if res.pdbres.strip() == residue]

    # Subsample a random set of k solute molecules
    if shells_per_frame > 0:
        molecules = random.sample(molecules, shells_per_frame)

    central_mol_nums = list({mol.molecule_number for mol in molecules})
    # Extract solvation shells
    shells = [
        set(analyze.evaluate_asl(st, f"fillres within {radius} mol {mol_num}"))
        for mol_num in central_mol_nums
    ]

    if spec_type == "solvent":
        # Only keep the shells that have no solute atoms and below a maximum size
        solute_atoms = st.chain["A"].getAtomList()
        shells = [
            (shell, central_mol)
            for shell, central_mol in zip(shells, central_mol_nums)
            if (not shell.intersection(solute_atoms)) and len(shell) <= max_shell_size
        ]
        extracted_shells = [
            extract_contracted_shell(st, at_list, central_mol)
            for at_list, central_mol in shells
        ]

    elif spec_type == "solute":
        extracted_shells = []
        for shell_ats, central_solute in zip(shells, central_mol_nums):
            # Now expand the shells
            expanded_shell_ats = shell_ats
            # If we have a solvent-free system, don't expand shells around solutes,
            # because we'll always have solutes and will never terminate
            if "B" in {ch.name for ch in st.chain}:
                # TODO: how to choose the mean/scale for sampling?
                # Should the mean be set to the number of atoms in the non-expanded shell?
                upper_bound = max(len(shell_ats), generate_lognormal_samples()[0])
                upper_bound = min(upper_bound, max_shell_size)
                expanded_shell_ats = expand_shell(
                    st,
                    shell_ats,
                    central_solute,
                    radius,
                    max_shell_size=upper_bound,
                )
            expanded_shell = extract_contracted_shell(
                st, expanded_shell_ats, central_solute
            )

            assert (
                expanded_shell.atom_total <= max_shell_size
            ), "Expanded shell too large"
            extracted_shells.append(expanded_shell)
    return extracted_shells


def extract_contracted_shell(
    st: Structure, at_list: list[int], central_mol: int
) -> Structure:
    """
    Extract the shell from the structure

    :param st: structure to extract from
    :param at_list: list of atom indices that specify shell
    :param central_mol: index of central molecule around which we should contract
                        with respect to PBC to get a non-PBC valid structure
    :return: extracted shell
    """
    central_at = st.molecule[central_mol].atom[1]
    central_at.property["b_m_central"] = True
    extracted_shell = st.extract(at_list, copy_props=True)
    central_at.property.pop("b_m_central")

    # find index of first atom of the central solute in the sorted shell_ats (adjust for 1-indexing)
    central_atom_idx = next(
        at for at in extracted_shell.atom if at.property.pop("b_m_central", False)
    ).index

    # contract everthing to be centered on our molecule of interest
    # (this will also handle if a molecule is split across a PBC)
    clusterstruct.contract_structure(
        extracted_shell, contract_on_atoms=[central_atom_idx]
    )
    return extracted_shell


def group_shells(shell_list: list[Structure], spec_type: str) -> list[list[Structure]]:
    """
    Partition shells by conformers.

    This checks the topological similarity of the shells to group them. For solvents,
    we don't check this topology explicitly but assume it holds if the molecules are
    at least isomers of each other. Revise if we have solvent mixtures where the
    components are isomers

    :param shell_list: list of structures to be partitioned
    :param spec_type: type of species being grouped, either 'solute' or 'solvent'
    :return: members of `shell_list` grouped by conformers, all members of a given
             sublist are conformers
    """
    # Now compare the expanded shells and group them by similarity
    # we will get lists of lists of shells where each list of structures are conformers of each other
    logging.info("Grouping solvation shells into conformers")
    grouped_shells = group_with_comparison(shell_list, are_isomeric_molecules)
    logging.info("Grouped into isomers")

    if spec_type == "solute":
        new_grouped_shells = []
        for isomer_group in tqdm(grouped_shells):
            new_grouped_shells.extend(groupby_molecules_are_conformers(isomer_group))
        grouped_shells = new_grouped_shells
    return grouped_shells


def get_structure_charge(st: Structure) -> int:
    """
    Get the charge on the structure as the sum of the partial charges
    of the atoms

    :param st: Structure to get charge of
    :return: charge on structure
    """
    charge = sum(at.partial_charge for at in st.atom)
    return round(charge)


def are_isomeric_molecules(st1: Structure, st2: Structure) -> bool:
    """
    Determine if two structures have molecules which are isomers of each other.

    This is stronger than just ensuring that the structures are isomers and should
    be sufficient for cases of just solvents as there are no expected topological
    differences.
    """
    isomers = st1.atom_total == st2.atom_total and st1.mol_total == st2.mol_total
    if isomers:
        isomers = Counter(at.atomic_number for at in st1.atom) == Counter(
            at.atomic_number for at in st2.atom
        )
    if isomers:
        cnt1 = {
            frozenset(Counter(at.atomic_number for at in mol.atom).items())
            for mol in st1.molecule
        }
        cnt2 = {
            frozenset(Counter(at.atomic_number for at in mol.atom).items())
            for mol in st2.molecule
        }
        isomers = cnt1 == cnt2
    return isomers


def groupby_molecules_are_conformers(st_list: list[Structure]) -> list[list[Structure]]:
    """
    Given a list of Structures which are assumed to have isomeric molecules,
    partition the structures by conformers.
    """

    def are_same_group_counts(group1, group2):
        return {count for count, _ in group1} == {count for count, _ in group2}

    def are_groups_conformers(group1, group2):
        matched_groups = 0
        for count1, st1 in group1:
            for count2, st2 in group2:
                if count1 == count2 and are_conformers(st1, st2):
                    matched_groups += 1
                    break
        return matched_groups == len(group1)

    if len(st_list) == 1:
        return [st_list]

    mol_to_st = {}
    # split structures up into lists of molecules
    for st in st_list:
        mol_list = [mol.extractStructure() for mol in st.molecule]
        mol_list.sort(key=lambda x: x.atom_total)
        # group those molecules by conformers
        grouped_mol_list = group_with_comparison(mol_list, are_conformers)
        # represent the structure as the counts of each type of molecule
        # and a representative structure
        st_mols = frozenset((len(grp), grp[0]) for grp in grouped_mol_list)
        mol_to_st[st_mols] = st

    # Group structures by if their counts of molecules are the same
    grouped_by_molecular_counts = group_with_comparison(
        mol_to_st.keys(), are_same_group_counts
    )
    conf_groups = []
    # Group structures by if their molecules (and their counts) are conformers
    for groups in grouped_by_molecular_counts:
        conf_groups.extend(group_with_comparison(groups, are_groups_conformers))
    conf_groups = [[mol_to_st[grp] for grp in cgroup] for cgroup in conf_groups]
    return conf_groups


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path containing PDB trajectory and LAMMPS data files",
    )
    parser.add_argument("--save_dir", type=str, help="Path to save xyz files")
    parser.add_argument(
        "--system_name", type=str, help="Name of system used for directory naming"
    )

    parser.add_argument(
        "--radii",
        type=list,
        default=[3],
        help="List of shell radii to extract around solutes and solvents",
    )

    parser.add_argument(
        "--skip_solvent_centered_shells",
        action="store_true",
        help="Skip extracting solvent-centered shells",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Number of MD simulation frames from the trajectory to use for extraction",
    )

    parser.add_argument(
        "--shells_per_frame",
        type=int,
        default=-1,
        help="Number of solutes or solvents per MD simulation frame from which to extract candidate shells",
    )

    parser.add_argument(
        "--max_shell_size",
        type=int,
        default=200,
        help="Maximum size (in atoms) of the saved shells",
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of most diverse shells to extract per topology",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    extract_solvation_shells(
        args.input_dir,
        args.save_dir,
        args.system_name,
        args.radii,
        args.skip_solvent_centered_shells,
        args.max_frames,
        args.shells_per_frame,
        args.max_shell_size,
        args.top_n,
    )
