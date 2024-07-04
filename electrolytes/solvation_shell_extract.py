from typing import List
import logging

logging.basicConfig(level=logging.INFO)

import random
import os
from tqdm import tqdm
import json
import argparse
import numpy as np

from schrodinger.structure import StructureReader
from schrodinger.structutils import analyze
from solvation_shell_utils import (
    expand_shell,
    renumber_molecules_to_match,
    filter_by_rmsd,
    filter_shells_with_solute_atoms,
    generate_lognormal_samples,
)
from utils import validate_metadata_file
from schrodinger.comparison import are_conformers
from schrodinger.application.jaguar.utils import group_with_comparison
from schrodinger.application.matsci import clusterstruct
from schrodinger.structutils import rmsd


def extract_solvation_shells(
    input_dir: str,
    save_dir: str,
    system_name: str,
    solute_radii: List[float],
    skip_solvent_centered_shells: bool,
    solvent_radii: List[float],
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
        solute_radii: List of shell radii to extract around solutes.
        skip_solvent_centered_shells: Skip extracting solvent-centered shells.
        solvent_radii: List of shell radii to extract around solvents.
        max_shell_size: Maximum size (in atoms) of saved shells.
        top_n: Number of snapshots to extract per topology.
    """

    # Read a structure and metadata file
    # TODO: add charges
    logging.info("Reading structure and metadata files")

    # Read metadata
    with open(os.path.join(input_dir, "metadata_system.json")) as f:
        metadata = json.load(f)

    validate_metadata_file(metadata)

    partial_charges = np.array(metadata["partial_charges"])

    solutes = {}
    solvents = {}
    for res, species, type in zip(
        metadata["residue"], metadata["species"], metadata["solute_or_solvent"]
    ):
        if type == "solute":
            solutes[species] = res
        elif type == "solvent":
            solvents[species] = res
    solute_resnames = set(solutes.values())
    solvent_resnames = set(solvents.values())

    # Read a structure
    structures = StructureReader(os.path.join(input_dir, "system_output.pdb"))

    # assign partial charges to atoms
    # TODO: fix this - the iterator over structures can only be called once
    # logging.info("Assigning partial charges to atoms")
    # for st in tqdm(structures):
    #     for at, charge in zip(st.atom, partial_charges):
    #         at.partial_charge = charge

    # For each solute: extract shells around the solute of some heuristic radii and bin by composition/graph hash
    # Choose the N most diverse in each bin
    for species, residue in solutes.items():
        logging.info(f"Extracting solvation shells around {species}")
        for radius in solute_radii:
            logging.info(f"Radius = {radius} A")
            expanded_shells = []
            for i, st in tqdm(enumerate(structures)):  # loop over timesteps

                if i > 100:  # TODO: fix this
                    break

                # extract all solute molecules
                solute_molecules = [
                    res for res in st.residue if res.pdbres.strip() == residue
                ]
                central_solute_nums = [mol.molecule_number for mol in solute_molecules]
                # Extract solvation shells
                shells = [
                    set(
                        analyze.evaluate_asl(
                            st, f"fillres within {radius} mol {mol_num}"
                        )
                    )
                    for mol_num in central_solute_nums
                ]

                # Now expand the shells
                for shell_ats, central_solute in zip(shells, central_solute_nums):
                    expanded_shell_ats = shell_ats
                    # If we have a solvent-free system, don't expand shells around solutes,
                    # because we'll always have solutes and will never terminate
                    if solvent_resnames:
                        # TODO: how to choose the mean/scale for sampling?
                        # Should the mean be set to the number of atoms in the non-expanded shell?
                        upper_bound = max(
                            len(shell_ats), generate_lognormal_samples()[0]
                        )
                        upper_bound = min(upper_bound, max_shell_size)
                        expanded_shell_ats = expand_shell(
                            st,
                            shell_ats,
                            central_solute,
                            radius,
                            solute_resnames,
                            max_shell_size=upper_bound,
                        )
                    expanded_shell_ats = sorted(expanded_shell_ats)
                    expanded_shell = st.extract(expanded_shell_ats, copy_props=True)

                    # find index of first atom of the central solute in the sorted shell_ats (adjust for 1-indexing)
                    central_solute_atom_idx = (
                        expanded_shell_ats.index(
                            st.molecule[central_solute].getAtomList()[0]
                        )
                        + 1
                    )

                    # contract everthing to be centered on our molecule of interest
                    # (this will also handle if a molecule is split across a PBC)
                    clusterstruct.contract_structure2(
                        expanded_shell, contract_on_atoms=[central_solute_atom_idx]
                    )
                    assert (
                        expanded_shell.atom_total <= max_shell_size
                    ), "Expanded shell too large"
                    expanded_shells.append(expanded_shell)

            # Now compare the expanded shells and group them by similarity
            # we will get lists of lists of shells where each list of structures are conformers of each other
            logging.info("Grouping solvation shells into conformers")
            # TODO: speed this up
            grouped_shells = group_with_comparison(expanded_shells, are_conformers)

            # Now ensure that topologically related atoms are equivalently numbered (up to molecular symmetry)
            grouped_shells = [
                renumber_molecules_to_match(items) for items in grouped_shells
            ]

            # Now extract the top N most diverse shells from each group
            logging.info(f"Extracting top {top_n} most diverse shells from each group")
            final_shells = []
            # example grouping - set of structures
            for shell_group in tqdm(grouped_shells):
                filtered = filter_by_rmsd(shell_group, n=top_n)
                final_shells.extend(filtered)

            # Save the final shells
            logging.info(f"Saving final shells")
            save_path = os.path.join(save_dir, system_name, species, f"radius_{radius}")
            os.makedirs(save_path, exist_ok=True)
            for i, st in enumerate(final_shells):
                # TODO: seems like this is saving an extra line at the end of the xyz files
                st.write(os.path.join(save_path, f"shell_{i}.xyz"))

    if not skip_solvent_centered_shells:
        # Now repeat for solvents to capture solvent-solvent interactions
        for species, residue in solvents.items():
            logging.info(f"Extracting shells around {species}")
            for radius in solvent_radii:
                logging.info(f"Radius = {radius} A")
                filtered_shells = []
                for i, st in tqdm(enumerate(structures)):  # loop over timesteps
                    # assign partial charges to atoms
                    for at, charge in zip(st.atom, partial_charges):
                        at.partial_charge = charge

                    if i > 100:  # TODO: fix this
                        break

                    # extract all solvent molecules
                    solvent_molecules = [
                        res for res in st.residue if res.pdbres.strip() == residue
                    ]
                    central_solvent_nums = [
                        mol.molecule_number for mol in solvent_molecules
                    ]

                    # Extract solvation shells
                    shells = [
                        set(
                            analyze.evaluate_asl(
                                st, f"fillres within {radius} mol {mol_num}"
                            )
                        )
                        for mol_num in central_solvent_nums
                    ]

                    # Only keep the shells that have no solute atoms and below a maximum size
                    solute_free_shells = filter_shells_with_solute_atoms(
                        shells, st, solute_resnames
                    )
                    size_filtered_shells = [
                        shell
                        for shell in solute_free_shells
                        if shell.atom_total <= max_shell_size
                    ]
                    filtered_shells.extend(size_filtered_shells)

                # Choose a random subset of shells
                assert (
                    len(filtered_shells) > 0
                ), "No solute-free shells found for solvent"
                # check that every shell is below the max size
                for shell in filtered_shells:
                    assert (
                        shell.atom_total <= max_shell_size
                    ), "Expanded shell too large"
                random.shuffle(filtered_shells)
                filtered_shells = filtered_shells[:1000]

                # Now compare the expanded shells and group them by similarity
                # we will get lists of lists of shells where each list of structures are conformers of each other
                logging.info("Grouping solvation shells into conformers")
                # TODO: speed this up
                grouped_shells = group_with_comparison(filtered_shells, are_conformers)

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
                for shell_group in tqdm(grouped_shells):
                    filtered = filter_by_rmsd(shell_group, n=top_n)
                    final_shells.extend(filtered)

                # Save the final shells
                logging.info(f"Saving final shells")
                save_path = os.path.join(
                    save_dir, system_name, species, f"radius_{radius}"
                )
                os.makedirs(save_path, exist_ok=True)
                for i, st in enumerate(final_shells):
                    # TODO: seems like this is saving an extra line at the end of the xyz files
                    st.write(os.path.join(save_path, f"shell_{i}.xyz"))


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
        "--solute_radii",
        type=list,
        default=[3],
        help="List of shell radii to extract around solutes",
    )

    parser.add_argument(
        "--skip_solvent_centered_shells",
        action="store_true",
        help="Skip extracting solvent-centered shells",
    )

    parser.add_argument(
        "--solvent_radii",
        type=list,
        default=[3],
        help="List of shell radii to extract around solvents",
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
        args.solute_radii,
        args.skip_solvent_centered_shells,
        args.solvent_radii,
        args.max_shell_size,
        args.top_n,
    )
