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
)
from utils import validate_metadata_file
from schrodinger.comparison import are_conformers
from schrodinger.application.jaguar.utils import group_with_comparison
from schrodinger.structutils import rmsd


def extract_solvation_shells(
    input_dir: str,
    save_dir: str,
    system_name: str,
    solute_radii: List[float],
    solvent_radii: List[float],
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
        solvent_radii: List of shell radii to extract around solvents.
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

    solutes = {
        species: res
        for res, species, type in zip(
            metadata["residue"], metadata["species"], metadata["solute_or_solvent"]
        )
        if type == "solute"
    }  # {species: residue name} mapping
    solute_resnames = list(solutes.values())

    solvents = {
        species: res
        for res, species, type in zip(
            metadata["residue"], metadata["species"], metadata["solute_or_solvent"]
        )
        if type == "solvent"
    }  # {species: residue name} mapping
    solvent_resnames = list(solvents.values())

    # Read a structure
    structures = StructureReader(os.path.join(input_dir, "system_output.pdb"))

    # For each solute: extract shells around the solute of some heuristic radii and bin by composition/graph hash
    # Choose the N most diverse in each bin

    for species, residue in solutes.items():
        logging.info(f"Extracting solvation shells around {species}")
        for radius in solute_radii:
            logging.info(f"Radius = {radius} A")
            expanded_shells = []
            for i, st in tqdm(enumerate(structures)):  # loop over timesteps
                # assign partial charges to atoms
                for at, charge in zip(st.atom, partial_charges):
                    at.partial_charge = charge

                if i > 10:  # TODO: fix this
                    break

                # extract all solute molecules
                solute_molecules = [
                    res for res in st.residue if res.pdbres.strip() == f"{residue}"
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
                for shell, central_solute in zip(shells, central_solute_nums):
                    expanded_shell = expand_shell(
                        st,
                        shell,
                        central_solute,
                        radius,
                        solute_resnames,
                        max_shell_size=200,
                    )
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
            save_path = os.path.join(save_dir, system_name, species, f"radius={radius}")
            os.makedirs(save_path, exist_ok=True)
            for i, st in enumerate(final_shells):
                # TODO: seems like this is saving an extra line at the end of the xyz files
                st.write(os.path.join(save_path, f"shell_{i}.xyz"))

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

                if i > 10:  # TODO: fix this
                    break

                # extract all solvent molecules
                solvent_molecules = [
                    res for res in st.residue if res.pdbres.strip() == f"{residue}"
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

                # Only keep the shells that have no solute atoms
                filtered_shells.extend(
                    filter_shells_with_solute_atoms(shells, st, solute_resnames)
                )

            # Choose a random subset of shells
            assert len(filtered_shells) > 0, "No solute-free shells found for solvent"
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
            logging.info(f"Extracting top {top_n} most diverse shells from each group")
            final_shells = []
            # example grouping - set of structures
            for shell_group in tqdm(grouped_shells):
                filtered = filter_by_rmsd(shell_group, n=top_n)
                final_shells.extend(filtered)

            # Save the final shells
            logging.info(f"Saving final shells")
            save_path = os.path.join(save_dir, system_name, species, f"radius={radius}")
            os.makedirs(save_path, exist_ok=True)
            for i, st in enumerate(final_shells):
                # TODO: seems like this is saving an extra line at the end of the xyz files
                st.write(os.path.join(save_path, f"shell_{i}.xyz"))


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    random.seed(10)

    parser = argparse.ArgumentParser()
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
        "--solvent_radii",
        type=list,
        default=[3],
        help="List of shell radii to extract around solvents",
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of most diverse shells to extract per topology",
    )

    args = parser.parse_args()

    extract_solvation_shells(
        args.input_dir,
        args.save_dir,
        args.system_name,
        args.solute_radii,
        args.solvent_radii,
        args.top_n,
    )
