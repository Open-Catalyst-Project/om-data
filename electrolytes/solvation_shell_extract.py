import logging
from typing import List

logging.basicConfig(level=logging.INFO)

import argparse
import json
import os
import random
from tqdm import tqdm

import numpy as np

from schrodinger.structure import StructureReader

from solvation_shell_utils import (
    extract_shells_from_structure,
    group_shells,
    get_structure_charge,
    filter_by_rmsd,
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

                    # extract shells from the structure, centered on the residue type
                    extracted_shells.extend(
                        extract_shells_from_structure(
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
