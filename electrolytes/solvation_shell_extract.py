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
from schrodinger.structutils.analyze import evaluate_asl

from solvation_shell_utils import (
    extract_shells_from_structure,
    group_shells,
    get_species_from_res,
    get_structure_charge,
    get_structure_spin,
    filter_by_rmsd,
    renumber_molecules_to_match,
)
from utils import validate_metadata_file


def neutralize_tempo(st, tempo_res):
    for at in evaluate_asl(st, f'res {tempo_res}'):
        st.atom[at].formal_charge = 0

def extract_solvation_shells(
    input_dir: str,
    save_dir: str,
    system_name: str,
    radii: List[float],
    skip_solvent_centered_shells: bool,
    max_frames: int,
    last_frame_only: bool,
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
        last_frame_only: Only use the last frame for extraction
        shells_per_frame: Number of solutes or solvents per MD simulation frame from which to extract candidate shells.
        max_shell_size: Maximum size (in atoms) of saved shells.
        top_n: Number of snapshots to extract per topology.
    """

    # Read a structure and metadata file
    logging.info("Reading structure and metadata files")
    # Read structures
    fname = os.path.join(input_dir, "frames.maegz")
    if not os.path.exists(fname):
        return
    structures = list(StructureReader(fname))

    solutes = {}
    solvents = {}
    chains ={ch.name for ch in structures[0].chain}
    if 'A' in chains:
        for res in structures[0].chain['A'].residue:
            species = get_species_from_res(res)
            if species not in solutes:
                solutes[species] = res.pdbres.strip()
    if 'B' in chains:
        for res in structures[0].chain['B'].residue:
            species = get_species_from_res(res)
            if species not in solutes:
                solvents[species] = res.pdbres.strip()

    spec_dicts = {"solute": solutes, "solvent": solvents}

    if last_frame_only:
        structures = structures[-1:]
    if max_frames > 0:
        structures = random.sample(structures, max_frames)
    # assign partial charges to atoms
    if 'C9H18NO' in solutes:
        logging.info("Adjusting chargess")
        for st in tqdm(structures):
            neutralize_tempo(st, solutes['C9H18NO'])

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
                save_path = os.path.join(
                    save_dir, system_name, species, f"radius_{radius}"
                )
                if os.path.exists(save_path):
                    continue
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

                if not extracted_shells:
                    # raise a warning and continue to the next radii/species
                    logging.warning(f"No acceptable shells found for {system_name}, {species}, radius={radius}. Leaving an empty directory so we know we tried.")
                    os.makedirs(save_path, exist_ok=True)
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
                    #filtered = random.sample(shell_group, min(top_n, len(shell_group)))
                    filtered = filter_by_rmsd(shell_group, n=top_n)
                    filtered = [(group_idx, st) for st in filtered]
                    final_shells.extend(filtered)

                # Save the final shells
                logging.info("Saving final shells")
                os.makedirs(save_path, exist_ok=True)
                cur_group_idx = None
                for group_idx, st in final_shells:
                    if group_idx == cur_group_idx:
                        counter += 1
                    else:
                        counter = 0
                    charge = get_structure_charge(st)
                    spin = get_structure_spin(st)
                    fname = os.path.join(
                        save_path, f"group_{group_idx}_shell_{counter}_{charge}_{spin}.mae"
                    )

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
        "--last_frame_only",
        action='store_true',
        help="Only use the last frame of the MD",
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
        args.last_frame_only,
        args.shells_per_frame,
        args.max_shell_size,
        args.top_n,
    )
