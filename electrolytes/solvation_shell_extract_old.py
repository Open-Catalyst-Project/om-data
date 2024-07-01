import os
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import List
from collections import defaultdict
import matplotlib.pyplot as plt
import MDAnalysis as mda
from solvation_analysis.solute import Solute
from solvation_analysis._column_names import FRAME, SOLUTE_IX
from pymatgen.core.structure import Molecule
from solvation_shell_utils import (
    filter_by_rmsd,
    wrap_positions,
    extract_charges_from_lammps_file,
)
from utils import parse_list_of_lists


def extract_solvation_shells(
    input_dir: str,
    save_dir: str,
    system_name: str,
    solute_atoms: List[str],
    solvent_atoms: List[List[str]],
    min_coord: int,
    max_coord: int,
    top_n: int,
):
    """
    Given a MD trajectory in a PDB file, perform a solvation analysis
    on the specified solute to extract the first solvation shell. For each coordination number in the specified range,
    extract and save the top_n most diverse snapshots based on a RMSD criterion.

    Args:
        input_dir: Path to 1) the PDB file containing the MD trajectory and 2) the LAMMPS file containing the initial structure (used to extract partial charges).
        save_dir: Directory in which to save extracted solvation shells.
        system_name: Name of the system - used for naming the save directory.
        solute_atoms: List of names (in the PDB file) of the solute atom types (e.g NA0) with which to perform the solvation analysis.
        solvent_atoms: List of names (in the PDB file) of the solvent atom types with which to perform the solvation analysis.
        min_coord: Minimum coordination number to consider.
        max_coord: Maximum coordination number to consider.
        top_n: Number of snapshots to extract per coordination number.
    """
    solute_atoms_name = " ".join(solute_atoms)
    # Create save directory
    os.makedirs(os.path.join(save_dir, system_name, solute_atoms_name), exist_ok=True)

    # Read charges from LAMMPS file
    lammps_file_path = os.path.join(input_dir, "system.data")
    charges = extract_charges_from_lammps_file(lammps_file_path)

    # Initialize MDA Universe
    pdb_file_path = os.path.join(input_dir, "system_output.pdb")
    universe = mda.Universe(pdb_file_path)

    # Add PBC box
    with open(pdb_file_path) as file:
        dimension_lines = file.readlines()[1]
        a = float(dimension_lines.split()[1])
        b = float(dimension_lines.split()[2])
        c = float(dimension_lines.split()[3])
        universe.dimensions = [a, b, c, 90, 90, 90]

    lattices = np.array([a, b, c])[None][None]

    # Choose solute atom
    # TODO: try to join multiple atoms in the same solute, doesn't currently work
    solute_query = " or ".join(f"name {atom}" for atom in solute_atoms)
    solute = universe.select_atoms(solute_query)
    # other_query = " or ".join(f"name {atom}" for atom in other_solute_atoms)
    # other_solu = universe.select_atoms(other_query)

    logging.info("Translating atoms to solute center of mass")
    for ts in tqdm(universe.trajectory):
        ts.dimensions = universe.dimensions
        solu_center = solute.center_of_mass(wrap=True)
        dim = ts.triclinic_dimensions
        box_center = np.sum(dim, axis=0) / 2
        universe.atoms.translate(box_center - solu_center)

        universe.atoms.unwrap()

    # TODO: need to remove the other components of the solute from the solvent atoms? Doesn't matter since the other solute atom won't be close?

    # This doesn't handle multiple solvents
    solvent = (
        universe.atoms - solute
    )  # TODO: might need to count anything other than the solute of interest as a potential solvent
    # TODO: set an upper cutoff for solute-solvent interactions

    solute_name = "".join(atom for atom in solute_atoms)

    logging.info("Running solvation analysis")
    # Identify the cutoff for the first solvation shell, based on the MD trajectory

    # discovered that the reason the analysis was failing was because the default cutoff region of 6.5 was too small for the Cl04 anion
    kernel_kwargs = {"cutoff_region": (1.5, 8)}
    solv_anal = Solute.from_atoms(
        solute,
        {"solvent": solvent},
        solute_name=solute_name,
        kernel_kwargs=kernel_kwargs,
    )
    # TODO: subsample fewer solutes to speed up the analysis

    solv_anal.run()  # which solute's radii is this using?

    max_radii = 0
    max_name = ""
    for solute_name in solv_anal.atom_solutes.keys():
        radii = solv_anal.atom_solutes[solute_name].radii["solvent"]
        if radii > max_radii:
            max_radii = radii
            max_name = solute_name

    # TODO: what do we do with this max or mean radii? Can we need to use it to re-run the analysis?

    # except AssertionError:
    #     logging.info("Failed to automatically find the radius, trying again with a heuristic radius")
    #     heuristic_radii = 2.5 # TODO: add heuristic radii based on effective sizes of solute and solvent
    #     solv_anal = Solute.from_atoms(solu, {"solvent": solvent}, solute_name = solute_name, radii={'solvent': heuristic_radii})
    #     solv_anal.run()

    # Plot the RDF
    solv_anal.plot_solvation_radius(max_name, "solvent")
    plt.savefig(
        os.path.join(save_dir, system_name, solute_atoms_name, "solvation_rdf.png")
    )

    # There's probably a much faster way to do this
    # But for now, we're prototyping, so slow is okay

    # Store the solvation shells with different numbers of solvents (sorted from highest to lowest frequency)
    shells = dict()
    for j in solv_anal.speciation.speciation_fraction["solvent"]:
        shells[j] = solv_anal.speciation.get_shells({"solvent": j})

    # Now let's try getting the most diverse structures for each particular coordination number
    # This is also a bit slow, particularly for the more common and/or larger solvent shells

    # TODO: instead of specifying a range of coordination numbers, we could specify the top n most common coordination numbers (but we also probably want to extract rare configurations)
    for c in range(min_coord, max_coord + 1):
        if c in shells.keys():
            logging.info(
                f"Processing {len(shells[c])} shells with coordination number {c}"
            )
            os.makedirs(
                os.path.join(save_dir, system_name, solute_atoms_name, f"coord={c}"),
                exist_ok=True,
            )
            shell_species = []
            shell_positions = []
            # Extract the shell positions

            for i, (index, _) in tqdm(enumerate(shells[c].iterrows())):
                universe.atoms.unwrap()
                frame, solute_idx = index
                shell = solv_anal.solvation_data.xs(
                    (frame, solute_idx), level=(FRAME, SOLUTE_IX)
                )
                shell = solv_anal._df_to_atom_group(shell, solute_index=index[1])
                # TODO: assertion that compositions/element orderings are the same
                if len(shell.atoms.elements) > len(shell_species):
                    shell_species = shell.atoms.elements
                shell_positions.append(wrap_positions(shell.atoms.positions, lattices))
                if i > 1000:  # TODO: don't hardcode this
                    break

            by_num_atoms = defaultdict(list)
            for sps in shell_positions:
                by_num_atoms[len(sps)].append(sps)

            # filter by number of atoms per shell
            selections_by_num_atoms = {
                num_atoms: filter_by_rmsd(shells_with_num_atoms, top_n)
                for num_atoms, shells_with_num_atoms in by_num_atoms.items()
            }

            for (
                shell_size,
                shell_positions,
            ) in selections_by_num_atoms.items():  # loop over sizes
                for idx, shell_pos in enumerate(shell_positions):
                    if shell_pos.shape[0] == shell_species.shape[0]:

                        # Save shell as xyz file
                        mol = Molecule(
                            shell_species, shell_pos, charge=round(sum(charges))
                        )
                        mol.to(
                            os.path.join(
                                save_dir,
                                system_name,
                                solute_atoms_name,
                                f"coord={c}",
                                f"size{shell_size}_selection{idx}.xyz",
                            ),
                            "xyz",
                        )


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
        "--solute_atoms",
        type=parse_list_of_lists,
        help="Which solute atom(s) to extract solvation shells for (accepts a list of lists, where each list is separated by a semicolon: e.g., 'a,b,c;d,e,f')",
    )

    parser.add_argument(
        "--solvent_atoms",
        type=parse_list_of_lists,
        help="Which solvent atom(s) to extract solvation shells for (accepts a list of lists, where each list is separated by a semicolon: e.g., 'a,b,c;d,e,f')",
    )

    parser.add_argument(
        "--min_coord", type=int, help="Minimum shell coordination number to extract"
    )
    parser.add_argument(
        "--max_coord", type=int, help="Maximum shell coordination number to extract"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of most diverse shells to extract per coordination number",
    )

    args = parser.parse_args()

    extract_solvation_shells(
        args.input_dir,
        args.save_dir,
        args.system_name,
        args.solute_atoms,
        args.solvent_atoms,
        args.min_coord,
        args.max_coord,
        args.top_n,
    )
