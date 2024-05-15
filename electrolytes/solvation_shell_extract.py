import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import MDAnalysis as mda
import nglview as nv
from solvation_analysis.solute import Solute
from solvation_analysis._column_names import *
from pymatgen.core.structure import Molecule
from solvation_shell_utils import filter_by_rmse, wrap_positions

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def extract_solvation_shells(
    pdb_file_path: str,
    save_dir: str,
    system_name: str,
    solute_atom: str,
    min_coord: int,
    max_coord: int,
    top_n: int,
):
    """
    Given a MD trajectory in a PDB file, perform a solvation analysis
    on the specified solute to extract the first solvation shell. For each coordination number in the specified range,
    extract and save the top_n most diverse snapshots based on a RMSD criterion.

    Args:
        pdb_file_path: Path to the PDB file containing the MD trajectory
        save_dir: Directory in which to save extracted solvation shells
        system_name: Name of the system - used for naming the save directory
        solute_atom: Name (in the PDB file) of the solute atom type (e.g NA0) with which to perform the solvation analysis
        min_coord: Minimum coordination number to consider
        max_coord: Maximum coordination number to consider
        top_n: Number of snapshots to extract per coordination number.
    """

    # Create save directory
    os.makedirs(os.path.join(save_dir, system_name, solute_atom), exist_ok=True)

    # Initialize MDA Universe
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
    solu = universe.select_atoms(f"name {solute_atom}")

    logging.info("Translating atoms to solute center of mass")
    for ts in tqdm(universe.trajectory):
        ts.dimensions = universe.dimensions
        solu_center = solu.center_of_mass(wrap=True)
        dim = ts.triclinic_dimensions
        box_center = np.sum(dim, axis=0) / 2
        universe.atoms.translate(box_center - solu_center)

        universe.atoms.unwrap()

    solvent = universe.atoms - solu

    solv_anal = Solute.from_atoms(solu, {"solvent": solvent}, solute_name=solute_atom)

    # Identify the cutoff for the first solvation shell, based on the MD trajectory
    logging.info("Running solvation analysis")
    solv_anal.run()

    # Plot the RDF
    solv_anal.plot_solvation_radius("solute", "solvent")
    plt.savefig(os.path.join(save_dir, system_name, solute_atom, "solvation_rdf.png"))

    # There's probably a much faster way to do this
    # But for now, we're prototyping, so slow is okay
    shells = dict()
    for j in solv_anal.speciation.speciation_fraction["solvent"]:
        shells[j] = solv_anal.speciation.get_shells({"solvent": j})

    # Now let's try getting the most diverse structures for each particular coordination number
    # This is also a bit slow, particularly for the more common and/or larger solvent shells
    for c in range(min_coord, max_coord + 1):
        logging.info(f"Processing shells with coordination number {c}")
        os.makedirs(
            os.path.join(save_dir, system_name, solute_atom, f"coord={c}"),
            exist_ok=True,
        )
        shell_species = []
        shell_positions = []
        for index, _ in tqdm(shells[c].iterrows()):
            ts = universe.trajectory[index[0]]
            universe.atoms.unwrap()
            shell = solv_anal.solvation_data.xs(
                (index[0], index[1]), level=(FRAME, SOLUTE_IX)
            )
            shell = solv_anal._df_to_atom_group(shell, solute_index=index[1])
            shell = shell.copy()
            if len(shell.atoms.elements) > len(shell_species):
                shell_species = shell.atoms.elements

            shell_positions.append(wrap_positions(shell.atoms.positions, lattices))

        by_num_atoms = defaultdict(list)
        for sps in shell_positions:
            by_num_atoms[len(sps)].append(sps)

        # filter by number of atoms per shell
        selections_by_num_atoms = {
            num_atoms: filter_by_rmse(shells_with_num_atoms, top_n)
            for num_atoms, shells_with_num_atoms in by_num_atoms.items()
        }

        for (
            shell_size,
            shell_positions,
        ) in selections_by_num_atoms.items():  # loop over sizes
            for idx, shell_pos in enumerate(shell_positions):
                if shell_pos.shape[0] == shell_species.shape[0]:

                    # Save shell as xyz file
                    mol = Molecule(shell_species, shell_pos, charge=-1)
                    mol.to(
                        os.path.join(
                            save_dir,
                            system_name,
                            solute_atom,
                            f"coord={c}",
                            f"size{shell_size}_selection{idx}.xyz",
                        ),
                        "xyz",
                    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file_path", type=str, help="PDB trajectory file path")
    parser.add_argument("--save_dir", type=str, help="Path to save xyz files")
    parser.add_argument(
        "--system_name", type=str, help="Name of system used for directory naming"
    )
    parser.add_argument(
        "--solute_atom",
        type=str,
        help="Which solute atom to extract solvation shells for",
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
        args.pdb_file_path,
        args.save_dir,
        args.system_name,
        args.solute_atom,
        args.min_coord,
        args.max_coord,
        args.top_n,
    )
