import re
import numpy as np
import random
from typing import List
from rmsd import kabsch_rmsd


def filter_by_rmsd(coords: List[np.array], n: int = 20) -> List[np.array]:
    """
    From a set of coordinates, determine the n most diverse, where "most diverse" means "most different, in terms of minimum RMSD.
    We use the Kabsch Algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm) to align coordinates based on rotation/translation before computing the RMSD.
    Note: The Max-Min Diversity Problem is in general NP-hard. This algorithm generates a candidate solution to MMDP for these coords
    by assuming that the random seed point is actually in the MMDP set (which there's no reason a priori to assume). As a result, if we ran this function multiple times, we would get different results.

    Args:
        coords: list of np.arrays of atom coordinates. Must all have the same shape ([N_atoms, 3]), and must all reflect the same atom order!
            Note that this latter requirement shouldn't be a problem, specifically when dealing with IonSolvR data.
        n: number of most diverse coordinates to return
    Returns:
        list of np.arrays of diverse atom coordinates
    """

    seed_point = random.randint(0, len(coords) - 1)
    states = {seed_point}
    min_rmsds = np.array(
        [kabsch_rmsd(coords[seed_point], coord, translate=True) for coord in coords]
    )
    for _ in range(n - 1):
        best = np.argmax(min_rmsds)
        min_rmsds = np.minimum(
            min_rmsds,
            np.array(
                [kabsch_rmsd(coords[best], coord, translate=True) for coord in coords]
            ),
        )
        states.add(best)

    return [coords[i] for i in states]


def wrap_positions(positions: np.array, lattices: np.array) -> np.array:
    """
    Wraps input positions based on periodic boundary conditions.
    Args:
        positions: numpy array of positions, shape [N_atoms, 3]
        lattices: numpy array representing dimensions of simulation box, shape [1, 1, 3]
    Returns:
        numpy array of wrapped_positions, shape [N_atoms, 3]
    """
    displacements = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    idx = np.where(displacements > lattices / 2)[0]
    dim = np.where(displacements > lattices / 2)[2]
    if idx.shape[0] > 0:
        positions[idx, dim] -= lattices[0, 0, dim]
    return positions


def extract_charges_from_lammps_file(file_path: str) -> List[float]:
    """
    Extracts partial charges of each atom from a LAMMPS file.
    Args:
        file_path (str): path to the LAMMPS file
    Returns:
        charges: list of partial charges of length N_atoms
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    atoms_section = False
    charges = []
    first = False
    for line in lines:
        if "Atoms" in line:
            atoms_section = True
            first = True
            continue
        if atoms_section:
            if line.strip() == "":
                if not first:
                    atoms_section = False
                continue
            first = False
            parts = re.split(r"\s+", line.strip())
            # Assuming the format is: atom-ID molecule-ID atom-type charge x y z
            charge = float(parts[3])
            charges.append(charge)

    return charges
