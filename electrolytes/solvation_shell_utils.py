import copy
import itertools
import numpy as np
from rmsd import kabsch_rmsd


def filter_by_rmse(coords, n=20):
    """
    From a set of coordinates, determine the n most diverse, where "most diverse" means "most different, in terms of minimum RMSD.
    We use the Kabsch Algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm) to align coordinates based on rotation/translation before computing the RMSD.
    Note: The Max-Min Diversity Problem is in general NP-hard. This algorithm generates a candidate solution to MMDP for these coords
    by assuming that the point 0 is actually in the MMDP set (which there's no reason a priori to assume). As a result, if we shuffled the order of coords, we would likely get a different result.

    Args:
        coords: list of np.ndarrays of atom coordinates. Must all have the same shape ([N_atoms, 3]), and must all reflect the same atom order!
            Note that this latter requirement shouldn't be a problem, specifically when dealing with IonSolvR data.
        n: number of most diverse coordinates to return
    """

    states = {0}
    min_rmsds = np.array(
        [kabsch_rmsd(coords[0], coord, translate=True) for coord in coords]
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


def wrap_positions(positions, lattices):
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


def reorient(box_dimensions, coords, nsolute, solute_natoms, solvent_natoms):
    """
    This function is not currently used in the pdb-file based solvation analysis
    TODO: remove once Evan clarifies
    """

    transforms = [
        np.array(x) * box_dimensions
        for x in itertools.product([0, -1, 1], [0, -1, 1], [0, -1, 1])
    ]

    cog_solu = np.mean(coords[:solute_natoms], axis=0)

    n_solvents = int((len(coords) - nsolute * solute_natoms) / solvent_natoms)

    final_box = np.zeros(coords.shape)
    final_box[: nsolute * solute_natoms] = coords[: nsolute * solute_natoms]

    for i in range(n_solvents):
        min_dist = np.inf
        best_coords = np.zeros((solvent_natoms, 3))

        start_index = nsolute * solute_natoms + i * solvent_natoms
        coords_i = coords[start_index : start_index + solvent_natoms]
        assert len(coords_i) == solvent_natoms

        for transform in transforms:
            coords_copy = copy.deepcopy(coords_i)
            for i in range(solvent_natoms):
                coords_copy[i] += transform

            cog_solv = np.mean(coords_copy, axis=0)
            dist = np.linalg.norm(cog_solv - cog_solu)
            if dist < min_dist:
                min_dist = dist
                best_coords = coords_copy
        final_box[start_index : start_index + solvent_natoms] = best_coords
    return final_box
