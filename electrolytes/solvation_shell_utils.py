import copy
import itertools
import numpy as np


def rmsd(a, b):
    """
    Compute the root mean squared distance between two sets of coordinates
    Args:
        a: numpy array of positions, shape [N_atoms, 3]
        b: numpy array of positions, shape [N_atoms, 3]
    """
    return np.sqrt(np.mean(np.linalg.norm(b - a, axis=1)))


def filter_by_rmsd(coords, n=20):
    """
    From a set of coordinates, determine the n most diverse, where "most diverse" means "most different, in terms of RMSD"

    Because of the way that IonSolvR data is organized, should be able to just directly calculate rmsd on numpy coordinate arrays

    Args:
        coords: list of np.ndarrays of atom coordinates. Must all have the same shape ([N_atoms, 3]), and must all reflect the same atom order!
            Note that this latter requirement shouldn't be a problem, specifically when dealing with IonSolvR data.
        n: number of most diverse coordinates to return
    """

    states = {0}
    min_rmsds = np.array([rmsd(coords[0], coords[i]) for i in range(len(coords))])
    for i in range(n - 1):
        best = np.argmax(min_rmsds)
        min_rmsds = np.minimum(
            min_rmsds,
            np.array([rmsd(coords[best], coords[i]) for i in range(len(coords))]),
        )
        states.add(best)

    return [coords[i] for i in states]


def reorient(box_dimensions, coords, nsolute, solute_natoms, solvent_natoms):
    """
    This function is not currently used in the pdb-file based solvation analysis
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
