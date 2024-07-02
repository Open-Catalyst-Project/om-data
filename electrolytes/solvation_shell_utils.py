import re
import numpy as np
import random
from typing import List, Set, Callable

from schrodinger.structutils import analyze
from schrodinger.application.matsci import clusterstruct
from schrodinger.comparison import are_conformers
from schrodinger.application.jaguar.utils import group_with_comparison
from schrodinger.comparison.atom_mapper import ConnectivityAtomMapper
from schrodinger.structutils import rmsd
from schrodinger.structure import Structure


def filter_shells_with_solute_atoms(
    shells: List[Set[int]], st: Structure, solute_res_names: List[str]
) -> List[Structure]:
    """
    Filter out shells that contain solute atoms.
    Args:
        shells: List of sets of atom indices that correspond to shells
        st: Entire structure from the PDB file
        solute_res_names: List of residue names that correspond to solute atoms in the simulation
    Returns:
        List of Schrodinger structures that do not contain solute atoms
    """
    actual_shells = [st.extract(shell, copy_props=True) for shell in shells]
    filtered_shells = [
        shell
        for shell in actual_shells
        if not any(
            at.getResidue().pdbres.strip() in solute_res_names for at in shell.atom
        )
    ]
    return filtered_shells


def expand_shell(
    st: Structure,
    shell_ats: Set[int],
    central_solute: int,
    radius: float,
    solute_res_names: List[str],
    max_shell_size: int = 200,
) -> Structure:
    """
    Expands a solvation shell. If there are any (non-central) solutes present in the shell,
    recursively include shells around those solutes.
    First, gets the molecule numbers of solute molecules that are within the radius
    and not already expanded around. Then, continuously expand around them as long as we don't hit an atom limit.
    Args:
        st: Entire structure from the PDB file
        shell_ats: Set of atom indices (of `st`) in a shell (1-indexed)
        central_solute: Molecule index (of 'st') of the central solute in the shell
        radius: Solvation radius (Angstroms) to consider
        solute_res_names: List of residue names that correspond to solute atoms in the simulation
        max_shell_size: Maximum size (in atoms) of the expanded shell
    Returns:
        Structure object containing the expanded solvation shell
    """
    # # TODO: I'm not sure why we have to do this, but sometimes the central solute is not in the shell
    # shell_ats.add(st.molecule[central_solute].getAtomList())
    solutes_included = set([central_solute])

    def get_new_solutes(st, shell_ats, solutes_included, solute_res_names):
        new_solutes = set()
        for at in shell_ats:
            # If atom is part of a non-central solute molecule - should expand the shell
            if (
                st.atom[at].molecule_number not in solutes_included
                and st.atom[at].getResidue().pdbres.strip() in solute_res_names
            ):
                new_solutes.add(st.atom[at].molecule_number)
        return new_solutes

    new_solutes = get_new_solutes(st, shell_ats, solutes_included, solute_res_names)
    while new_solutes:
        # add entire residues within solvation shell radius of any extra solute atoms
        new_shell_ats = shell_ats.union(
            analyze.evaluate_asl(
                st,
                f'fillres within {radius} mol {",".join([str(i) for i in new_solutes])}',
            )
        )
        if len(new_shell_ats) <= max_shell_size:
            shell_ats = new_shell_ats
            solutes_included.update(new_solutes)
            new_solutes = get_new_solutes(
                st, shell_ats, solutes_included, solute_res_names
            )
        else:
            break

    shell_ats = sorted(shell_ats)
    final_structure = st.extract(shell_ats, copy_props=True)

    # find index of first atom of the central solute in the sorted shell_ats (adjust for 1-indexing)
    central_solute_atom_idx = (
        shell_ats.index(st.molecule[central_solute].getAtomList()[0]) + 1
    )

    # contract everthing to be centered on our molecule of interest
    # (this will also handle if a molecule is split across a PBC)
    clusterstruct.contract_structure2(
        final_structure, contract_on_atoms=[central_solute_atom_idx]
    )
    return final_structure


def filter_by_rmsd(shells: List[Structure], n: int = 20) -> List[Structure]:
    """
    From a set of shell coordinates, determine the n most diverse shells, where "most diverse" means "most different, in terms of minimum RMSD.
    Note: The Max-Min Diversity Problem (MMDP) is in general NP-hard. This algorithm generates a candidate solution to MMDP for these coords
    by assuming that the random seed point is actually in the MMDP set (which there's no reason a priori to assume). As a result, if we ran
    this function multiple times, we would get different results.

    Args:
        shell: List of Schrodinger structure objects containing solvation shells
        n: number of most diverse shells to return
    Returns:
        List of n Schrodinger structures that are the most diverse in terms of minimum RMSD
    """

    seed_point = random.randint(0, len(shells) - 1)
    final_shell_idxs = {seed_point}
    min_rmsds = np.array([rmsd_wrapper(shells[seed_point], shell) for shell in shells])
    for _ in range(n - 1):
        best = np.argmax(min_rmsds)
        min_rmsds = np.minimum(
            min_rmsds,
            np.array([rmsd_wrapper(shells[best], shell) for shell in shells]),
        )
        final_shell_idxs.add(best)
    return [shells[i] for i in final_shell_idxs]


def rmsd_wrapper(st1: Structure, st2: Structure) -> float:
    """
    Wrapper around Schrodinger's RMSD calculation function.
    """
    assert (
        st1.atom_total == st2.atom_total
    ), "Structures must have the same number of atoms for RMSD calculation"
    if st1 == st2:
        return 0.0
    at_list = list(range(1, st1.atom_total + 1))
    return rmsd.superimpose(st1, at_list, st2.copy(), at_list, use_symmetry=True)


def renumber_molecules_to_match(mol_list):
    """
    Ensure that topologically equivalent sites are equivalently numbered
    """
    mapper = ConnectivityAtomMapper(use_chirality=False)
    atlist = range(1, mol_list[0].atom_total + 1)
    renumbered_mols = [mol_list[0]]
    for mol in mol_list[1:]:
        _, r_mol = mapper.reorder_structures(mol_list[0], atlist, mol, atlist)
        renumbered_mols.append(r_mol)
    return renumbered_mols
