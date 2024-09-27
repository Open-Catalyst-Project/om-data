import random
from typing import List, Set
from tqdm import tqdm
import logging
from collections import Counter


import numpy as np
from schrodinger.comparison.atom_mapper import ConnectivityAtomMapper
from schrodinger.comparison import are_conformers
from schrodinger.structure import Structure
from schrodinger.structutils import analyze, rmsd
from schrodinger.application.jaguar.utils import group_with_comparison
from schrodinger.application.matsci import clusterstruct


def extract_shells_from_structure(
    st: Structure,
    radius: float,
    residue: str,
    spec_type: str,
    shells_per_frame: int,
    max_shell_size: int,
) -> Structure:
    """
    Extract around a given residue type from a structure by a given radius

    :param st: Structure to extract from
    :param radius: distance (in Angstrom) around residue to expand
                   (initially in the case of solutes)
    :param residue: name of residue to consider
    :param spec_type: type of species being extracted, either 'solute' or 'solvent'
    :param shells_per_frame: number of shells to extract from this structure
    :param max_shell_size: maximum number of atoms in a shell
    :return: extracted shell structure
    """
    # extract all molecules of interest
    molecules = [res for res in st.residue if res.pdbres.strip() == residue]

    # Subsample a random set of k solute molecules
    if shells_per_frame > 0:
        molecules = random.sample(molecules, shells_per_frame)

    central_mol_nums = list({mol.molecule_number for mol in molecules})
    # Extract solvation shells
    shells = [
        set(analyze.evaluate_asl(st, f"fillres within {radius} mol {mol_num}"))
        for mol_num in central_mol_nums
    ]

    if spec_type == "solvent":
        # Only keep the shells that have no solute atoms and below a maximum size
        solute_atoms = st.chain["A"].getAtomList()
        shells = [
            (shell, central_mol)
            for shell, central_mol in zip(shells, central_mol_nums)
            if (not shell.intersection(solute_atoms)) and len(shell) <= max_shell_size
        ]
        extracted_shells = [
            extract_contracted_shell(st, at_list, central_mol)
            for at_list, central_mol in shells
        ]

    elif spec_type == "solute":
        extracted_shells = []
        for shell_ats, central_solute in zip(shells, central_mol_nums):
            # Now expand the shells
            expanded_shell_ats = shell_ats
            # If we have a solvent-free system, don't expand shells around solutes,
            # because we'll always have solutes and will never terminate

            # But if we have a solvent, we can expand the shell
            if "B" in {ch.name for ch in st.chain}:  # solvent is present
                # TODO: how to choose the mean/scale for sampling?
                # Should the mean be set to the number of atoms in the non-expanded shell?
                upper_bound = max(len(shell_ats), generate_lognormal_samples()[0])
                upper_bound = min(upper_bound, max_shell_size)
                expanded_shell_ats = expand_shell(
                    st,
                    shell_ats,
                    central_solute,
                    radius,
                    max_shell_size=upper_bound,
                )
            expanded_shell = extract_contracted_shell(
                st, expanded_shell_ats, central_solute
            )

            assert (
                expanded_shell.atom_total <= max_shell_size
            ), "Expanded shell too large"
            extracted_shells.append(expanded_shell)
    return extracted_shells


def expand_shell(
    st: Structure,
    shell_ats: Set[int],
    central_solute: int,
    radius: float,
    max_shell_size: int = 200,
) -> Set[int]:
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
        max_shell_size: Maximum size (in atoms) of the expanded shell
    Returns:
        Set of atom indices (of `st`) of the expanded shell (1-indexed)
    """
    solutes_included = set([central_solute])

    def get_new_solutes(st, shell_ats, solutes_included):
        new_solutes = set()
        for at in shell_ats:
            # If atom is part of a non-central solute molecule - should expand the shell
            if (
                st.atom[at].molecule_number not in solutes_included
                and st.atom[at].chain == "A"
            ):
                new_solutes.add(st.atom[at].molecule_number)
        return new_solutes

    new_solutes = get_new_solutes(st, shell_ats, solutes_included)
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
            new_solutes = get_new_solutes(st, shell_ats, solutes_included)
        else:
            break

    return shell_ats


def extract_contracted_shell(
    st: Structure, at_list: list[int], central_mol: int
) -> Structure:
    """
    Extract the shell from the structure

    :param st: structure to extract from
    :param at_list: list of atom indices that specify shell
    :param central_mol: index of central molecule around which we should contract
                        with respect to PBC to get a non-PBC valid structure
    :return: extracted shell
    """
    central_at = st.molecule[central_mol].atom[1]
    central_at.property["b_m_central"] = True
    extracted_shell = st.extract(at_list, copy_props=True)
    central_at.property.pop("b_m_central")

    # find index of first atom of the central solute in the sorted shell_ats (adjust for 1-indexing)
    central_atom_idx = next(
        at for at in extracted_shell.atom if at.property.pop("b_m_central", False)
    ).index

    # contract everthing to be centered on our molecule of interest
    # (this will also handle if a molecule is split across a PBC)
    clusterstruct.contract_structure(
        extracted_shell, contract_on_atoms=[central_atom_idx]
    )
    return extracted_shell


def group_shells(shell_list: list[Structure], spec_type: str) -> list[list[Structure]]:
    """
    Partition shells by conformers.

    This checks the topological similarity of the shells to group them. For solvents,
    we don't check this topology explicitly but assume it holds if the molecules are
    at least isomers of each other. Revise if we have solvent mixtures where the
    components are isomers

    :param shell_list: list of structures to be partitioned
    :param spec_type: type of species being grouped, either 'solute' or 'solvent'
    :return: members of `shell_list` grouped by conformers, all members of a given
             sublist are conformers
    """
    # Now compare the expanded shells and group them by similarity
    # we will get lists of lists of shells where each list of structures are conformers of each other
    logging.info("Grouping solvation shells into conformers")
    grouped_shells = group_with_comparison(shell_list, are_isomeric_molecules)
    logging.info("Grouped into isomers")

    if spec_type == "solute":
        new_grouped_shells = []
        for isomer_group in tqdm(grouped_shells):
            new_grouped_shells.extend(groupby_molecules_are_conformers(isomer_group))
        grouped_shells = new_grouped_shells
    return grouped_shells


def get_structure_charge(st: Structure) -> int:
    """
    Get the charge on the structure as the sum of the partial charges
    of the atoms

    :param st: Structure to get charge of
    :return: charge on structure
    """
    charge = sum(at.partial_charge for at in st.atom)
    return round(charge)


def are_isomeric_molecules(st1: Structure, st2: Structure) -> bool:
    """
    Determine if two structures have molecules which are isomers of each other.

    This is stronger than just ensuring that the structures are isomers and should
    be sufficient for cases of just solvents as there are no expected topological
    differences.
    """
    isomers = st1.atom_total == st2.atom_total and st1.mol_total == st2.mol_total
    if isomers:
        isomers = Counter(at.atomic_number for at in st1.atom) == Counter(
            at.atomic_number for at in st2.atom
        )
    if isomers:
        cnt1 = {
            frozenset(Counter(at.atomic_number for at in mol.atom).items())
            for mol in st1.molecule
        }
        cnt2 = {
            frozenset(Counter(at.atomic_number for at in mol.atom).items())
            for mol in st2.molecule
        }
        isomers = cnt1 == cnt2
    return isomers


def groupby_molecules_are_conformers(st_list: list[Structure]) -> list[list[Structure]]:
    """
    Given a list of Structures which are assumed to have isomeric molecules,
    partition the structures by conformers.
    """

    def are_same_group_counts(group1, group2):
        return {count for count, _ in group1} == {count for count, _ in group2}

    def are_groups_conformers(group1, group2):
        matched_groups = 0
        for count1, st1 in group1:
            for count2, st2 in group2:
                if count1 == count2 and are_conformers(st1, st2):
                    matched_groups += 1
                    break
        return matched_groups == len(group1)

    if len(st_list) == 1:
        return [st_list]

    mol_to_st = {}
    # split structures up into lists of molecules
    for st in st_list:
        mol_list = [mol.extractStructure() for mol in st.molecule]
        mol_list.sort(key=lambda x: x.atom_total)
        # group those molecules by conformers
        grouped_mol_list = group_with_comparison(mol_list, are_conformers)
        # represent the structure as the counts of each type of molecule
        # and a representative structure
        st_mols = frozenset((len(grp), grp[0]) for grp in grouped_mol_list)
        mol_to_st[st_mols] = st

    # Group structures by if their counts of molecules are the same
    grouped_by_molecular_counts = group_with_comparison(
        mol_to_st.keys(), are_same_group_counts
    )
    conf_groups = []
    # Group structures by if their molecules (and their counts) are conformers
    for groups in grouped_by_molecular_counts:
        conf_groups.extend(group_with_comparison(groups, are_groups_conformers))
    conf_groups = [[mol_to_st[grp] for grp in cgroup] for cgroup in conf_groups]
    return conf_groups


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


def generate_lognormal_samples(loc=75, sigma=0.45, size=1):
    """
    Generate random samples from a lognormal distribution.

    Parameters:
    - loc: float, mean of the distribution
    - sigma: float, standard deviation of the log of the distribution
    - size: int, number of samples to generate (default is 1000)

    Returns:
    - samples: numpy array, random samples from the lognormal distribution
    """
    samples = np.random.lognormal(mean=np.log(loc), sigma=sigma, size=size)
    return samples
