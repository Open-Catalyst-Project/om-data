import random
from typing import List, Set
from tqdm import tqdm
import logging
from collections import Counter


import numpy as np
from schrodinger.comparison.atom_mapper import ConnectivityAtomMapper
from schrodinger.comparison import are_conformers
from schrodinger.structure import Residue, Structure
from schrodinger.structutils import analyze, rmsd
from schrodinger.application.jaguar.utils import group_with_comparison
from schrodinger.application.matsci import clusterstruct
from schrodinger.application.jaguar.utils import get_stoichiometry_string

ION_SPIN = {
    "Ag+2": 1,
    "Co+2": 3,
    "Cr+2": 4,
    "Cr+3": 3,
    "Cu+2": 1,
    "Fe+2": 4,
    "Fe+3": 5,
    "Mn+2": 5,
    "Ni+2": 2,
    "Pd+2": 2,
    "Pt+2": 2,
    "Ti+": 3,
    "OV+2": 1,
    "V+2": 3,
    "V+3": 2,
    "C9H18NO": 1,
}

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
            if len(shell) <= max_shell_size and (not shell.intersection(solute_atoms))
        ]
        extracted_shells = [
            extract_contracted_shell(st, at_list, central_mol)
            for at_list, central_mol in shells
        ]

    elif spec_type == "solute":
        extracted_shells = []
        for shell_ats, central_solute in zip(shells, central_mol_nums):
            if len(shell_ats) > max_shell_size:
                continue
            expanded_shell = extract_contracted_shell(
                st, shell_ats, central_solute
            )

            if expanded_shell.atom_total <= max_shell_size:
                extracted_shells.append(expanded_shell)
    return extracted_shells


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

def get_structure_spin(st: Structure) -> int:
    """
    Get the overall spin of the structure by adding up unpaired
    spins for each molecule.

    We use a dictionary for this since things are fairly circumscribed.
    Anything not in the dictionary is assumed to have no unpaired spins.
    """
    unpaired_count = sum(ION_SPIN.get(get_species_from_res(res), 0) for res in st.residue)
    return unpaired_count + 1

def get_species_from_res(res:Residue)->str:
    """
    Get a species name from the Residue object.

    The species name is the stoichiometry string plus any charge.

    :param res: Residue to get name of
    :return: species name
    """
    stoich = get_stoichiometry_string([at.element for at in res.atom])
    charge = sum(at.formal_charge for at in res.atom)
    if stoich == 'C9H18NO' and 'r_ffio_custom_charge' in res.atom[1].property:
        charge = 0
    label = stoich
    if res.chain == 'A' and charge == 0:
        label += '0'
    elif charge > 0:
        label += '+'
        if charge > 1:
            label += f'{charge}'
    elif charge < 0:
        label += f'{charge}'
    return label

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
        cnt1 = Counter(
            frozenset(Counter(at.atomic_number for at in mol.atom).items())
            for mol in st1.molecule
        )
        cnt2 = Counter(
            frozenset(Counter(at.atomic_number for at in mol.atom).items())
            for mol in st2.molecule
        )
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
    if len(shells) > n:
        final_shell_idxs = {seed_point}
        min_rmsds = np.array([rmsd_wrapper(shells[seed_point], shell) for shell in shells])
        for _ in range(n - 1):
            best = np.argmax(min_rmsds)
            min_rmsds = np.minimum(
                min_rmsds,
                np.array([rmsd_wrapper(shells[best], shell) for shell in shells]),
            )
            final_shell_idxs.add(best)
        selected_shells = [shells[i] for i in final_shell_idxs]
    else:
        selected_shells = list(shells)
    return selected_shells


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
