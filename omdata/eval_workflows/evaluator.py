from __future__ import annotations

import numpy as np
from pymatgen.io.ase import MSONAtoms
from schrodinger.application.jaguar.utils import mmjag_reset_connectivity
from schrodinger.application.matsci.aseutils import get_structure
from schrodinger.comparison.atom_mapper import ConnectivityAtomMapper
from schrodinger.structure import Structure
from schrodinger.structutils import rmsd
from scipy.optimize import linear_sum_assignment

boltzmann_constant = 8.617333262 * 10**-5


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


def sdgr_rmsd(atoms1, atoms2):
    """
    Calculate the RMSD between two sets of atoms.
    """
    atoms1 = MSONAtoms.from_dict(atoms1)
    atoms2 = MSONAtoms.from_dict(atoms2)

    st1 = get_structure(atoms1)
    mmjag_reset_connectivity(st1)
    st2 = get_structure(atoms2)
    mmjag_reset_connectivity(st2)
    renumbered_sts = renumber_molecules_to_match([st1, st2])
    return rmsd_wrapper(renumbered_sts[0], renumbered_sts[1])


def cosine_similarity(forces_1, forces_2):
    return np.sum(forces_1 * forces_2) / (
        np.linalg.norm(forces_1) * np.linalg.norm(forces_2)
    )


def interaction_energy_and_forces(results, principal_identifier):
    """
    Calculate the interaction energy and interaction forces between a principal structure and each individual component in the complex.

    Args:
        results (dict): Results from ORCA or MLIP calculations.
        principal_identifier (str): The identifier of the principal structure.

    Returns:
        interaction_energy, interaction_forces
    """
    interaction_energy = {}
    interaction_forces = {}
    for identifier in results.keys():
        indices_found = set()
        interaction_energy[identifier] = results[identifier][principal_identifier][
            "energy"
        ]
        interaction_forces[identifier] = results[identifier][principal_identifier][
            "forces"
        ]
        principal_atoms = results[identifier][principal_identifier]["atoms"]
        for component_identifier in results[identifier].keys():
            if component_identifier != principal_identifier:
                interaction_energy[identifier] -= results[identifier][
                    component_identifier
                ]["energy"]
                for ii, sub_atom in enumerate(
                    results[identifier][component_identifier]["atoms"]
                ):
                    for jj, principal_atom in enumerate(principal_atoms):
                        if sub_atom.symbol == principal_atom.symbol:
                            if (sub_atom.position == principal_atom.position).all():
                                indices_found.add(jj)
                                interaction_forces[identifier][jj] -= np.array(
                                    results[identifier][component_identifier]["forces"]
                                )[ii]
        assert len(indices_found) == len(principal_atoms)

    return interaction_energy, interaction_forces


def distance_scaling_processing(results):
    results = {}


def spin_deltas(results):
    """
    Calculate deltaE and deltaF values for the spin gap evaluation task.

    Args:
        results (dict): Results from ORCA or MLIP calculations performed at different spins.

    Returns:
        deltaE (dict), deltaF (dict)
    """
    deltaE = {}
    deltaF = {}
    for identifier in results.keys():
        deltaE[identifier] = {}
        deltaF[identifier] = {}
        spins = []
        for spin in results[identifier].keys():
            spins.append(int(spin))
        spins.sort(reverse=True)
        for spin in spins[1:]:
            deltaE[identifier][spin] = (
                results[identifier][str(spins[0])]["energy"]
                - results[identifier][str(spin)]["energy"]
            )
            deltaF[identifier][spin] = np.array(
                results[identifier][str(spins[0])]["forces"]
            ) - np.array(results[identifier][str(spin)]["forces"])
    return deltaE, deltaF


def charge_deltas(results):
    """
    Calculate deltaE and deltaF values for adding and removing electrons

    Args:
        results (dict): Results from ORCA or MLIP calculations performed at different charges.

    Returns:
        deltaE (dict), deltaF (dict)
    """
    deltaE = {}
    deltaF = {}
    for identifier in results.keys():
        charges = []
        for charge in results[identifier].keys():
            charges.append(int(charge))
        charges = sorted(charges)
        assert charges[1] - 1 == charges[0]
        assert charges[1] + 1 == charges[2]
        deltaE[identifier] = {"add_electron": {}, "remove_electron": {}}
        deltaF[identifier] = {"add_electron": {}, "remove_electron": {}}
        orig_energy = results[identifier][str(charges[1])]["energy"]
        orig_forces = np.array(results[identifier][str(charges[1])]["forces"])
        for charge_val, tag in [
            (str(charges[0]), "add_electron"),
            (str(charges[2]), "remove_electron"),
        ]:
            for spin in results[identifier][charge_val].keys():
                deltaE[identifier][tag][spin] = (
                    results[identifier][charge_val][spin]["energy"] - orig_energy
                )
                deltaF[identifier][tag][spin] = (
                    np.array(results[identifier][charge_val][spin]["forces"])
                    - orig_forces
                )
    return deltaE, deltaF


def boltzmann_weighted_structures(results, temp=298.15):
    """
    Assign Boltzmann weights to the conformers in each family.

    Args:
        results (dict): Conformer results from ORCA or MLIP calculations.
        temp (float): Temperature in Kelvin.

    Returns:
        dict: Boltzmann weighted structures for each family.
    """
    weighted_families = {}
    for family_identifier, structs in results.items():
        weights = {}
        weighted_structs = {}
        sum = 0
        min_energy = float("inf")
        for conformer_identifier, struct in structs.items():
            if struct["final"]["energy"] < min_energy:
                min_energy = struct["final"]["energy"]
        for (
            conformer_identifier,
            struct,
        ) in structs.items():  # If we don't subtract min_energy, we get overflow errors
            weights[conformer_identifier] = np.exp(
                -(struct["final"]["energy"] - min_energy) / (boltzmann_constant * temp)
            )
            sum += weights[conformer_identifier]
        for conformer_identifier, struct in structs.items():
            weights[conformer_identifier] /= sum
            if weights[conformer_identifier] > 0.01:
                weighted_structs[conformer_identifier] = {
                    "atoms": struct["final"]["atoms"],
                    "weight": weights[conformer_identifier],
                }
        weighted_families[family_identifier] = weighted_structs
    return weighted_families


def calc_boltzmann_weighted_rmsd(orca_weighted_structs, mlip_weighted_structs):
    """
    Calculate the Boltzmann weighted RMSD via a cost matrix on ORCA and MLIP conformer ensembles.
    """
    cost_matrix = np.zeros(
        shape=(len(orca_weighted_structs.keys()), len(mlip_weighted_structs.keys()))
    )
    for ii, o_key in enumerate(orca_weighted_structs.keys()):
        for jj, m_key in enumerate(mlip_weighted_structs.keys()):
            cost_matrix[ii][jj] = abs(
                orca_weighted_structs[o_key]["weight"]
                - mlip_weighted_structs[m_key]["weight"]
            ) * sdgr_rmsd(
                orca_weighted_structs[o_key]["atoms"],
                mlip_weighted_structs[m_key]["atoms"],
            )
    row_ind, column_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, column_ind].sum()


def calc_ensemble_rmsd(orca_structs, mlip_structs):
    """
    Calculate the ensemble RMSD via a cost matrix on ORCA and MLIP conformer ensembles.
    """
    cost_matrix = np.zeros(shape=(len(orca_structs.keys()), len(mlip_structs.keys())))
    for ii, o_key in enumerate(orca_structs.keys()):
        for jj, m_key in enumerate(mlip_structs.keys()):
            cost_matrix[ii][jj] = sdgr_rmsd(
                orca_structs[o_key]["final"]["atoms"],
                mlip_structs[m_key]["final"]["atoms"],
            )
    row_ind, column_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, column_ind].sum()


def ligand_strain_processing(results):
    """
    Process results for the ligand strain evaluation task.
    Calculate the strain energy as the difference in energy between the global minimum and the loosely optimized ligand-in-pocket structure.
    Also save the global minimum structure for RMSD calculations.

    Args:
        results (dict): Results from ORCA or MLIP calculations.

    Returns:
        dict: Processed results for the ligand strain evaluation task.
    """
    processed_results = {}
    for identifier in results.keys():
        min_energy = float("inf")
        min_energy_struct = None
        for conformer_identifier, struct in results[identifier].items():
            if conformer_identifier != "ligand_in_pocket":
                if struct["final"]["energy"] < min_energy:
                    min_energy = struct["final"]["energy"]
                    min_energy_struct = struct
        processed_results[identifier]["global_min"] = min_energy_struct
        processed_results[identifier]["strain_energy"] = (
            results[identifier]["ligand_in_pocket"]["final"]["energy"] - min_energy
        )
    return processed_results


def get_one_prot_diff_name_pairs(names):
    """
    Get all pairs of names that have a charge difference of 1.

    Assumes that the names are in the format "name_charge_spin"
    """
    name_pairs = []
    for ii, name0 in enumerate(names):
        for jj in range(ii + 1, len(names)):
            name1 = names[jj]
            name0_charge = int(name0.split("_")[-2])
            name1_charge = int(name1.split("_")[-2])
            if abs(name0_charge - name1_charge) == 1:
                name_pairs.append((name0, name1))
    return name_pairs


# The use of task_metrics and eval are both mockups and will need help to function as envisioned
class OMol_Evaluator:
    def __init__(self, task_name):
        self.task_name = task_name

    def eval(self, orca_results, mlip_results):
        return eval(self.task_name)(orca_results, mlip_results)


def ligand_pocket(orca_results, mlip_results):
    """
    Calculate error metrics for ligand pocket evaluation task.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for ligand pocket evaluation task
    """
    energy_mae = 0
    forces_mae = 0
    forces_cosine_similarity = 0
    interaction_energy_mae = 0
    interaction_forces_mae = 0
    interaction_forces_cosine_similarity = 0
    orca_interaction_energy, orca_interaction_forces = interaction_energy_and_forces(
        orca_results, "ligand_pocket"
    )
    mlip_interaction_energy, mlip_interaction_forces = interaction_energy_and_forces(
        mlip_results, "ligand_pocket"
    )
    for identifier in orca_results.keys():
        for component_identifier in orca_results[identifier].keys():
            energy_mae += abs(
                orca_results[identifier][component_identifier]["energy"]
                - mlip_results[identifier][component_identifier]["energy"]
            )
            forces_mae += np.mean(
                np.abs(
                    np.array(orca_results[identifier][component_identifier]["forces"])
                    - np.array(mlip_results[identifier][component_identifier]["forces"])
                )
            )
            forces_cosine_similarity += cosine_similarity(
                np.array(orca_results[identifier][component_identifier]["forces"]),
                np.array(mlip_results[identifier][component_identifier]["forces"]),
            )
        interaction_energy_mae += abs(
            orca_interaction_energy[identifier] - mlip_interaction_energy[identifier]
        )
        interaction_forces_mae += np.mean(
            np.abs(
                orca_interaction_forces[identifier]
                - mlip_interaction_forces[identifier]
            )
        )
        interaction_forces_cosine_similarity += cosine_similarity(
            orca_interaction_forces[identifier], mlip_interaction_forces[identifier]
        )

    results = {
        "energy_mae": energy_mae / len(orca_results.keys()),
        "forces_mae": forces_mae / len(orca_results.keys()),
        "forces_cosine_similarity": forces_cosine_similarity / len(orca_results.keys()),
        "interaction_energy_mae": interaction_energy_mae / len(orca_results.keys()),
        "interaction_forces_mae": interaction_forces_mae / len(orca_results.keys()),
        "interaction_forces_cosine_similarity": interaction_forces_cosine_similarity
        / len(orca_results.keys()),
    }
    return results


def ligand_strain(orca_results, mlip_results):
    """
    Calculate error metrics for ligand strain evaluation task.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for ligand strain evaluation task
    """
    processed_orca_results = ligand_strain_processing(orca_results)
    processed_mlip_results = ligand_strain_processing(mlip_results)
    strain_energy_mae = 0
    global_min_rmsd = 0
    for identifier in orca_results.keys():
        strain_energy_mae += abs(
            processed_orca_results[identifier]["strain_energy"]
            - processed_mlip_results[identifier]["strain_energy"]
        )
        global_min_rmsd += sdgr_rmsd(
            processed_orca_results[identifier]["global_min"]["atoms"],
            processed_mlip_results[identifier]["global_min"]["atoms"],
        )

    results = {
        "strain_energy_mae": strain_energy_mae / len(orca_results.keys()),
        "global_min_rmsd": global_min_rmsd / len(orca_results.keys()),
    }
    return results


def geom_conformers_type1(orca_results, mlip_results):
    """
    Calculate error metrics for type1 conformer evaluation task.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for type1 conformer evaluation task
    """
    boltzmann_weighted_rmsd = 0
    ensemble_rmsd = 0
    orca_boltzmann_weighted_structures = boltzmann_weighted_structures(orca_results)
    mlip_boltzmann_weighted_structures = boltzmann_weighted_structures(mlip_results)
    for family_identifier, structs in orca_results.items():
        # Only need to do ensemble_rmsd once because there will always be the same
        # number of ORCA and MLIP structures
        ensemble_rmsd += calc_ensemble_rmsd(structs, mlip_results[family_identifier])
        # Need to do boltzmann_weighted_rmsd in both directions because there may be a
        # different number of ORCA and MLIP structures with non-trivial Boltzmann weights
        boltzmann_weighted_rmsd += calc_boltzmann_weighted_rmsd(
            orca_boltzmann_weighted_structures[family_identifier],
            mlip_boltzmann_weighted_structures[family_identifier],
        )
        boltzmann_weighted_rmsd += calc_boltzmann_weighted_rmsd(
            mlip_boltzmann_weighted_structures[family_identifier],
            orca_boltzmann_weighted_structures[family_identifier],
        )

    results = {
        "ensemble_rmsd": ensemble_rmsd,
        "boltzmann_weighted_rmsd": boltzmann_weighted_rmsd,
    }
    return results


def geom_conformers_type2(orca_results, mlip_results):
    """
    Calculate error metrics for type2 conformer evaluation task.
    deltaE_MLSP_i is the difference in energy calculated with the MLIP between the ORCA minimum energy conformer and conformer i without reoptimization by the MLIP
    deltaE_MLRX_i is the difference in energy calculated with the MLIP between the ORCA minimum energy conformer and conformer i after reoptimization by the MLIP
    Both are compared against the ORCA deltaE_i, with lower deltaE_MLSP_mae and deltaE_MLRX_mae values indicating a better match between the MLIP and DFT.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for type2 conformer evaluation task
    """
    energy_mae = 0
    forces_mae = 0
    forces_cosine_similarity = 0
    deltaE_MLSP_mae = 0
    deltaE_MLRX_mae = 0
    boltzmann_weighted_rmsd = 0
    orca_boltzmann_weighted_structures = boltzmann_weighted_structures(orca_results)
    mlip_boltzmann_weighted_structures = boltzmann_weighted_structures(mlip_results)
    for family_identifier, structs in orca_results.items():
        orca_min_energy = float("inf")
        orca_min_energy_id = None
        for conformer_identifier, struct in structs.items():
            if struct["final"]["energy"] < orca_min_energy:
                orca_min_energy = struct["final"]["energy"]
                orca_min_energy_id = conformer_identifier
            # Compare DFT to MLIP on the DFT optimized structure, which is the MLIP initial structure
            energy_mae += abs(
                struct["final"]["energy"]
                - mlip_results[family_identifier][conformer_identifier]["initial"][
                    "energy"
                ]
            )
            forces_mae += np.mean(
                np.abs(
                    np.array(struct["final"]["forces"])
                    - np.array(
                        mlip_results[family_identifier][conformer_identifier][
                            "initial"
                        ]["forces"]
                    )
                )
            )
            forces_cosine_similarity += cosine_similarity(
                np.array(struct["final"]["forces"]),
                np.array(
                    mlip_results[family_identifier][conformer_identifier]["initial"][
                        "forces"
                    ],
                ),
            )
        for conformer_identifier, struct in structs.items():
            orca_deltaE = struct["final"]["energy"] - orca_min_energy
            deltaE_MLSP = (
                mlip_results[family_identifier][conformer_identifier]["initial"][
                    "energy"
                ]
                - mlip_results[family_identifier][orca_min_energy_id]["initial"][
                    "energy"
                ]
            )
            deltaE_MLRX = (
                mlip_results[family_identifier][conformer_identifier]["final"]["energy"]
                - mlip_results[family_identifier][orca_min_energy_id]["final"]["energy"]
            )
            deltaE_MLSP_mae += abs(orca_deltaE - deltaE_MLSP)
            deltaE_MLRX_mae += abs(orca_deltaE - deltaE_MLRX)
        boltzmann_weighted_rmsd += calc_boltzmann_weighted_rmsd(
            orca_boltzmann_weighted_structures[family_identifier],
            mlip_boltzmann_weighted_structures[family_identifier],
        )
        boltzmann_weighted_rmsd += calc_boltzmann_weighted_rmsd(
            mlip_boltzmann_weighted_structures[family_identifier],
            orca_boltzmann_weighted_structures[family_identifier],
        )

    results = {
        "energy_mae": energy_mae / len(orca_results.keys()),
        "forces_mae": forces_mae / len(orca_results.keys()),
        "forces_cosine_similarity": forces_cosine_similarity / len(orca_results.keys()),
        "deltaE_MLSP_mae": deltaE_MLSP_mae / len(orca_results.keys()),
        "deltaE_MLRX_mae": deltaE_MLRX_mae / len(orca_results.keys()),
        "boltzmann_weighted_rmsd": boltzmann_weighted_rmsd,
    }
    return results


def protonation_energies_type1(orca_results, mlip_results):
    """
    Calculate error metrics for the type1 protonation energies evaluation task.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for protonation energies evaluation task
    """
    deltaE_mae = 0
    rmsd = 0
    for identifier, structs in orca_results.items():
        one_prot_diff_name_pairs = get_one_prot_diff_name_pairs(list(structs.keys()))
        for name0, name1 in one_prot_diff_name_pairs:
            orca_deltaE = (
                structs[name0]["final"]["energy"] - structs[name1]["final"]["energy"]
            )
            mlip_deltaE = (
                mlip_results[identifier][name0]["final"]["energy"]
                - mlip_results[identifier][name1]["final"]["energy"]
            )
            deltaE_mae += abs(orca_deltaE - mlip_deltaE)
            rmsd += sdgr_rmsd(
                structs[name0]["final"]["atoms"],
                mlip_results[identifier][name0]["final"]["atoms"],
            )
            rmsd += sdgr_rmsd(
                structs[name1]["final"]["atoms"],
                mlip_results[identifier][name1]["final"]["atoms"],
            )

    results = {
        "deltaE_mae": deltaE_mae / len(orca_results.keys()),
        "rmsd": rmsd / len(orca_results.keys()),
    }
    return results


def protonation_energies_type2(orca_results, mlip_results):
    """
    Calculate error metrics for the type2 protonation energies evaluation task.
    deltaE_MLSP_ij is energy difference between species i and j calculated with the MLIP using ORCA minimized structures without reoptimization by the MLIP
    deltaE_MLRX_ij is energy difference between species i and j calculated with the MLIP where ORCA minimized structures are reoptimized by the MLIP
    Both are compared against the ORCA deltaE_ij, with lower deltaE_MLSP_mae and deltaE_MLRX_mae values indicating a better match between the MLIP and DFT.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for type2 conformer evaluation task
    """
    energy_mae = 0
    forces_mae = 0
    forces_cosine_similarity = 0
    deltaE_MLSP_mae = 0
    deltaE_MLRX_mae = 0
    for identifier, structs in orca_results.items():
        for name in structs.keys():
            # Compare DFT to MLIP on the DFT optimized structure, which is the MLIP initial structure
            energy_mae += abs(
                structs[name]["final"]["energy"]
                - mlip_results[identifier][name]["initial"]["energy"]
            )
            forces_mae += np.mean(
                np.abs(
                    np.array(structs[name]["final"]["forces"])
                    - np.array(mlip_results[identifier][name]["initial"]["forces"])
                )
            )
            forces_cosine_similarity += cosine_similarity(
                np.array(structs[name]["final"]["forces"]),
                np.array(mlip_results[identifier][name]["initial"]["forces"]),
            )
        one_prot_diff_name_pairs = get_one_prot_diff_name_pairs(list(structs.keys()))
        for name0, name1 in one_prot_diff_name_pairs:
            orca_deltaE = (
                structs[name0]["final"]["energy"] - structs[name1]["final"]["energy"]
            )
            deltaE_MLSP = (
                mlip_results[identifier][name0]["initial"]["energy"]
                - mlip_results[identifier][name1]["initial"]["energy"]
            )
            deltaE_MLRX = (
                mlip_results[identifier][name0]["final"]["energy"]
                - mlip_results[identifier][name1]["final"]["energy"]
            )
            deltaE_MLSP_mae += abs(orca_deltaE - deltaE_MLSP)
            deltaE_MLRX_mae += abs(orca_deltaE - deltaE_MLRX)

    results = {
        "energy_mae": energy_mae / len(orca_results.keys()),
        "forces_mae": forces_mae / len(orca_results.keys()),
        "forces_cosine_similarity": forces_cosine_similarity / len(orca_results.keys()),
        "deltaE_MLSP_mae": deltaE_MLSP_mae / len(orca_results.keys()),
        "deltaE_MLRX_mae": deltaE_MLRX_mae / len(orca_results.keys()),
    }
    return results


def unoptimized_ie_ea(orca_results, mlip_results):
    """
    Calculate error metrics for unoptimized IE and EA calculations.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for unoptimized IE and EA calculations.
    """
    energy_mae = 0
    forces_mae = 0
    forces_cosine_similarity = 0
    deltaE_mae = 0
    deltaF_mae = 0
    deltaF_cosine_similarity = 0
    orca_deltaE, orca_deltaF = charge_deltas(orca_results)
    mlip_deltaE, mlip_deltaF = charge_deltas(mlip_results)
    for identifier in orca_results.keys():
        for tag in ["original", "add_electron", "remove_electron"]:
            for spin in orca_results[identifier][tag].keys():
                energy_mae += abs(
                    orca_results[identifier][tag][spin]["energy"]
                    - mlip_results[identifier][tag][spin]["energy"]
                )
                forces_mae += np.mean(
                    np.abs(
                        np.array(orca_results[identifier][tag][spin]["forces"])
                        - np.array(mlip_results[identifier][tag][spin]["forces"])
                    )
                )
                forces_cosine_similarity += cosine_similarity(
                    np.array(orca_results[identifier][tag][spin]["forces"]),
                    np.array(mlip_results[identifier][tag][spin]["forces"]),
                )
                if tag != "original":
                    deltaE_mae += abs(
                        orca_deltaE[identifier][tag][spin]
                        - mlip_deltaE[identifier][tag][spin]
                    )
                    deltaF_mae += np.mean(
                        np.abs(
                            orca_deltaF[identifier][tag][spin]
                            - mlip_deltaF[identifier][tag][spin]
                        )
                    )
                    deltaF_cosine_similarity += cosine_similarity(
                        orca_deltaF[identifier][tag][spin],
                        mlip_deltaF[identifier][tag][spin],
                    )

    results = {
        "energy_mae": energy_mae / len(orca_results.keys()),
        "forces_mae": forces_mae / len(orca_results.keys()),
        "forces_cosine_similarity": forces_cosine_similarity / len(orca_results.keys()),
        "deltaE_mae": deltaE_mae / len(orca_results.keys()),
        "deltaF_mae": deltaF_mae / len(orca_results.keys()),
        "deltaF_cosine_similarity": deltaF_cosine_similarity / len(orca_results.keys()),
    }
    return results


def distance_scaling(orca_results, mlip_results):
    """
    Calculate error metrics for distance scaling evaluation task.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for distance scaling evaluation task
    """
    energy_mae = 0
    forces_mae = 0
    forces_cosine_similarity = 0
    interaction_energy_mae = 0
    interaction_forces_mae = 0
    interaction_forces_cosine_similarity = 0
    orca_interaction_energy, orca_interaction_forces = distance_scaling_processing(
        orca_results
    )
    mlip_interaction_energy, mlip_interaction_forces = distance_scaling_processing(
        mlip_results
    )
    for identifier in orca_results.keys():
        for component_identifier in orca_results[identifier].keys():
            energy_mae += abs(
                orca_results[identifier][component_identifier]["energy"]
                - mlip_results[identifier][component_identifier]["energy"]
            )
            forces_mae += np.mean(
                np.abs(
                    np.array(orca_results[identifier][component_identifier]["forces"])
                    - np.array(mlip_results[identifier][component_identifier]["forces"])
                )
            )
            forces_cosine_similarity += cosine_similarity(
                np.array(orca_results[identifier][component_identifier]["forces"]),
                np.array(mlip_results[identifier][component_identifier]["forces"]),
            )
        interaction_energy_mae += abs(
            orca_interaction_energy[identifier] - mlip_interaction_energy[identifier]
        )
        interaction_forces_mae += np.mean(
            np.abs(
                orca_interaction_forces[identifier]
                - mlip_interaction_forces[identifier]
            )
        )
        interaction_forces_cosine_similarity += cosine_similarity(
            orca_interaction_forces[identifier], mlip_interaction_forces[identifier]
        )

    results = {
        "energy_mae": energy_mae / len(orca_results.keys()),
        "forces_mae": forces_mae / len(orca_results.keys()),
        "forces_cosine_similarity": forces_cosine_similarity / len(orca_results.keys()),
        "interaction_energy_mae": interaction_energy_mae / len(orca_results.keys()),
        "interaction_forces_mae": interaction_forces_mae / len(orca_results.keys()),
        "interaction_forces_cosine_similarity": interaction_forces_cosine_similarity
        / len(orca_results.keys()),
    }
    return results


def unoptimized_spin_gap(orca_results, mlip_results):
    """
    Calculate error metrics for unoptimized spin gap evaluation task.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for unoptimized spin gap evaluation task
    """
    energy_mae = 0
    forces_mae = 0
    forces_cosine_similarity = 0
    deltaE_mae = 0
    deltaF_mae = 0
    deltaF_cosine_similarity = 0
    orca_deltaE, orca_deltaF = spin_deltas(orca_results)
    mlip_deltaE, mlip_deltaF = spin_deltas(mlip_results)
    for identifier in orca_results.keys():
        spins = []
        for spin in orca_results[identifier].keys():
            spins.append(int(spin))
        spins.sort(reverse=True)
        for spin in spins:
            energy_mae += abs(
                orca_results[identifier][str(spin)]["energy"]
                - mlip_results[identifier][str(spin)]["energy"]
            )
            forces_mae += np.mean(
                np.abs(
                    np.array(orca_results[identifier][str(spin)]["forces"])
                    - np.array(mlip_results[identifier][str(spin)]["forces"])
                )
            )
            forces_cosine_similarity += cosine_similarity(
                np.array(orca_results[identifier][str(spin)]["forces"]),
                np.array(mlip_results[identifier][str(spin)]["forces"]),
            )
            if spin != spins[0]:
                deltaE_mae += abs(
                    orca_deltaE[identifier][spin] - mlip_deltaE[identifier][spin]
                )
                deltaF_mae += np.mean(
                    np.abs(
                        orca_deltaF[identifier][spin] - mlip_deltaF[identifier][spin]
                    )
                )
                deltaF_cosine_similarity += cosine_similarity(
                    orca_deltaF[identifier][spin], mlip_deltaF[identifier][spin]
                )

    results = {
        "energy_mae": energy_mae / len(orca_results.keys()),
        "forces_mae": forces_mae / len(orca_results.keys()),
        "forces_cosine_similarity": forces_cosine_similarity / len(orca_results.keys()),
        "deltaE_mae": deltaE_mae / len(orca_results.keys()),
        "deltaF_mae": deltaF_mae / len(orca_results.keys()),
        "deltaF_cosine_similarity": deltaF_cosine_similarity / len(orca_results.keys()),
    }
    return results
