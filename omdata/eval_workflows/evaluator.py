from __future__ import annotations

from collections import defaultdict

import numpy as np
from pymatgen.io.ase import MSONAtoms
from schrodinger.application.jaguar.utils import mmjag_reset_connectivity
from schrodinger.application.matsci.aseutils import get_structure
from schrodinger.comparison.atom_mapper import ConnectivityAtomMapper
from schrodinger.structure import Structure
from schrodinger.structutils import rmsd
from schrodinger.application.jaguar.autots_bonding import copy_bonding
from scipy.optimize import linear_sum_assignment

boltzmann_constant = 8.617333262 * 10**-5



# Helper/processing functions

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
    return rmsd.superimpose(st1, at_list, st2, at_list, use_symmetry=False)


def sdgr_rmsd(orca_atoms, mlip_atoms):
    """
    Calculate the RMSD between atoms optimized with ORCA and the MLIP,
    where we assume that ORCA atoms have sensible bonding.
    """
    orca_atoms = MSONAtoms.from_dict(orca_atoms)
    mlip_atoms = MSONAtoms.from_dict(mlip_atoms)

    orca_st = get_structure(orca_atoms)
    mlip_st = get_structure(mlip_atoms)
    return rmsd_wrapper(orca_st, mlip_st)


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
    for identifier in results:
        indices_found = set()
        interaction_energy[identifier] = results[identifier][principal_identifier][
            "energy"
        ]
        interaction_forces[identifier] = np.array(
            results[identifier][principal_identifier]["forces"]
        )
        principal_atoms = MSONAtoms.from_dict(
            results[identifier][principal_identifier]["atoms"]
        )
        for component_identifier in results[identifier]:
            if component_identifier != principal_identifier:
                interaction_energy[identifier] -= results[identifier][
                    component_identifier
                ]["energy"]
                component_atoms = MSONAtoms.from_dict(
                    results[identifier][component_identifier]["atoms"]
                )
                for ii, sub_atom in enumerate(component_atoms):
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
    """
    Calculate the interaction energy and interaction forces for each snapshot with a distance scaling factor
    by subtracting the energy and forces of the snapshot with the maximum distance scaling factor. Short range
    and long range samples are calculated separately.

    Args:
        results (dict): Results from ORCA or MLIP calculations.

    Returns:
        interaction_energy_sr, interaction_forces_sr, interaction_energy_lr, interaction_forces_lr
    """
    interaction_energy_sr = {}
    interaction_forces_sr = {}
    interaction_energy_lr = {}
    interaction_forces_lr = {}
    for identifier in results:
        interaction_energy_sr[identifier] = {}
        interaction_forces_sr[identifier] = {}
        interaction_energy_lr[identifier] = {}
        interaction_forces_lr[identifier] = {}
        scaling_factors = []
        for scaling_factor in results[identifier]:
            scaling_factors.append(float(scaling_factor))
        scaling_factors.sort(reverse=True)
        for scaling_factor in scaling_factors[1:]:
            if scaling_factor > 2.5:
                interaction_energy_lr[identifier][scaling_factor] = (
                    results[identifier][str(scaling_factor)]["energy"]
                    - results[identifier][str(scaling_factors[0])]["energy"]
                )
                interaction_forces_lr[identifier][scaling_factor] = np.array(
                    results[identifier][str(scaling_factor)]["forces"]
                ) - np.array(results[identifier][str(scaling_factors[0])]["forces"])
            else:
                interaction_energy_sr[identifier][scaling_factor] = (
                    results[identifier][str(scaling_factor)]["energy"]
                    - results[identifier][str(scaling_factors[0])]["energy"]
                )
                interaction_forces_sr[identifier][scaling_factor] = np.array(
                    results[identifier][str(scaling_factor)]["forces"]
                ) - np.array(results[identifier][str(scaling_factors[0])]["forces"])
    return interaction_energy_sr, interaction_forces_sr, interaction_energy_lr, interaction_forces_lr


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
    for identifier in results:
        deltaE[identifier] = {}
        deltaF[identifier] = {}
        spins = [int(spin) for spin in results[identifier]]
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
    for identifier in results:
        charges = [int(charge) for charge in results[identifier]]
        charges.sort()
        assert charges[1] - 1 == charges[0]
        assert charges[1] + 1 == charges[2]
        deltaE[identifier] = {"add_electron": {}, "remove_electron": {}}
        deltaF[identifier] = {"add_electron": {}, "remove_electron": {}}
        assert len(results[identifier][str(charges[1])]) == 1
        orig_spin = list(results[identifier][str(charges[1])])[0]
        orig_energy = results[identifier][str(charges[1])][orig_spin]["energy"]
        orig_forces = np.array(
            results[identifier][str(charges[1])][orig_spin]["forces"]
        )
        for charge_val, tag in [
            (str(charges[0]), "add_electron"),
            (str(charges[2]), "remove_electron"),
        ]:
            for spin in results[identifier][charge_val]:
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
        min_energy_struct = min(structs.values(), key=lambda x: x["final"]["energy"])
        min_energy = min_energy_struct["energy"]
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


def calc_boltzmann_weighted_rmsd(weighted_structs0, weighted_structs1):
    """
    Calculate the Boltzmann weighted RMSD via linear sum assignment on a cost matrix for weighted conformer ensembles.
    """
    cost_matrix = np.zeros(
        shape=(len(weighted_structs0), len(weighted_structs1))
    )
    for ii, key0 in enumerate(weighted_structs0):
        for jj, key1 in enumerate(weighted_structs1):
            cost_matrix[ii][jj] = abs(
                weighted_structs0[key0]["weight"]
            ) * sdgr_rmsd(
                weighted_structs0[key0]["atoms"],
                weighted_structs1[key1]["atoms"],
            )
    row_ind, column_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, column_ind].sum() # Boltzmann weights provide implicit normalization


def calc_ensemble_rmsd(structs0, structs1):
    """
    Calculate the ensemble RMSD via linear sum assignment on a cost matrix for conformer ensembles.
    """
    cost_matrix = np.zeros(shape=(len(structs0), len(structs1)))
    for ii, key0 in enumerate(structs0):
        for jj, key1 in enumerate(structs1):
            cost_matrix[ii][jj] = sdgr_rmsd(
                structs0[key0]["final"]["atoms"],
                structs1[key1]["final"]["atoms"],
            )
    row_ind, column_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, column_ind].mean() # Mean gives normalized cost


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
    processed_results = defaultdict(dict)
    for identifier in results:
        min_energy_struct = min(results[identifier]["gas_phase"].values(), key=lambda x: x["final"]["energy"])
        min_energy = min_energy_struct["energy"]
        processed_results[identifier]["global_min"] = min_energy_struct
        processed_results[identifier]["strain_energy"] = (
            results[identifier]["bioactive"]["energy"] - min_energy
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



# Evaluator functions

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
    for identifier in orca_results:
        assert len(orca_results[identifier]) == 3
        assert len(mlip_results[identifier]) == 3
        for component_identifier in orca_results[identifier]:
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
        "energy_mae": energy_mae / (3 * len(orca_results)),
        "forces_mae": forces_mae / (3 * len(orca_results)),
        "forces_cosine_similarity": forces_cosine_similarity / (3 * len(orca_results)),
        "interaction_energy_mae": interaction_energy_mae / len(orca_results),
        "interaction_forces_mae": interaction_forces_mae / len(orca_results),
        "interaction_forces_cosine_similarity": interaction_forces_cosine_similarity
        / len(orca_results),
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
    for identifier in orca_results:
        strain_energy_mae += abs(
            processed_orca_results[identifier]["strain_energy"]
            - processed_mlip_results[identifier]["strain_energy"]
        )
        global_min_rmsd += sdgr_rmsd(
            processed_orca_results[identifier]["global_min"]["atoms"],
            processed_mlip_results[identifier]["global_min"]["atoms"],
        )

    results = {
        "strain_energy_mae": strain_energy_mae / len(orca_results),
        "global_min_rmsd": global_min_rmsd / len(orca_results),
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
        ensemble_rmsd += calc_ensemble_rmsd(structs, mlip_results[family_identifier])
        boltzmann_weighted_rmsd += calc_boltzmann_weighted_rmsd(
            orca_boltzmann_weighted_structures[family_identifier],
            mlip_boltzmann_weighted_structures[family_identifier],
        )/2
        boltzmann_weighted_rmsd += calc_boltzmann_weighted_rmsd(
            mlip_boltzmann_weighted_structures[family_identifier],
            orca_boltzmann_weighted_structures[family_identifier],
        )/2

    results = {
        "ensemble_rmsd": ensemble_rmsd / len(orca_results),
        "boltzmann_weighted_rmsd": boltzmann_weighted_rmsd / len(orca_results),
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
    reopt_rmsd = 0
    for family_identifier, structs in orca_results.items():
        for conformer_identifier, struct in structs.items():
            # Compare DFT to MLIP on the DFT optimized structure, which is the MLIP initial structure
            energy_mae += abs(
                struct["final"]["energy"]
                - mlip_results[family_identifier][conformer_identifier]["initial"][
                    "energy"
                ]
            )/len(structs)
            forces_mae += np.mean(
                np.abs(
                    np.array(struct["final"]["forces"])
                    - np.array(
                        mlip_results[family_identifier][conformer_identifier][
                            "initial"
                        ]["forces"]
                    )
                )
            )/len(structs)
            forces_cosine_similarity += cosine_similarity(
                np.array(struct["final"]["forces"]),
                np.array(
                    mlip_results[family_identifier][conformer_identifier]["initial"][
                        "forces"
                    ],
                ),
            )/len(structs)
        orca_min_energy_id, min_energy_struct = min(structs.items(), key=lambda x: x[1]['final']['energy'])
        orca_min_energy = min_energy_struct["energy"]
        for conformer_identifier, struct in structs.items():
            orca_deltaE = struct["final"]["energy"] - orca_min_energy
            # TODO: Determine if deltaE_MLSP is directly correlated with energy_mae
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
            deltaE_MLSP_mae += abs(orca_deltaE - deltaE_MLSP)/len(structs)
            deltaE_MLRX_mae += abs(orca_deltaE - deltaE_MLRX)/len(structs)
            reopt_rmsd += sdgr_rmsd(
                mlip_results[family_identifier][conformer_identifier]["initial"]["atoms"],
                mlip_results[family_identifier][conformer_identifier]["final"]["atoms"],
            )/len(structs)

    results = {
        "energy_mae": energy_mae / len(orca_results),
        "forces_mae": forces_mae / len(orca_results),
        "forces_cosine_similarity": forces_cosine_similarity / len(orca_results),
        "deltaE_MLSP_mae": deltaE_MLSP_mae / len(orca_results),
        "deltaE_MLRX_mae": deltaE_MLRX_mae / len(orca_results),
        "reopt_rmsd": reopt_rmsd / len(orca_results),
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
        one_prot_diff_name_pairs = get_one_prot_diff_name_pairs(list(structs))
        for name, struct in structs.items():
            rmsd += sdgr_rmsd(
                struct["final"]["atoms"],
                mlip_results[identifier][name]["final"]["atoms"],
            )/len(structs)
        for name0, name1 in one_prot_diff_name_pairs:
            orca_deltaE = (
                structs[name0]["final"]["energy"] - structs[name1]["final"]["energy"]
            )
            mlip_deltaE = (
                mlip_results[identifier][name0]["final"]["energy"]
                - mlip_results[identifier][name1]["final"]["energy"]
            )
            deltaE_mae += abs(orca_deltaE - mlip_deltaE)/len(one_prot_diff_name_pairs)

    results = {
        "deltaE_mae": deltaE_mae / len(orca_results),
        "rmsd": rmsd / len(orca_results),
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
    reopt_rmsd = 0
    for identifier, structs in orca_results.items():
        for name in structs:
            # Compare DFT to MLIP on the DFT optimized structure, which is the MLIP initial structure
            energy_mae += abs(
                structs[name]["final"]["energy"]
                - mlip_results[identifier][name]["initial"]["energy"]
            )/len(structs)
            forces_mae += np.mean(
                np.abs(
                    np.array(structs[name]["final"]["forces"])
                    - np.array(mlip_results[identifier][name]["initial"]["forces"])
                )
            )/len(structs)
            forces_cosine_similarity += cosine_similarity(
                np.array(structs[name]["final"]["forces"]),
                np.array(mlip_results[identifier][name]["initial"]["forces"]),
            )/len(structs)
            reopt_rmsd += sdgr_rmsd(
                mlip_results[identifier][name]["initial"]["atoms"],
                mlip_results[identifier][name]["final"]["atoms"],
            )/len(structs)
        one_prot_diff_name_pairs = get_one_prot_diff_name_pairs(list(structs))
        for name0, name1 in one_prot_diff_name_pairs:
            orca_deltaE = (
                structs[name0]["final"]["energy"] - structs[name1]["final"]["energy"]
            )
            # TODO: Determine if deltaE_MLSP is directly correlated with energy_mae
            deltaE_MLSP = (
                mlip_results[identifier][name0]["initial"]["energy"]
                - mlip_results[identifier][name1]["initial"]["energy"]
            )
            deltaE_MLRX = (
                mlip_results[identifier][name0]["final"]["energy"]
                - mlip_results[identifier][name1]["final"]["energy"]
            )
            deltaE_MLSP_mae += abs(orca_deltaE - deltaE_MLSP)/len(one_prot_diff_name_pairs)
            deltaE_MLRX_mae += abs(orca_deltaE - deltaE_MLRX)/len(one_prot_diff_name_pairs)

    results = {
        "energy_mae": energy_mae / len(orca_results),
        "forces_mae": forces_mae / len(orca_results),
        "forces_cosine_similarity": forces_cosine_similarity / len(orca_results),
        "deltaE_MLSP_mae": deltaE_MLSP_mae / len(orca_results),
        "deltaE_MLRX_mae": deltaE_MLRX_mae / len(orca_results),
        "reopt_rmsd": reopt_rmsd / len(orca_results),
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
    for identifier in orca_results:
        num_samples = sum(len(structs) for structs in orca_results[identifier].values())
        num_deltas = sum(len(structs) for structs in orca_deltaE[identifier].values())
        assert num_samples - 1 == num_deltas
        for charge in orca_results[identifier]:
            for spin in orca_results[identifier][charge]:
                energy_mae += abs(
                    orca_results[identifier][charge][spin]["energy"]
                    - mlip_results[identifier][charge][spin]["energy"]
                )/num_samples
                forces_mae += np.mean(
                    np.abs(
                        np.array(orca_results[identifier][charge][spin]["forces"])
                        - np.array(mlip_results[identifier][charge][spin]["forces"])
                    )
                )/num_samples
                forces_cosine_similarity += cosine_similarity(
                    np.array(orca_results[identifier][charge][spin]["forces"]),
                    np.array(mlip_results[identifier][charge][spin]["forces"]),
                )/num_samples

        for tag in ["add_electron", "remove_electron"]:
            for spin in orca_deltaE[identifier][tag]:
                deltaE_mae += abs(
                    orca_deltaE[identifier][tag][spin]
                    - mlip_deltaE[identifier][tag][spin]
                )/num_deltas
                deltaF_mae += np.mean(
                    np.abs(
                        orca_deltaF[identifier][tag][spin]
                        - mlip_deltaF[identifier][tag][spin]
                    )
                )/num_deltas
                deltaF_cosine_similarity += cosine_similarity(
                    orca_deltaF[identifier][tag][spin],
                    mlip_deltaF[identifier][tag][spin],
                )/num_deltas

    results = {
        "energy_mae": energy_mae / len(orca_results),
        "forces_mae": forces_mae / len(orca_results),
        "forces_cosine_similarity": forces_cosine_similarity / len(orca_results),
        "deltaE_mae": deltaE_mae / len(orca_results),
        "deltaF_mae": deltaF_mae / len(orca_results),
        "deltaF_cosine_similarity": deltaF_cosine_similarity / len(orca_results),
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
    for identifier in orca_results:
        for component_identifier in orca_results[identifier]:
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
        "energy_mae": energy_mae / len(orca_results),
        "forces_mae": forces_mae / len(orca_results),
        "forces_cosine_similarity": forces_cosine_similarity / len(orca_results),
        "interaction_energy_mae": interaction_energy_mae / len(orca_results),
        "interaction_forces_mae": interaction_forces_mae / len(orca_results),
        "interaction_forces_cosine_similarity": interaction_forces_cosine_similarity
        / len(orca_results),
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
    for identifier in orca_results:
        spins = [int(spin) for spin in orca_results[identifier]]
        spins.sort(reverse=True)
        for spin in spins:
            energy_mae += abs(
                orca_results[identifier][str(spin)]["energy"]
                - mlip_results[identifier][str(spin)]["energy"]
            )/len(spins)
            forces_mae += np.mean(
                np.abs(
                    np.array(orca_results[identifier][str(spin)]["forces"])
                    - np.array(mlip_results[identifier][str(spin)]["forces"])
                )
            )/len(spins)
            forces_cosine_similarity += cosine_similarity(
                np.array(orca_results[identifier][str(spin)]["forces"]),
                np.array(mlip_results[identifier][str(spin)]["forces"]),
            )/len(spins)
            if spin != spins[0]:
                deltaE_mae += abs(
                    orca_deltaE[identifier][spin] - mlip_deltaE[identifier][spin]
                )/(len(spins)-1)
                deltaF_mae += np.mean(
                    np.abs(
                        orca_deltaF[identifier][spin] - mlip_deltaF[identifier][spin]
                    )
                )/(len(spins)-1)
                deltaF_cosine_similarity += cosine_similarity(
                    orca_deltaF[identifier][spin], mlip_deltaF[identifier][spin]
                )/(len(spins)-1)

    results = {
        "energy_mae": energy_mae / len(orca_results),
        "forces_mae": forces_mae / len(orca_results),
        "forces_cosine_similarity": forces_cosine_similarity / len(orca_results),
        "deltaE_mae": deltaE_mae / len(orca_results),
        "deltaF_mae": deltaF_mae / len(orca_results),
        "deltaF_cosine_similarity": deltaF_cosine_similarity / len(orca_results),
    }
    return results


def singlepoint(orca_results, mlip_results):
    target_energies = []
    target_forces = []
    energies = []
    forces = []
    for identifier in orca_results:
        target_energies.append(orca_results[identifier]["energy"])
        target_forces.append(orca_results[identifier]["forces"])
        energies.append(mlip_results[identifier]["energy"])
        forces.append(mlip_results[identifier]["forces"])

    target_energies = np.array(target_energies)
    target_forces = np.concatenate(target_forces)
    energies = np.array(energies)
    forces = np.concatenate(forces)

    metrics = {
        "energy_mae": np.mean(np.abs(energies - target_energies)),
        "forces_mae": np.mean(np.abs(forces - target_forces)),
    }

    return metrics
