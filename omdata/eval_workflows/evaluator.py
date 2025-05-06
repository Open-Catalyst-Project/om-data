from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
from pymatgen.io.ase import MSONAtoms
from schrodinger.application.matsci.aseutils import get_structure
from schrodinger.structure import Structure
from schrodinger.structutils import rmsd
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
    return np.sum(forces_1 * forces_2) / max(
        np.linalg.norm(forces_1) * np.linalg.norm(forces_2), 1e-8
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


def calc_boltzmann_weights(results, temp=298.15):
    """
    Assign Boltzmann weights to the conformers in each family.

    Args:
        results (dict): Conformer results from ORCA or MLIP calculations.
        temp (float): Temperature in Kelvin.

    Returns:
        list: Boltzmann weights for each conformer.
    """
    family_weights = {}
    for family_identifier, structs in results.items():
        weights = np.zeros(len(structs))
        sum = 0
        min_energy = min(struct["final"]["energy"] for struct in structs.values())
        for ii, struct in enumerate(structs.values()):  # If we don't subtract min_energy, we get overflow errors
            weights[ii] = np.exp(
                -(struct["final"]["energy"] - min_energy) / (boltzmann_constant * temp)
            )
            sum += weights[ii]
        family_weights[family_identifier] = np.array([weight / sum for weight in weights])
    return family_weights


def rmsd_mapping(structs0, structs1):
    """
    Map two conformer ensembles via linear sum assignment on an RMSD cost matrix.
    """
    cost_matrix = np.zeros(shape=(len(structs0), len(structs1)))
    for ii, key0 in enumerate(structs0):
        for jj, key1 in enumerate(structs1):
            cost_matrix[ii][jj] = sdgr_rmsd(
                structs0[key0]["final"]["atoms"],
                structs1[key1]["final"]["atoms"],
            )
    row_ind, column_ind = linear_sum_assignment(cost_matrix)
    assert (row_ind == sorted(row_ind)).all()
    mapping = {}
    for ii, jj in zip(row_ind, column_ind):
        mapping[list(structs0)[ii]] = list(structs1)[jj]
    return mapping, cost_matrix[row_ind, column_ind]


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
        min_energy_struct = min(
            results[identifier]["gas_phase"].values(),
            key=lambda x: x["final"]["energy"],
        )
        min_energy = min_energy_struct["final"]["energy"]
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
    for name0, name1 in itertools.combinations(names, 2):
        name0_charge = int(name0.split("_")[-2])
        name1_charge = int(name1.split("_")[-2])
        if abs(name0_charge - name1_charge) == 1:
            name_pairs.append((name0, name1))
    return name_pairs


def sr_or_lr(name):
    if "short_range" in name:
        return "sr"
    elif "long_range" in name:
        return "lr"


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
            processed_orca_results[identifier]["global_min"]["final"]["atoms"],
            processed_mlip_results[identifier]["global_min"]["final"]["atoms"],
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
    deltaE_mae = 0
    orca_boltzmann_weights = calc_boltzmann_weights(orca_results)
    mlip_boltzmann_weights = calc_boltzmann_weights(mlip_results)
    for family_identifier, structs in orca_results.items():
        mapping, cost_vector = rmsd_mapping(structs, mlip_results[family_identifier])
        ensemble_rmsd += cost_vector.mean()
        boltzmann_weighted_rmsd += sum(orca_boltzmann_weights[family_identifier] * cost_vector) / 2
        boltzmann_weighted_rmsd += sum(mlip_boltzmann_weights[family_identifier] * cost_vector) / 2

        orca_min_energy_id, min_energy_struct = min(
            structs.items(), key=lambda x: x[1]["final"]["energy"]
        )
        orca_min_energy = min_energy_struct["final"]["energy"]
        mlip_energy_of_orca_min = mlip_results[family_identifier][mapping[orca_min_energy_id]]["final"]["energy"]
        for conformer_identifier, struct in structs.items():
            if conformer_identifier != orca_min_energy_id:
                orca_deltaE = struct["final"]["energy"] - orca_min_energy
                mlip_deltaE = mlip_results[family_identifier][mapping[conformer_identifier]]["final"]["energy"] - mlip_energy_of_orca_min
                deltaE_mae += abs(orca_deltaE - mlip_deltaE) / len(structs)

    results = {
        "ensemble_rmsd": ensemble_rmsd / len(orca_results),
        "boltzmann_weighted_rmsd": boltzmann_weighted_rmsd / len(orca_results),
        "deltaE_mae": deltaE_mae / len(orca_results),
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
            ) / len(structs)
            forces_mae += np.mean(
                np.abs(
                    np.array(struct["final"]["forces"])
                    - np.array(
                        mlip_results[family_identifier][conformer_identifier][
                            "initial"
                        ]["forces"]
                    )
                )
            ) / len(structs)
            forces_cosine_similarity += cosine_similarity(
                np.array(struct["final"]["forces"]),
                np.array(
                    mlip_results[family_identifier][conformer_identifier]["initial"][
                        "forces"
                    ],
                ),
            ) / len(structs)
        orca_min_energy_id, min_energy_struct = min(
            structs.items(), key=lambda x: x[1]["final"]["energy"]
        )
        orca_min_energy = min_energy_struct["final"]["energy"]
        for conformer_identifier, struct in structs.items():
            if conformer_identifier != orca_min_energy_id:
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
                    mlip_results[family_identifier][conformer_identifier]["final"][
                        "energy"
                    ]
                    - mlip_results[family_identifier][orca_min_energy_id]["final"][
                        "energy"
                    ]
                )
                deltaE_MLSP_mae += abs(orca_deltaE - deltaE_MLSP) / (len(structs) - 1)
                deltaE_MLRX_mae += abs(orca_deltaE - deltaE_MLRX) / (len(structs) - 1)
                reopt_rmsd += sdgr_rmsd(
                    mlip_results[family_identifier][conformer_identifier]["initial"][
                        "atoms"
                    ],
                    mlip_results[family_identifier][conformer_identifier]["final"][
                        "atoms"
                    ],
                ) / (len(structs) - 1)

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
            ) / len(structs)
        for name0, name1 in one_prot_diff_name_pairs:
            orca_deltaE = (
                structs[name0]["final"]["energy"] - structs[name1]["final"]["energy"]
            )
            mlip_deltaE = (
                mlip_results[identifier][name0]["final"]["energy"]
                - mlip_results[identifier][name1]["final"]["energy"]
            )
            deltaE_mae += abs(orca_deltaE - mlip_deltaE) / len(one_prot_diff_name_pairs)

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
            ) / len(structs)
            forces_mae += np.mean(
                np.abs(
                    np.array(structs[name]["final"]["forces"])
                    - np.array(mlip_results[identifier][name]["initial"]["forces"])
                )
            ) / len(structs)
            forces_cosine_similarity += cosine_similarity(
                np.array(structs[name]["final"]["forces"]),
                np.array(mlip_results[identifier][name]["initial"]["forces"]),
            ) / len(structs)
            reopt_rmsd += sdgr_rmsd(
                mlip_results[identifier][name]["initial"]["atoms"],
                mlip_results[identifier][name]["final"]["atoms"],
            ) / len(structs)
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
            deltaE_MLSP_mae += abs(orca_deltaE - deltaE_MLSP) / len(
                one_prot_diff_name_pairs
            )
            deltaE_MLRX_mae += abs(orca_deltaE - deltaE_MLRX) / len(
                one_prot_diff_name_pairs
            )

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
                energy_mae += (
                    abs(
                        orca_results[identifier][charge][spin]["energy"]
                        - mlip_results[identifier][charge][spin]["energy"]
                    )
                    / num_samples
                )
                forces_mae += (
                    np.mean(
                        np.abs(
                            np.array(orca_results[identifier][charge][spin]["forces"])
                            - np.array(mlip_results[identifier][charge][spin]["forces"])
                        )
                    )
                    / num_samples
                )
                forces_cosine_similarity += (
                    cosine_similarity(
                        np.array(orca_results[identifier][charge][spin]["forces"]),
                        np.array(mlip_results[identifier][charge][spin]["forces"]),
                    )
                    / num_samples
                )

        for tag in ["add_electron", "remove_electron"]:
            for spin in orca_deltaE[identifier][tag]:
                deltaE_mae += (
                    abs(
                        orca_deltaE[identifier][tag][spin]
                        - mlip_deltaE[identifier][tag][spin]
                    )
                    / num_deltas
                )
                deltaF_mae += (
                    np.mean(
                        np.abs(
                            orca_deltaF[identifier][tag][spin]
                            - mlip_deltaF[identifier][tag][spin]
                        )
                    )
                    / num_deltas
                )
                deltaF_cosine_similarity += (
                    cosine_similarity(
                        orca_deltaF[identifier][tag][spin],
                        mlip_deltaF[identifier][tag][spin],
                    )
                    / num_deltas
                )

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
    energy_mae = {"sr": 0, "lr": 0}
    forces_mae = {"sr": 0, "lr": 0}
    forces_cosine_similarity = {"sr": 0, "lr": 0}
    deltaE_mae = {"sr": 0, "lr": 0}
    deltaF_mae = {"sr": 0, "lr": 0}
    deltaF_cosine_similarity = {"sr": 0, "lr": 0}
    num_sr = 0
    num_lr = 0
    for vertical in orca_results:
        for identifier in orca_results[vertical]:
            orca_min_energy_name, orca_min_energy_struct = min(
                orca_results[vertical][identifier].items(), key=lambda x: x[1]["energy"]
            )
            num_r = {}
            num_r["sr"] = len(
                [
                    name
                    for name in orca_results[vertical][identifier]
                    if sr_or_lr(name, vertical) == "sr"
                ]
            )
            num_r["lr"] = len(
                [
                    name
                    for name in orca_results[vertical][identifier]
                    if sr_or_lr(name, vertical) == "lr"
                ]
            )
            if num_r["sr"] > 0:
                num_sr += 1
            if num_r["lr"] > 0:
                num_lr += 1
            assert num_r["sr"] + num_r["lr"] == len(orca_results[vertical][identifier])
            for name in orca_results[vertical][identifier]:
                my_range = sr_or_lr(name, vertical)
                energy_mae[my_range] += (
                    abs(
                        orca_results[vertical][identifier][name]["energy"]
                        - mlip_results[vertical][identifier][name]["energy"]
                    )
                    / num_r[my_range]
                )
                forces_mae[my_range] += (
                    np.mean(
                        np.abs(
                            np.array(orca_results[vertical][identifier][name]["forces"])
                            - np.array(
                                mlip_results[vertical][identifier][name]["forces"]
                            )
                        )
                    )
                    / num_r[my_range]
                )
                forces_cosine_similarity[my_range] += (
                    cosine_similarity(
                        np.array(orca_results[vertical][identifier][name]["forces"]),
                        np.array(mlip_results[vertical][identifier][name]["forces"]),
                    )
                    / num_r[my_range]
                )
                if name != orca_min_energy_name:
                    orca_deltaE = (
                        orca_results[vertical][identifier][name]["energy"]
                        - orca_min_energy_struct["energy"]
                    )
                    orca_deltaF = np.array(
                        orca_results[vertical][identifier][name]["forces"]
                    ) - np.array(orca_min_energy_struct["forces"])
                    mlip_deltaE = (
                        mlip_results[vertical][identifier][name]["energy"]
                        - mlip_results[vertical][identifier][orca_min_energy_name][
                            "energy"
                        ]
                    )
                    mlip_deltaF = np.array(
                        mlip_results[vertical][identifier][name]["forces"]
                    ) - np.array(
                        mlip_results[vertical][identifier][orca_min_energy_name][
                            "forces"
                        ]
                    )
                    norm_factor = num_r[my_range] - (
                        1 if my_range == sr_or_lr(orca_min_energy_name, vertical) else 0
                    )
                    deltaE_mae[my_range] += abs(orca_deltaE - mlip_deltaE) / norm_factor
                    deltaF_mae[my_range] += (
                        np.mean(np.abs(orca_deltaF - mlip_deltaF)) / norm_factor
                    )
                    deltaF_cosine_similarity[my_range] += (
                        cosine_similarity(orca_deltaF, mlip_deltaF) / norm_factor
                    )

    results = {
        "sr_energy_mae": energy_mae["sr"] / num_sr,
        "sr_forces_mae": forces_mae["sr"] / num_sr,
        "sr_forces_cosine_similarity": forces_cosine_similarity["sr"] / num_sr,
        "lr_energy_mae": energy_mae["lr"] / num_lr,
        "lr_forces_mae": forces_mae["lr"] / num_lr,
        "lr_forces_cosine_similarity": forces_cosine_similarity["lr"] / num_lr,
        "sr_deltaE_mae": deltaE_mae["sr"] / num_sr,
        "sr_deltaF_mae": deltaF_mae["sr"] / num_sr,
        "sr_deltaF_cosine_similarity": deltaF_cosine_similarity["sr"] / num_sr,
        "lr_deltaE_mae": deltaE_mae["lr"] / num_lr,
        "lr_deltaF_mae": deltaF_mae["lr"] / num_lr,
        "lr_deltaF_cosine_similarity": deltaF_cosine_similarity["lr"] / num_lr,
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
            ) / len(spins)
            forces_mae += np.mean(
                np.abs(
                    np.array(orca_results[identifier][str(spin)]["forces"])
                    - np.array(mlip_results[identifier][str(spin)]["forces"])
                )
            ) / len(spins)
            forces_cosine_similarity += cosine_similarity(
                np.array(orca_results[identifier][str(spin)]["forces"]),
                np.array(mlip_results[identifier][str(spin)]["forces"]),
            ) / len(spins)
            if spin != spins[0]:
                deltaE_mae += abs(
                    orca_deltaE[identifier][spin] - mlip_deltaE[identifier][spin]
                ) / (len(spins) - 1)
                deltaF_mae += np.mean(
                    np.abs(
                        orca_deltaF[identifier][spin] - mlip_deltaF[identifier][spin]
                    )
                ) / (len(spins) - 1)
                deltaF_cosine_similarity += cosine_similarity(
                    orca_deltaF[identifier][spin], mlip_deltaF[identifier][spin]
                ) / (len(spins) - 1)

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
        "forces_cos": cosine_similarity(target_forces, forces),
    }

    return metrics
