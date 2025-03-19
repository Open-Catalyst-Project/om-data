from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, ClassVar

import numpy as np


def cosine_similarity(forces_1, forces_2):
    return np.dot(forces_1, forces_2) / (np.linalg.norm(forces_1) * np.linalg.norm(forces_2))


def interaction_energy_and_forces(results, principal_identifier):
    interaction_energy = {}
    interaction_forces = {}
    for identifier in results.keys():
        non_principal_forces = None
        non_principal_energy = 0
        for component_identifier in results[identifier].keys():
            if component_identifier != principal_identifier:
                non_principal_energy += results[identifier][component_identifier]["energy"]
                if non_principal_forces is None:
                    non_principal_forces = results[identifier][component_identifier]["forces"]
                else: # Note that the following concatenation assumes that the principal atoms are in the order of each component in sequence
                    non_principal_forces = np.concatenate((non_principal_forces, results[identifier][component_identifier]["forces"]))
        interaction_energy[identifier] = results[identifier][principal_identifier]["energy"] - non_principal_energy
        interaction_forces[identifier] = results[identifier][principal_identifier]["forces"] - non_principal_forces
    return interaction_energy, interaction_forces


def spin_deltas(results):
    deltaE = {}
    deltaF = {}
    for identifier in results.keys():
        deltaE[identifier] = results[identifier]["high_spin"]["energy"] - results[identifier]["low_spin"]["energy"]
        deltaF[identifier] = results[identifier]["high_spin"]["forces"] - results[identifier]["low_spin"]["forces"]
    return deltaE, deltaF


def charge_deltas(results):
    deltaE = {}
    deltaF = {}
    for identifier in results.keys():
        deltaE[identifier] = {"add_electron": {}, "remove_electron": {}}
        deltaF[identifier] = {"add_electron": {}, "remove_electron": {}}
        orig_energy = results[identifier]["original"][results[identifier]["original"].keys()[0]]["energy"]
        orig_forces = results[identifier]["original"][results[identifier]["original"].keys()[0]]["forces"]
        for tag in ["add_electron", "remove_electron"]:
            for spin_multiplicity in results[identifier][tag].keys():
                deltaE[identifier][tag][spin_multiplicity] = results[identifier][tag][spin_multiplicity]["energy"] - orig_energy
                deltaF[identifier][tag][spin_multiplicity] = results[identifier][tag][spin_multiplicity]["forces"] - orig_forces
    return deltaE, deltaF


# The use of task_metrics and eval are both mockups and will need help to function as envisioned
class OMol_Evaluator:
    task_metrics = {
        "ligand_pocket": {
            "energy": ["mae"],
            "forces": ["mae", "cosine_similarity"],
            "interaction_energy": ["mae"],
            "interaction_forces": ["mae", "cosine_similarity"]
        },
        "ligand_strain": {
        },
        "geom_conformers_type1": {
            "structures": ["ensemble_rmsd", "boltzmann_weighted_rmsd"],
        },
        "geom_conformers_type2": {
            "energy": ["mae"],
            "forces": ["mae", "cosine_similarity"],
            "deltaE": ["mae"],
            "structures": ["boltzmann_weighted_rmsd"],
        },
        "protonation_energies": {
        },
        "unoptimized_ie_ea": {
            "energy": ["mae"],
            "forces": ["mae", "cosine_similarity"],
            "deltaE": ["mae"],
            "deltaF": ["mae", "cosine_similarity"]
        },
        "distance_scaling": {
            "energy": ["mae"],
            "forces": ["mae", "cosine_similarity"],
            "interaction_energy": ["mae"],
            "interaction_forces": ["mae", "cosine_similarity"]
        },
        "unoptimized_spin_gap": {
            "energy": ["mae"],
            "forces": ["mae", "cosine_similarity"],
            "deltaE": ["mae"],
            "deltaF": ["mae", "cosine_similarity"]
        }
    }


    def __init__(self, orca_json, ml_json):
        self.orca_results = loadfn(orca_json)
        self.ml_results = loadfn(ml_json)
        self.eval_metrics = {}
    

    def eval(self):
        for task in self.task_metrics.keys():
            self.eval_metrics[task] = eval(task)(self.orca_results[task], self.ml_results[task])
        return self.eval_metrics


    def ligand_pocket(self, orca_results, mlip_results):
        energy_mae = 0
        forces_mae = 0
        forces_cosine_similarity = 0
        interaction_energy_mae = 0
        interaction_forces_mae = 0
        interaction_forces_cosine_similarity = 0
        orca_interaction_energy, orca_interaction_forces = interaction_energy_and_forces(orca_results, "ligand_pocket")
        mlip_interaction_energy, mlip_interaction_forces = interaction_energy_and_forces(mlip_results, "ligand_pocket")
        for identifier in orca_results.keys():
            for component_identifier in orca_results[identifier].keys():
                energy_mae += abs(orca_results[identifier][component_identifier]["energy"] - mlip_results[identifier][component_identifier]["energy"])
                forces_mae += np.mean(np.abs(orca_results[identifier][component_identifier]["forces"] - mlip_results[identifier][component_identifier]["forces"]))
                forces_cosine_similarity += cosine_similarity(orca_results[identifier][component_identifier]["forces"], mlip_results[identifier][component_identifier]["forces"])
            interaction_energy_mae += abs(orca_interaction_energy[identifier] - mlip_interaction_energy[identifier])
            interaction_forces_mae += np.mean(np.abs(orca_interaction_forces[identifier] - mlip_interaction_forces[identifier]))
            interaction_forces_cosine_similarity += cosine_similarity(orca_interaction_forces[identifier], mlip_interaction_forces[identifier])

        results = {
            "energy": {"mae": energy_mae / len(orca_results.keys())},
            "forces": {"mae": forces_mae / len(orca_results.keys()), "cosine_similarity": forces_cosine_similarity / len(orca_results.keys())},
            "interaction_energy": {"mae": interaction_energy_mae / len(orca_results.keys())},
            "interaction_forces": {"mae": interaction_forces_mae / len(orca_results.keys()), "cosine_similarity": interaction_forces_cosine_similarity / len(orca_results.keys())}
        }
        return results


    def ligand_strain(self, orca_results, mlip_results):
        pass


    def geom_conformers_type1(self, orca_results, mlip_results):
        pass


    def geom_conformers_type2(self, orca_results, mlip_results):
        energy_mae = 0
        forces_mae = 0
        forces_cosine_similarity = 0
        for family_identifier, structs in orca_results.items():
            dft_min_energy = float("inf")
            dft_min_energy_id = None
            for conformer_identifier, struct in structs.items():
                if struct["energy"] < dft_min_energy:
                    dft_min_energy = struct["energy"]
                    dft_min_energy_id = conformer_identifier
                energy_mae += abs(struct["energy"] - mlip_results[family_identifier][conformer_identifier]["initial"]["energy"])
                forces_mae += np.mean(np.abs(struct["forces"] - mlip_results[family_identifier][conformer_identifier]["initial"]["forces"]))
                forces_cosine_similarity += cosine_similarity(struct["forces"], mlip_results[family_identifier][conformer_identifier]["initial"]["forces"])
            for conformer_identifier, struct in structs.items():
                

        results = {
            "energy": {"mae": energy_mae / len(orca_results.keys())},
            "forces": {"mae": forces_mae / len(orca_results.keys()), "cosine_similarity": forces_cosine_similarity / len(orca_results.keys())}
        }
        return results


    def protonation_energies(self, orca_results, mlip_results):
        pass


    def unoptimized_ie_ea(self, orca_results, mlip_results):
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
                for spin_multiplicity in orca_results[identifier][tag].keys():
                    energy_mae += abs(orca_results[identifier][tag][spin_multiplicity]["energy"] - mlip_results[identifier][tag][spin_multiplicity]["energy"])
                    forces_mae += np.mean(np.abs(orca_results[identifier][tag][spin_multiplicity]["forces"] - mlip_results[identifier][tag][spin_multiplicity]["forces"]))
                    forces_cosine_similarity += cosine_similarity(orca_results[identifier][tag][spin_multiplicity]["forces"], ml_results[identifier][tag][spin_multiplicity]["forces"])
                    if tag != "original":
                        deltaE_mae += abs(orca_deltaE[identifier][tag][spin_multiplicity] - mlip_deltaE[identifier][tag][spin_multiplicity])
                        deltaF_mae += np.mean(np.abs(orca_deltaF[identifier][tag][spin_multiplicity] - mlip_deltaF[identifier][tag][spin_multiplicity]))
                        deltaF_cosine_similarity += cosine_similarity(orca_deltaF[identifier][tag][spin_multiplicity], mlip_deltaF[identifier][tag][spin_multiplicity])

        results = {
            "energy": {"mae": energy_mae / len(orca_results.keys())},
            "forces": {"mae": forces_mae / len(orca_results.keys()), "cosine_similarity": forces_cosine_similarity / len(orca_results.keys())},
            "deltaE": {"mae": deltaE_mae / len(orca_results.keys())},
            "deltaF": {"mae": deltaF_mae / len(orca_results.keys()), "cosine_similarity": deltaF_cosine_similarity / len(orca_results.keys())}
        }
        return results


    def distance_scaling(self, orca_results, mlip_results):
        energy_mae = 0
        forces_mae = 0
        forces_cosine_similarity = 0
        interaction_energy_mae = 0
        interaction_forces_mae = 0
        interaction_forces_cosine_similarity = 0
        orca_interaction_energy, orca_interaction_forces = interaction_energy_and_forces(orca_results, "complex")
        mlip_interaction_energy, mlip_interaction_forces = interaction_energy_and_forces(mlip_results, "complex")
        for identifier in orca_results.keys():
            for component_identifier in orca_results[identifier].keys():
                energy_mae += abs(orca_results[identifier][component_identifier]["energy"] - mlip_results[identifier][component_identifier]["energy"])
                forces_mae += np.mean(np.abs(orca_results[identifier][component_identifier]["forces"] - mlip_results[identifier][component_identifier]["forces"]))
                forces_cosine_similarity += cosine_similarity(orca_results[identifier][component_identifier]["forces"], mlip_results[identifier][component_identifier]["forces"])
            interaction_energy_mae += abs(orca_interaction_energy[identifier] - mlip_interaction_energy[identifier])
            interaction_forces_mae += np.mean(np.abs(orca_interaction_forces[identifier] - mlip_interaction_forces[identifier]))
            interaction_forces_cosine_similarity += cosine_similarity(orca_interaction_forces[identifier], mlip_interaction_forces[identifier])

        results = {
            "energy": {"mae": energy_mae / len(orca_results.keys())},
            "forces": {"mae": forces_mae / len(orca_results.keys()), "cosine_similarity": forces_cosine_similarity / len(orca_results.keys())},
            "interaction_energy": {"mae": interaction_energy_mae / len(orca_results.keys())},
            "interaction_forces": {"mae": interaction_forces_mae / len(orca_results.keys()), "cosine_similarity": interaction_forces_cosine_similarity / len(orca_results.keys())}
        }
        return results


    def unoptimized_spin_gap(self, orca_results, mlip_results):
        energy_mae = 0
        forces_mae = 0
        forces_cosine_similarity = 0
        deltaE_mae = 0
        deltaF_mae = 0
        deltaF_cosine_similarity = 0
        orca_deltaE, orca_deltaF = spin_deltas(orca_results)
        mlip_deltaE, mlip_deltaF = spin_deltas(mlip_results)
        for identifier in orca_results.keys():
            for key in ["high_spin", "low_spin"]:
                energy_mae += abs(orca_results[identifier][key]["energy"] - mlip_results[identifier][key]["energy"])
                forces_mae += np.mean(np.abs(orca_results[identifier][key]["forces"] - mlip_results[identifier][key]["forces"]))
                forces_cosine_similarity += cosine_similarity(orca_results[identifier][key]["forces"], mlip_results[identifier][key]["forces"])
            deltaE_mae += abs(orca_deltaE[identifier] - mlip_deltaE[identifier])
            deltaF_mae += np.mean(np.abs(orca_deltaF[identifier] - mlip_deltaF[identifier]))
            deltaF_cosine_similarity += cosine_similarity(orca_deltaF[identifier], mlip_deltaF[identifier])

        results = {
            "energy": {"mae": energy_mae / len(orca_results.keys())},
            "forces": {"mae": forces_mae / len(orca_results.keys()), "cosine_similarity": forces_cosine_similarity / len(orca_results.keys())},
            "deltaE": {"mae": deltaE_mae / len(orca_results.keys())},
            "deltaF": {"mae": deltaF_mae / len(orca_results.keys()), "cosine_similarity": deltaF_cosine_similarity / len(orca_results.keys())}
        }
        return results