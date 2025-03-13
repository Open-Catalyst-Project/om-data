from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, ClassVar

import numpy as np


def cosine_similarity(forces_1, forces_2):
    return np.dot(forces_1, forces_2) / (np.linalg.norm(forces_1) * np.linalg.norm(forces_2))


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
        "geom_conformers": {
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
        for identifier in orca_results.keys():
            for component_identifier in orca_results[identifier].keys():
                energy_mae += abs(orca_results[identifier][component_identifier]["energy"] - mlip_results[identifier][component_identifier]["energy"])
                forces_mae += np.mean(np.abs(orca_results[identifier][component_identifier]["forces"] - mlip_results[identifier][component_identifier]["forces"]))
                forces_cosine_similarity += cosine_similarity(orca_results[identifier][component_identifier]["forces"], mlip_results[identifier][component_identifier]["forces"])
            orca_interaction_energy = orca_results[identifier]["ligand_pocket"]["energy"] - orca_results[identifier]["ligand"]["energy"] - orca_results[identifier]["pocket"]["energy"]
            mlip_interaction_energy = mlip_results[identifier]["ligand_pocket"]["energy"] - mlip_results[identifier]["ligand"]["energy"] - mlip_results[identifier]["pocket"]["energy"]
            interaction_energy_mae += abs(orca_interaction_energy - mlip_interaction_energy)
            # Note that the following concatenation assumes that ligand_pocket atom indices are in the order of ligand followed by pocket
            orca_interaction_forces = orca_results[identifier]["ligand_pocket"]["forces"] - np.concatenate((orca_results[identifier]["ligand"]["forces"], orca_results[identifier]["pocket"]["forces"]))
            mlip_interaction_forces = mlip_results[identifier]["ligand_pocket"]["forces"] - np.concatenate((mlip_results[identifier]["ligand"]["forces"], mlip_results[identifier]["pocket"]["forces"]))
            interaction_forces_mae += np.mean(np.abs(orca_interaction_forces - mlip_interaction_forces))
            interaction_forces_cosine_similarity += cosine_similarity(orca_interaction_forces, mlip_interaction_forces)
        results = {
            "energy": {"mae": energy_mae / len(orca_results.keys())},
            "forces": {"mae": forces_mae / len(orca_results.keys()), "cosine_similarity": forces_cosine_similarity / len(orca_results.keys())},
            "interaction_energy": {"mae": interaction_energy_mae / len(orca_results.keys())},
            "interaction_forces": {"mae": interaction_forces_mae / len(orca_results.keys()), "cosine_similarity": interaction_forces_cosine_similarity / len(orca_results.keys())}
        }
        return results


    def ligand_strain(self, orca_results, mlip_results):
        pass


    def geom_conformers(self, orca_results, mlip_results):
        pass


    def protonation_energies(self, orca_results, mlip_results):
        pass


    def unoptimized_ie_ea(self, orca_results, mlip_results):
        energy_mae = 0
        forces_mae = 0
        forces_cosine_similarity = 0
        deltaE_mae = 0
        deltaF_mae = 0
        deltaF_cosine_similarity = 0
        for identifier in orca_results.keys():
            for tag in ["original", "add_electron", "remove_electron"]:
                for spin_multiplicity in orca_results[identifier][tag].keys():
                    energy_mae += abs(orca_results[identifier][tag][spin_multiplicity]["energy"] - ml_results[identifier][tag][spin_multiplicity]["energy"])
                    forces_mae += np.mean(np.abs(orca_results[identifier][tag][spin_multiplicity]["forces"] - ml_results[identifier][tag][spin_multiplicity]["forces"]))
                    forces_cosine_similarity += cosine_similarity(orca_results[identifier][tag][spin_multiplicity]["forces"], ml_results[identifier][tag][spin_multiplicity]["forces"])
            
            orca_orig_energy = orca_results[identifier]["original"][orca_results[identifier]["original"].keys()[0]]["energy"]
            orca_orig_forces = orca_results[identifier]["original"][orca_results[identifier]["original"].keys()[0]]["forces"]
            mlip_orig_energy = ml_results[identifier]["original"][mlip_results[identifier]["original"].keys()[0]]["energy"]
            mlip_orig_forces = ml_results[identifier]["original"][mlip_results[identifier]["original"].keys()[0]]["forces"]
            for spin_multiplicity in orca_results[identifier]["add_electron"].keys():
                orca_deltaE = orca_results[identifier]["add_electron"][spin_multiplicity]["energy"] - orca_orig_energy
                orca_deltaF = orca_results[identifier]["add_electron"][spin_multiplicity]["forces"] - orca_orig_forces
                mlip_deltaE = ml_results[identifier]["add_electron"][spin_multiplicity]["energy"] - mlip_orig_energy
                mlip_deltaF = ml_results[identifier]["add_electron"][spin_multiplicity]["forces"] - mlip_orig_forces
                deltaE_mae += abs(orca_deltaE - mlip_deltaE)
                deltaF_mae += np.mean(np.abs(orca_deltaF - mlip_deltaF))
                deltaF_cosine_similarity += cosine_similarity(orca_deltaF, mlip_deltaF)

            for spin_multiplicity in orca_results[identifier]["remove_electron"].keys():
                orca_deltaE = orca_results[identifier]["remove_electron"][spin_multiplicity]["energy"] - orca_orig_energy
                orca_deltaF = orca_results[identifier]["remove_electron"][spin_multiplicity]["forces"] - orca_orig_forces
                mlip_deltaE = ml_results[identifier]["remove_electron"][spin_multiplicity]["energy"] - mlip_orig_energy
                mlip_deltaF = ml_results[identifier]["remove_electron"][spin_multiplicity]["forces"] - mlip_orig_forces
                deltaE_mae += abs(orca_deltaE - mlip_deltaE)
                deltaF_mae += np.mean(np.abs(orca_deltaF - mlip_deltaF))
                deltaF_cosine_similarity += cosine_similarity(orca_deltaF, mlip_deltaF)

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
        for identifier in orca_results.keys():
            orca_non_complex_forces = None
            orca_non_complex_energy = 0
            mlip_non_complex_forces = None
            mlip_non_complex_energy = 0
            for component_identifier in orca_results[identifier].keys():
                energy_mae += abs(orca_results[identifier][component_identifier]["energy"] - mlip_results[identifier][component_identifier]["energy"])
                forces_mae += np.mean(np.abs(orca_results[identifier][component_identifier]["forces"] - mlip_results[identifier][component_identifier]["forces"]))
                forces_cosine_similarity += cosine_similarity(orca_results[identifier][component_identifier]["forces"], mlip_results[identifier][component_identifier]["forces"])
                if component_identifier != "complex":
                    orca_non_complex_energy += orca_results[identifier][component_identifier]["energy"]
                    mlip_non_complex_energy += mlip_results[identifier][component_identifier]["energy"]
                    if orca_non_complex_forces is None:
                        orca_non_complex_forces = orca_results[identifier][component_identifier]["forces"]
                        mlip_non_complex_forces = mlip_results[identifier][component_identifier]["forces"]
                    else: # Note that the following concatenation assumes that the complex atoms are in the order of each component in sequence
                        orca_non_complex_forces = np.concatenate((orca_non_complex_forces, orca_results[identifier][component_identifier]["forces"]))
                        mlip_non_complex_forces = np.concatenate((mlip_non_complex_forces, mlip_results[identifier][component_identifier]["forces"]))
            orca_interaction_energy = orca_results[identifier]["complex"]["energy"] - orca_non_complex_energy
            mlip_interaction_energy = mlip_results[identifier]["complex"]["energy"] - mlip_non_complex_energy
            orca_interaction_forces = orca_results[identifier]["complex"]["forces"] - orca_non_complex_forces
            mlip_interaction_forces = mlip_results[identifier]["complex"]["forces"] - mlip_non_complex_forces
            interaction_energy_mae += abs(orca_interaction_energy - mlip_interaction_energy)
            interaction_forces_mae += np.mean(np.abs(orca_interaction_forces - mlip_interaction_forces))
            interaction_forces_cosine_similarity += cosine_similarity(orca_interaction_forces, mlip_interaction_forces)

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
        for identifier in orca_results.keys():
            for key in ["high_spin", "low_spin"]:
                energy_mae += abs(orca_results[identifier][key]["energy"] - mlip_results[identifier][key]["energy"])
                forces_mae += np.mean(np.abs(orca_results[identifier][key]["forces"] - mlip_results[identifier][key]["forces"]))
                forces_cosine_similarity += cosine_similarity(orca_results[identifier][key]["forces"], mlip_results[identifier][key]["forces"])
            orca_deltaE = orca_results[identifier]["high_spin"]["energy"] - orca_results[identifier]["low_spin"]["energy"]
            mlip_deltaE = mlip_results[identifier]["high_spin"]["energy"] - mlip_results[identifier]["low_spin"]["energy"]
            deltaE_mae += abs(orca_deltaE - mlip_deltaE)
            orca_deltaF = orca_results[identifier]["high_spin"]["forces"] - orca_results[identifier]["low_spin"]["forces"]
            mlip_deltaF = mlip_results[identifier]["high_spin"]["forces"] - mlip_results[identifier]["low_spin"]["forces"]
            deltaF_mae += np.mean(np.abs(orca_deltaF - mlip_deltaF))
            deltaF_cosine_similarity += cosine_similarity(orca_deltaF, mlip_deltaF)

        results = {
            "energy": {"mae": energy_mae / len(orca_results.keys())},
            "forces": {"mae": forces_mae / len(orca_results.keys()), "cosine_similarity": forces_cosine_similarity / len(orca_results.keys())},
            "deltaE": {"mae": deltaE_mae / len(orca_results.keys())},
            "deltaF": {"mae": deltaF_mae / len(orca_results.keys()), "cosine_similarity": deltaF_cosine_similarity / len(orca_results.keys())}
        }
        return results