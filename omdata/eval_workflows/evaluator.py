from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, ClassVar

import numpy as np
from itertools import permutations

from schrodinger.application.jaguar.autots_bonding import clean_st
from schrodinger.application.jaguar.packages.shared import read_cartesians
from schrodinger.comparison.atom_mapper import ConnectivityAtomMapper
from schrodinger.structure import Structure, StructureWriter
from schrodinger.structutils import rmsd
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.structutils.transform import get_centroid, translate_structure
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment


boltzmann_constant = 8.617333262*10**-5


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


def boltzmann_weighted_structures(results, temp=298.15):
    weighted_families = {}
    for family_identifier, structs in results.items():
        weights = {}
        weighted_structs = {}
        sum = 0
        for conformer_identifier, struct in structs.items():
            weights[conformer_identifier] = np.exp(-struct["energy"] / (boltzmann_constant * temp))
            sum += weights[conformer_identifier]
        for conformer_identifier, struct in structs.items():
            weights[conformer_identifier] /= sum
            if weights[conformer_identifier] > 0.01:
                weighted_structs[conformer_identifier]["atoms"] = struct["atoms"]
                weighted_structs[conformer_identifier]["weight"] = weights[conformer_identifier]
        weighted_families[family_identifier] = weighted_structs
    return weighted_families


def sdgr_rmsd(atoms1, atoms2):
    st1 = Structure()
    st1.set_atoms(atoms1)
    st2 = Structure()
    st2.set_atoms(atoms2)
    renumbered_sts = renumber_molecules_to_match([st1, st2])
    return rmsd_wrapper(renumbered_sts[0], renumbered_sts[1])


def boltzmann_weighted_rmsd(orca_weighted_structs, mlip_weighted_structs):
    cost_matrix = np.array(shape=(len(orca_weighted_structs.keys()), len(mlip_weighted_structs.keys())))
    for ii, o_key in enumerate(orca_weighted_structs.keys()):
        for jj, m_key in enumerate(mlip_weighted_structs.keys()):
            cost_matrix[ii][jj] = abs(orca_weighted_structs[o_key]["weight"] - mlip_weighted_structs[m_key]["weight"]) * sdgr_rmsd(orca_weighted_structs[o_key]["atoms"], mlip_weighted_structs[m_key]["atoms"])
    row_ind, column_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].sum()


def ensemble_rmsd(orca_structs, mlip_structs):
    cost_matrix = np.array(shape=(len(orca_structs.keys()), len(mlip_structs.keys())))
    for ii, o_key in enumerate(orca_structs.keys()):
        for jj, m_key in enumerate(mlip_structs.keys()):
            cost_matrix[ii][jj] = sdgr_rmsd(orca_structs[o_key]["atoms"], mlip_structs[m_key]["atoms"])
    row_ind, column_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].sum()



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
            "deltaE_MLSP": ["mae"],
            "deltaE_MLRX": ["mae"],
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
        boltzmann_rmsd = 0
        ensemble_rmsd = 0
        orca_boltzmann_weighted_structures = boltzmann_weighted_structures(orca_results)
        mlip_boltzmann_weighted_structures = boltzmann_weighted_structures(mlip_results)
        for family_identifier, structs in orca_results.items():
            ensemble_rmsd += ensemble_rmsd(structs, mlip_results[family_identifier])
            boltzmann_rmsd += boltzmann_weighted_rmsd(orca_boltzmann_weighted_structures[family_identifier], mlip_boltzmann_weighted_structures[family_identifier])
            boltzmann_rmsd += boltzmann_weighted_rmsd(mlip_boltzmann_weighted_structures[family_identifier], orca_boltzmann_weighted_structures[family_identifier])
            
        results = {
            "structures": {"ensemble_rmsd": ensemble_rmsd, "boltzmann_weighted_rmsd": boltzmann_rmsd}
        }
        return results


    def geom_conformers_type2(self, orca_results, mlip_results):
        energy_mae = 0
        forces_mae = 0
        forces_cosine_similarity = 0
        deltaE_MLSP_mae = 0
        deltaE_MLRX_mae = 0
        boltzmann_rmsd = 0
        orca_boltzmann_weighted_structures = boltzmann_weighted_structures(orca_results)
        mlip_boltzmann_weighted_structures = boltzmann_weighted_structures(mlip_results)
        for family_identifier, structs in orca_results.items():
            orca_min_energy = float("inf")
            orca_min_energy_id = None
            for conformer_identifier, struct in structs.items():
                if struct["energy"] < orca_min_energy:
                    orca_min_energy = struct["energy"]
                    orca_min_energy_id = conformer_identifier
                energy_mae += abs(struct["energy"] - mlip_results[family_identifier][conformer_identifier]["initial"]["energy"])
                forces_mae += np.mean(np.abs(struct["forces"] - mlip_results[family_identifier][conformer_identifier]["initial"]["forces"]))
                forces_cosine_similarity += cosine_similarity(struct["forces"], mlip_results[family_identifier][conformer_identifier]["initial"]["forces"])
            for conformer_identifier, struct in structs.items():
                orca_deltaE = struct["energy"] - orca_min_energy
                deltaE_MLSP = mlip_results[family_identifier][conformer_identifier]["initial"]["energy"] - mlip_results[family_identifier][orca_min_energy_id]["initial"]["energy"]
                deltaE_MLRX = mlip_results[family_identifier][conformer_identifier]["final"]["energy"] - mlip_results[family_identifier][orca_min_energy_id]["final"]["energy"]
                deltaE_MLSP_mae += abs(orca_deltaE - deltaE_MLSP)
                deltaE_MLRX_mae += abs(orca_deltaE - deltaE_MLRX)
            boltzmann_rmsd += boltzmann_weighted_rmsd(orca_boltzmann_weighted_structures[family_identifier], mlip_boltzmann_weighted_structures[family_identifier])
            boltzmann_rmsd += boltzmann_weighted_rmsd(mlip_boltzmann_weighted_structures[family_identifier], orca_boltzmann_weighted_structures[family_identifier])
                
        results = {
            "energy": {"mae": energy_mae / len(orca_results.keys())},
            "forces": {"mae": forces_mae / len(orca_results.keys()), "cosine_similarity": forces_cosine_similarity / len(orca_results.keys())},
            "deltaE_MLSP": {"mae": deltaE_MLSP_mae / len(orca_results.keys())},
            "deltaE_MLRX": {"mae": deltaE_MLRX_mae / len(orca_results.keys())},
            "structures": {"boltzmann_weighted_rmsd": boltzmann_rmsd}
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