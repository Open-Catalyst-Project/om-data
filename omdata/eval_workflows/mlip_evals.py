from typing import Optional, Union, List, Literal
from pathlib import Path
from ase.atoms import Atoms
from quacc.recipes.orca._base import prep_calculator
import os
import psutil
import tqdm
from quacc import job
from quacc.runners.ase import Runner
from quacc.schemas.ase import Summarize, VibSummarize
from quacc.utils.dicts import recursive_dict_merge
from quacc.atoms.core import perturb
from monty.serialization import dumpfn

from omdata.orca.calc import (
    TIGHT_OPT_PARAMETERS,
    OPT_PARAMETERS,
    LOOSE_OPT_PARAMETERS,
)

from omdata.eval_workflows.eval_jobs import ase_calc_single_point_job, ase_calc_relax_job

# NOTE: We assume that all atoms objects will have charge and spin in the atoms.info dictionary


# key: identifier, value: {
#                "ligand_pocket": Atoms,
#                "ligand": Atoms, 
#                "pocket": Atoms
#                }
ligand_pocket_structures = {}
# NOTE: ligand and pocket should have mask info in atoms.info to map their atoms to the ligand_pocket atoms

# key: family_identifier, value: {
#                "ligand_in_pocket": Atoms,
#                "conformer_0_identifier": Atoms,
#                "conformer_1_identifier": Atoms, 
#                ...
#                }
ligand_strain_structures = {}

# key: family_identifier, value: {
#                "conformer_0_identifier": Atoms,
#                "conformer_1_identifier": Atoms, 
#                ...
#                }
geom_conformers_structures_type1 = {}

# key: family_identifier, value: {
#                "conformer_0_identifier": Atoms,
#                "conformer_1_identifier": Atoms, 
#                ...
#                }
geom_conformers_structures_type2 = {}
# NOTE: these structures should have already been optimized by DFT

# key: identifier, value: {
#                "name1": Atoms, 
#                "name2": Atoms, 
#                ...
#                }
protonation_structures_type1 = {}

# key: identifier, value: {
#                "name1": Atoms, 
#                "name2": Atoms, 
#                ...
#                }
protonation_structures_type2 = {}
# NOTE: these structures should have already been optimized by DFT

# key: identifier, value: {
#                    charge0: {
#                        "spin0": Atoms,
#                        ...
#                    },
#                    charge1: {
#                        "spin0": Atoms,
#                        ...
#                    },
#                    charge2: {
#                        "spin0": Atoms,
#                        ...
#                    },
#                }
unoptimized_ie_ea_structures = {}

# key: identifier, value: { 
#                scaling_value0: Atoms,
#                scaling_value1: Atoms,
#                ...
#                }
distance_scaling_structures = {}

# key: identifier, value: {
#                spin0: Atoms,
#                ...
#                }
unoptimized_spin_gap_structures = {}


def format_relax_result(result):
    initial_atoms = result["trajectory"][0]
    initial_atoms.info["charge"] = result["charge"]
    initial_atoms.info["spin"] = result["spin_multiplicity"]
    final_atoms = result["atoms"]
    final_atoms.info["charge"] = result["charge"]
    final_atoms.info["spin"] = result["spin_multiplicity"]
    return {"initial": {"atoms": initial_atoms, "energy": result["trajectory_results"][0]["energy"], "forces": result["trajectory_results"][0]["forces"]}, 
            "final": {"atoms": final_atoms, "energy": result["results"]["energy"], "forces": result["results"]["forces"]},
    }


def format_single_point_result(result):
    atoms = result["atoms"]
    atoms.info["charge"] = result["charge"]
    atoms.info["spin"] = result["spin_multiplicity"]
    return {"atoms": atoms, "energy": result["results"]["energy"], "forces": result["results"]["forces"]}


def mlip_ligand_pocket(calc, ligand_pocket_structures, results_directory=None):
    results = {}
    for identifier, structs in tqdm.tqdm(ligand_pocket_structures.items()):
        lp_result = format_single_point_result(ase_calc_single_point_job(calc, structs["ligand_pocket"], structs["ligand_pocket"].info["charge"], structs["ligand_pocket"].info["spin"]))
        l_result = format_single_point_result(ase_calc_single_point_job(calc, structs["ligand"], structs["ligand"].info["charge"], structs["ligand"].info["spin"]))
        p_result = format_single_point_result(ase_calc_single_point_job(calc, structs["pocket"], structs["pocket"].info["charge"], structs["pocket"].info["spin"]))
        results[identifier] = {"ligand_pocket": lp_result, "ligand": l_result, "pocket": p_result}

    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, "mlip_ligand_pocket.json"))
    else:
        return results


def mlip_ligand_strain(calc, ligand_strain_structures, results_directory=None):
    results = {}
    for family_identifier, structs in tqdm.tqdm(ligand_strain_structures.items()):
        family_results = {}
        for conformer_identifier, struct in structs.items():
            if conformer_identifier == "ligand_in_pocket":
                result = ase_calc_relax_job(calc, struct, struct.info["charge"], struct.info["spin"], opt_params=LOOSE_OPT_PARAMETERS)
            else:
                result = ase_calc_relax_job(calc, struct, struct.info["charge"], struct.info["spin"], opt_params=TIGHT_OPT_PARAMETERS)
            family_results[conformer_identifier] = format_relax_result(result)
        results[family_identifier] = family_results

    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, "mlip_ligand_strain.json"))
    else:
        return results


def mlip_geom_conformers(calc, geom_conformers_structures, conf_task_type, results_directory=None):
    results = {}
    for family_identifier, structs in tqdm.tqdm(geom_conformers_structures.items()):
        family_results = {}
        print(family_identifier)
        for conformer_identifier, struct in structs.items():
            result = ase_calc_relax_job(calc, struct, struct.info["charge"], struct.info["spin"], opt_params=TIGHT_OPT_PARAMETERS)
            family_results[conformer_identifier] = format_relax_result(result)
        results[family_identifier] = family_results

    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, f"mlip_geom_conformers_{conf_task_type}.json"))
    else:
        return results


def mlip_protonation_energies(calc, protonation_structures, results_directory=None):
    results = {}
    for identifier, structs in tqdm.tqdm(protonation_structures.items()):
        family_results = {}
        for name in structs.keys():
            result = ase_calc_relax_job(calc, structs[name], structs[name].info["charge"], structs[name].info["spin"], opt_params=TIGHT_OPT_PARAMETERS)
            family_results[name] = format_relax_result(result)
        results[identifier] = family_results

    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, "mlip_protonation_energies.json"))
    else:
        return results


def mlip_unoptimized_ie_ea(calc, unoptimized_ie_ea_structures, results_directory=None):
    results = {}
    for identifier, structs in tqdm.tqdm(unoptimized_ie_ea_structures.items()):
        results[identifier] = {} 
        for charge in structs.keys():
            results[identifier][charge] = {}
            for spin in structs[charge].keys():
                assert structs[charge][spin].info["charge"] == charge
                assert structs[charge][spin].info["spin"] == spin
                results[identifier][charge][spin] = format_single_point_result(ase_calc_single_point_job(calc, structs[charge][spin], charge, spin))
    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, "mlip_unoptimized_ie_ea.json"))
    else:
        return results


def mlip_distance_scaling(calc, distance_scaling_structures, results_directory=None):
    results = {}
    for identifier, structs in tqdm.tqdm(distance_scaling_structures.items()):
        results[identifier] = {}
        for scaling_value, struct in structs.items():
            results[identifier][scaling_value] = format_single_point_result(ase_calc_single_point_job(calc, struct, struct.info["charge"], struct.info["spin"]))

    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, "mlip_distance_scaling.json"))
    else:
        return results


def mlip_unoptimized_spin_gap(calc, unoptimized_spin_gap_structures, results_directory=None):
    results = {}
    for identifier, structs in tqdm.tqdm(unoptimized_spin_gap_structures.items()):
        results[identifier] = {}
        for spin in structs.keys():
            assert structs[spin].info["spin"] == spin
            results[identifier][spin] = format_single_point_result(ase_calc_single_point_job(calc, structs[spin], structs[spin].info["charge"], structs[spin].info["spin"]))

    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, "mlip_unoptimized_spin_gap.json"))
    else:
        return results


def assemble_mlip_results(results_directory=None):
    results = {}
    if results_directory is None: # Run all the tasks in series
        results["ligand_pocket"] = mlip_ligand_pocket(ligand_pocket_structures)
        results["ligand_strain"] = mlip_ligand_strain(ligand_strain_structures)
        results["geom_conformers_type1"] = mlip_geom_conformers(geom_conformers_structures_type1, "type1")
        results["geom_conformers_type2"] = mlip_geom_conformers(geom_conformers_structures_type2, "type2")
        results["protonation_energies_type1"] = mlip_protonation_energies(protonation_structures_type1)
        results["protonation_energies_type2"] = mlip_protonation_energies(protonation_structures_type2)
        results["unoptimized_ie_ea"] = mlip_unoptimized_ie_ea(unoptimized_ie_ea_structures)
        results["distance_scaling"] = mlip_distance_scaling(distance_scaling_structures)
        results["unoptimized_spin_gap"] = mlip_unoptimized_spin_gap(unoptimized_spin_gap_structures)
    else: # Assume that all tasks have previously been run, likely in parallel, with results dumped, so load the results
        results_directory = Path(results_directory)
        results["ligand_pocket"] = loadfn(os.path.join(results_directory, "mlip_ligand_pocket.json"))
        results["ligand_strain"] = loadfn(os.path.join(results_directory, "mlip_ligand_strain.json"))
        results["geom_conformers_type1"] = loadfn(os.path.join(results_directory, "mlip_geom_conformers_type1.json"))
        results["geom_conformers_type2"] = loadfn(os.path.join(results_directory, "mlip_geom_conformers_type2.json"))
        results["protonation_energies_type1"] = loadfn(os.path.join(results_directory, "mlip_protonation_energies_type1.json"))
        results["protonation_energies_type2"] = loadfn(os.path.join(results_directory, "mlip_protonation_energies_type2.json"))
        results["unoptimized_ie_ea"] = loadfn(os.path.join(results_directory, "mlip_unoptimized_ie_ea.json"))
        results["distance_scaling"] = loadfn(os.path.join(results_directory, "mlip_distance_scaling.json"))
        results["unoptimized_spin_gap"] = loadfn(os.path.join(results_directory, "mlip_unoptimized_spin_gap.json"))

    dumpfn(results, os.path.join(results_directory, "mlip_results.json"))