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
    EVAL_TS_PARAMETERS,
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
#                "unprotonated": Atoms, 
#                "protonated": Atoms, 
#                }
protonation_structures_type1 = {}

# key: identifier, value: {
#                "unprotonated": Atoms, 
#                "protonated": Atoms, 
#                }
protonation_structures_type2 = {}
# NOTE: these structures should have already been optimized by DFT

# key: identifier, value: {
#                "orig": Atoms,
#                "add_electron_spins": [spin0, ...],
#                "remove_electron_spins": [spin0, ...]
#                }
unoptimized_ie_ea_structures = {}

# key: identifier, value: { 
#                "scaled_complex_0": Atoms,
#                "scaled_complex_1": Atoms,
#                ...
#                "component_0": Atoms,
#                "component_1": Atoms,
#                ...
#                }
distance_scaling_structures = {}
# NOTE: component atoms should have mask info in atoms.info to map their atoms to the complex atoms

# key: identifier, value: {
#                "orig": Atoms,
#                "additional_spins": [spin0, ...],
#                }
unoptimized_spin_gap_structures = {}
# NOTE: structures should all be "high spin", i.e. spin_multiplicity > 2


def mlip_ligand_pocket(ligand_pocket_structures, results_directory=None):
    results = {}
    for identifier, structs in tqdm.tqdm(ligand_pocket_structures.items()):
        lp_calc = prep_mlip_calc(structs["ligand_pocket"].info["charge"], structs["ligand_pocket"].info["spin"])
        l_calc = prep_mlip_calc(structs["ligand"].info["charge"], structs["ligand"].info["spin"])
        p_calc = prep_mlip_calc(structs["pocket"].info["charge"], structs["pocket"].info["spin"])

        lp_result = ase_calc_single_point_job(lp_calc, structs["ligand_pocket"], structs["ligand_pocket"].info["charge"], structs["ligand_pocket"].info["spin"])
        l_result = ase_calc_single_point_job(l_calc, structs["ligand"], structs["ligand"].info["charge"], structs["ligand"].info["spin"])
        p_result = ase_calc_single_point_job(p_calc, structs["pocket"], structs["pocket"].info["charge"], structs["pocket"].info["spin"])

        results[identifier] = {"ligand_pocket": lp_result, "ligand": l_result, "pocket": p_result}
        # TODO: Confirm that mask info is passed through correctly

    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, "mlip_ligand_pocket.json"))
    else:
        return results


def mlip_ligand_strain(ligand_strain_structures, results_directory=None):
    results = {}
    for family_identifier, structs in tqdm.tqdm(ligand_strain_structures.items()):
        family_results = {}
        for conformer_identifier, struct in structs.items():
            calc = prep_mlip_calc(struct.info["charge"], struct.info["spin"])
            if conformer_identifier == "ligand_in_pocket":
                result = ase_calc_relax_job(calc, struct, struct.info["charge"], struct.info["spin"], opt_params=LOOSE_OPT_PARAMETERS)
                # TODO: Save the first structure and the last structure, but can jettison the rest to avoid large files
            else:
                result = ase_calc_relax_job(calc, struct, struct.info["charge"], struct.info["spin"], opt_params=TIGHT_OPT_PARAMETERS)
                # TODO: Save the first structure and the last structure, but can jettison the rest to avoid large files
            family_results[conformer_identifier] = result
        results[family_identifier] = family_results

    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, "mlip_ligand_strain.json"))
    else:
        return results


def mlip_geom_conformers(geom_conformers_structures, conf_task_type, results_directory=None):
    results = {}
    for family_identifier, structs in tqdm.tqdm(geom_conformers_structures.items()):
        family_results = {}
        for conformer_identifier, struct in structs.items():
            calc = prep_mlip_calc(struct.info["charge"], struct.info["spin"])
            result = ase_calc_relax_job(calc, struct, struct.info["charge"], struct.info["spin"], opt_params=TIGHT_OPT_PARAMETERS)
            # TODO: Save the first structure and the last structure, but can jettison the rest to avoid large files
            family_results[conformer_identifier] = result
        results[family_identifier] = family_results

    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, f"mlip_geom_conformers_{conf_task_type}.json"))
    else:
        return results


def mlip_protonation_energies(protonation_structures, results_directory=None):
    results = {}
    for identifier, structs in tqdm.tqdm(protonation_structures.items()):
        family_results = {}
        for protonation_state in ["unprotonated", "protonated"]:
            calc = prep_mlip_calc(structs[protonation_state].info["charge"], structs[protonation_state].info["spin"])
            result = ase_calc_relax_job(calc, structs[protonation_state], structs[protonation_state].info["charge"], structs[protonation_state].info["spin"], opt_params=TIGHT_OPT_PARAMETERS)
            # TODO: Save the first structure and the last structure, but can jettison the rest to avoid large files
            family_results[protonation_state] = result
        results[identifier] = family_results

    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, "mlip_protonation_energies.json"))
    else:
        return results


def mlip_unoptimized_ie_ea(unoptimized_ie_ea_structures, results_directory=None):
    results = {}
    for identifier, struct in tqdm.tqdm(unoptimized_ie_ea_structures.items()):
        calc_orig = prep_mlip_calc(struct["orig"].info["charge"], struct["orig"].info["spin"])
        results[identifier] = {"original": ase_calc_single_point_job(calc_orig, struct["orig"], struct["orig"].info["charge"], struct["orig"].info["spin"]),
                               "add_electron": {},
                               "remove_electron": {},
        } 
        for spin in struct["add_electron_spins"]:
            calc = prep_mlip_calc(struct["orig"].info["charge"] - 1, spin)
            results[identifier]["add_electron"][spin] = ase_calc_single_point_job(calc, struct["orig"], struct["orig"].info["charge"] - 1, spin)
        for spin in struct["remove_electron_spins"]:
            calc = prep_mlip_calc(struct["orig"].info["charge"] + 1, spin)
            results[identifier]["remove_electron"][spin] = ase_calc_single_point_job(calc, struct["orig"], struct["orig"].info["charge"] + 1, spin)

    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, "mlip_unoptimized_ie_ea.json"))
    else:
        return results


def mlip_distance_scaling(distance_scaling_structures, results_directory=None):
    results = {}
    for identifier, structs in tqdm.tqdm(distance_scaling_structures.items()):
        for component_identifier, component_struct in structs.items():
            calc = prep_mlip_calc(component_struct.info["charge"], component_struct.info["spin"])
            results[identifier][component_identifier] = ase_calc_single_point_job(calc, component_struct, component_struct.info["charge"], component_struct.info["spin"])

    if results_directory is not None:
        dumpfn(results, os.path.join(results_directory, "mlip_distance_scaling.json"))
    else:
        return results


def mlip_unoptimized_spin_gap(unoptimized_spin_gap_structures, results_directory=None):
    results = {}
    for identifier, struct in tqdm.tqdm(unoptimized_spin_gap_structures.items()):
        calc_orig = prep_mlip_calc(struct["orig"].info["charge"], struct["orig"].info["spin"])
        results[identifier] = {struct["orig"].info["spin"]: ase_calc_single_point_job(calc_orig, struct["orig"], struct["orig"].info["charge"], struct["orig"].info["spin"])}
        for spin in struct["additional_spins"]:
            calc = prep_mlip_calc(struct["orig"].info["charge"], spin)
            results[identifier][spin] = ase_calc_single_point_job(calc, struct["orig"], struct["orig"].info["charge"], spin)

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