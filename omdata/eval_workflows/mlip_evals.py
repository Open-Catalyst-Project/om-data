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
    OPT_PARAMETERS,
    EVAL_OPT_PARAMETERS,
    EVAL_TS_PARAMETERS,
)

from omdata.eval_workflows.eval_jobs import ase_calc_single_point_job, ase_calc_relax_job

# key: identifier, value: {
#                "ligand_pocket": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}, 
#                "ligand": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}, 
#                "pocket": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}
#                }
ligand_pocket_structures = {}
# NOTE: in order to make it easy to calculate an "interaction force", the ligand_pocket atoms should be in the order ligand_atoms, pocket_atoms.
# Otherwise, we will need to track the indices of the ligand atoms and pocket atoms in the ligand_pocket atoms.

# key: family_identifier, value: {
#                "ligand_in_pocket": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}, 
#                "conformer_0_identifier": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}, 
#                "conformer_1_identifier": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}, 
#                ...
#                }
ligand_strain_structures = {}

# key: family_identifier, value: {
#                "conformer_0_identifier": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}, 
#                "conformer_1_identifier": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}, 
#                ...
#                }
geom_conformers_structures = {}

# key: identifier, value: {
#                "unprotonated": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}, 
#                "protonated": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}, 
#                }
protonation_structures = {}

# key: identifier, value: {
#                "atoms": Atoms, "charge": int, "spin_multiplicity": int
#                }
unoptimized_ie_ea_structures = {}

# key: identifier, value: { 
#                "complex": {"atoms": Atoms, "charge": int, "spin_multiplicity": int},
#                "component_0_identifier": {"atoms": Atoms, "charge": int, "spin_multiplicity": int},
#                "component_1_identifier": {"atoms": Atoms, "charge": int, "spin_multiplicity": int},
#                ...
#                }
distance_scaling_structures = {}
# NOTE: in order to make it easy to calculate an "interaction force", the complex atoms should be in the order component_0_atoms, component_1_atoms, ...
# Otherwise, we will need to track the indices of the component atoms in the complex atoms.

# key: identifier, value: {
#                "atoms": Atoms, "charge": int, "spin_multiplicity": int
#                }
unoptimized_spin_gap_structures = {}
# NOTE: structures should all be "high spin", i.e. spin_multiplicity > 2


def mlip_ligand_pocket(ligand_pocket_structures):
    results = {}
    for identifier, structs in tqdm.tqdm(ligand_pocket_structures.items()):
        lp_calc = prep_mlip_calc(structs["ligand_pocket"]["charge"], structs["ligand_pocket"]["spin_multiplicity"])
        l_calc = prep_mlip_calc(structs["ligand"]["charge"], structs["ligand"]["spin_multiplicity"])
        p_calc = prep_mlip_calc(structs["pocket"]["charge"], structs["pocket"]["spin_multiplicity"])

        lp_result = ase_calc_single_point_job(lp_calc, structs["ligand_pocket"]["atoms"], structs["ligand_pocket"]["charge"], structs["ligand_pocket"]["spin_multiplicity"])
        l_result = ase_calc_single_point_job(l_calc, structs["ligand"]["atoms"], structs["ligand"]["charge"], structs["ligand"]["spin_multiplicity"])
        p_result = ase_calc_single_point_job(p_calc, structs["pocket"]["atoms"], structs["pocket"]["charge"], structs["pocket"]["spin_multiplicity"])

        results[identifier] = {"ligand_pocket": lp_result, "ligand": l_result, "pocket": p_result}

    dumpfn(results, "mlip_ligand_pocket.json")


def ligand_strain(ligand_strain_structures):
    results = {}
    for family_identifier, structs in tqdm.tqdm(geom_conformers_structures.items()):
        family_results = {}
        for conformer_identifier, struct in structs.items():
            calc = prep_mlip_calc(struct["charge"], struct["spin_multiplicity"])
            if conformer_identifier == "ligand_in_pocket":
                result = ase_calc_relax_job(calc, struct["atoms"], struct["charge"], struct["spin_multiplicity"], opt_params=OPT_PARAMETERS)
            else:
                result = ase_calc_relax_job(calc, struct["atoms"], struct["charge"], struct["spin_multiplicity"])
            family_results[conformer_identifier] = result
        results[family_identifier] = family_results

    dumpfn(results, "mlip_ligand_strain.json")


def mlip_geom_conformers(geom_conformers_structures):
    results = {}
    for family_identifier, structs in tqdm.tqdm(geom_conformers_structures.items()):
        family_results = {}
        for conformer_identifier, struct in structs.items():
            calc = prep_mlip_calc(struct["charge"], struct["spin_multiplicity"])
            result = ase_calc_relax_job(calc, struct["atoms"], struct["charge"], struct["spin_multiplicity"])
            family_results[conformer_identifier] = result
        results[family_identifier] = family_results

    dumpfn(results, "mlip_geom_conformers.json")


def mlip_protonation_energies(protonation_structures):
    results = {}
    for identifier, structs in tqdm.tqdm(protonation_structures.items()):
        unprotonated_calc = prep_mlip_calc(structs["unprotonated"]["charge"], structs["unprotonated"]["spin_multiplicity"])
        unprotonated_result = ase_calc_relax_job(unprotonated_calc, structs["unprotonated"]["atoms"], structs["unprotonated"]["charge"], structs["unprotonated"]["spin_multiplicity"])
        protonated_calc = prep_mlip_calc(structs["protonated"]["charge"], structs["protonated"]["spin_multiplicity"])
        protonated_result = ase_calc_relax_job(protonated_calc, structs["protonated"]["atoms"], structs["protonated"]["charge"], structs["protonated"]["spin_multiplicity"])
        results[identifier] = {"unprotonated": unprotonated_result, "protonated": protonated_result}

    dumpfn(results, "mlip_protonation_energies.json")


def mlip_unoptimized_ie_ea(unoptimized_ie_ea_structures):
    results = {}
    for identifier, struct in tqdm.tqdm(unoptimized_ie_ea_structures.items()):
        identifier_results = {}
        add_electron_charge = struct["charge"] - 1
        remove_electron_charge = struct["charge"] + 1
        if struct["spin_multiplicity"] == 1:
            add_electron_spin_multiplicity = [2]
            remove_electron_spin_multiplicity = [2]
        else:
            add_electron_spin_multiplicity = [struct["spin_multiplicity"]+1, struct["spin_multiplicity"]-1]
            remove_electron_spin_multiplicity = [struct["spin_multiplicity"]+1, struct["spin_multiplicity"]-1]

        calc_orig = prep_mlip_calc(struct["charge"], struct["spin_multiplicity"])
        result_orig = ase_calc_single_point_job(calc_orig, struct["atoms"], struct["charge"], struct["spin_multiplicity"])
        identifier_results["original"] = result_orig

        for spin_multiplicity in add_electron_spin_multiplicity:
            calc_add_electron = prep_mlip_calc(add_electron_charge, spin_multiplicity)
            result_add_electron = ase_calc_single_point_job(calc_add_electron, struct["atoms"], add_electron_charge, spin_multiplicity)
            identifier_results[f"add_electron_{spin_multiplicity}"] = result_add_electron

        for spin_multiplicity in remove_electron_spin_multiplicity:
            calc_remove_electron = prep_mlip_calc(remove_electron_charge, spin_multiplicity)
            result_remove_electron = ase_calc_single_point_job(calc_remove_electron, struct["atoms"], remove_electron_charge, spin_multiplicity)
            identifier_results[f"remove_electron_{spin_multiplicity}"] = result_remove_electron

        results[identifier] = identifier_results

    dumpfn(results, "mlip_unoptimized_ie_ea.json")


def mlip_distance_scaling(distance_scaling_structures):
    results = {}
    for identifier, structs in tqdm.tqdm(distance_scaling_structures.items()):
        for component_identifier, component_struct in structs.items():
            calc = prep_mlip_calc(component_struct["charge"], component_struct["spin_multiplicity"])
            result = ase_calc_single_point_job(calc, component_struct["atoms"], component_struct["charge"], component_struct["spin_multiplicity"])
            results[identifier][component_identifier] = result
    dumpfn(results, "mlip_distance_scaling.json")   


def mlip_unoptimized_spin_gap(unoptimized_spin_gap_structures):
    results = {}
    for identifier, struct in tqdm.tqdm(unoptimized_spin_gap_structures.items()):
        assert struct["spin_multiplicity"] > 2
        low_spin_multiplicity = (struct["spin_multiplicity"]-1)%2 + 1

        calc_orig = prep_mlip_calc(struct["charge"], struct["spin_multiplicity"])
        result_orig = ase_calc_single_point_job(calc_orig, struct["atoms"], struct["charge"], struct["spin_multiplicity"])
        
        calc_low_spin = prep_mlip_calc(struct["charge"], low_spin_multiplicity) 
        result_low_spin = ase_calc_single_point_job(calc_low_spin, struct["atoms"], struct["charge"], low_spin_multiplicity)

        results[identifier] = {"high_spin": result_orig, "low_spin": result_low_spin}

    dumpfn(results, "mlip_unoptimized_spin_gap.json")