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
    EVAL_OPT_PARAMETERS,
    EVAL_TS_PARAMETERS,
    ORCA_BASIS,
    ORCA_BLOCKS,
    NBO_FLAGS,
    ORCA_FUNCTIONAL,
    ORCA_SIMPLE_INPUT,
    Vertical,
    get_symm_break_block,
)

from omdata.eval_workflows.eval_jobs import ase_calc_single_point_job, ase_calc_relax_job

# key: identifier, value: {"ligand_pocket": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}, "ligand": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}, "pocket": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}}
ligand_pocket_structures = {}

# key: family_identifier, value: {"conformer_identifier": {"atoms": Atoms, "charge": int, "spin_multiplicity": int}, ...}
geom_conformers_structures = {}

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


