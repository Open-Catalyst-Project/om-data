from typing import Optional, Union, List, Literal
from pathlib import Path
from ase.atoms import Atoms
from quacc.recipes.orca._base import prep_calculator
import os
import psutil
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

from omdata.eval_workflows.eval_workflows import ase_calc_single_point_job, ase_calc_freq_job, ase_calc_relax_job

# Each entry in this list should be a three element list with ligand_pocket structure, ligand structure, pocket structure
ligand_pocket_structures = []

# Each entry in this list should be an N element list defining the N different conformer structures for a given family
geom_conformers_structures = []

def mlip_ligand_pocket(ligand_pocket_structures):
    results = {}
    for ii, structs in enumerate(ligand_pocket_structures):
        lp_calc = prep_mlip_calc(structs[0]["charge"], structs[0]["spin_multiplicity"])
        l_calc = prep_mlip_calc(structs[1]["charge"], structs[1]["spin_multiplicity"])
        p_calc = prep_mlip_calc(structs[2]["charge"], structs[2]["spin_multiplicity"])

        lp_result = ase_calc_single_point_job(lp_calc, structs[0]["atoms"], structs[0]["charge"], structs[0]["spin_multiplicity"])
        l_result = ase_calc_single_point_job(l_calc, structs[1]["atoms"], structs[1]["charge"], structs[1]["spin_multiplicity"])
        p_result = ase_calc_single_point_job(p_calc, structs[2]["atoms"], structs[2]["charge"], structs[2]["spin_multiplicity"])

        results[ii] = [lp_result, l_result, p_result]

    dumpfn(results, "mlip_ligand_pocket.json")

def mlip_geom_conformers(geom_conformers_structures):
    results = {}
    for ii, structs in enumerate(geom_conformers_structures):
        family_results = {}
        for jj, struct in enumerate(structs):
            calc = prep_mlip_calc(struct["charge"], struct["spin_multiplicity"])
            result = ase_calc_relax_job(calc, struct["atoms"], struct["charge"], struct["spin_multiplicity"])
            family_results[jj] = result
        results[ii] = family_results

    dumpfn(results, "mlip_geom_conformers.json")
