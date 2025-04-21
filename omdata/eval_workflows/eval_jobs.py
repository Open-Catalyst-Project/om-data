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
    LOOSE_OPT_PARAMETERS,
    OPT_PARAMETERS,
    TIGHT_OPT_PARAMETERS,
    ORCA_BASIS,
    ORCA_BLOCKS,
    NBO_FLAGS,
    ORCA_FUNCTIONAL,
    ORCA_SIMPLE_INPUT,
    Vertical,
    get_symm_break_block,
)

GEOM_FILE = "geom.xyz"


def ase_calc_relax_job(
    calc,
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
    opt_params=TIGHT_OPT_PARAMETERS,
    additional_fields=None,
    copy_files=None,
):
    """
    Carry out a relaxation calculation via ASE utilities.

    Parameters
    ----------
    atoms
        Atoms object
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    opt_params
        Dictionary of custom kwargs for [quacc.runners.ase.Runner.run_opt][]
    additional_fields
        Any additional fields to supply to the summarizer.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.

    Returns
    -------
    OptSchema
        Dictionary of results
    """

    dyn = Runner(atoms, calc, copy_files=copy_files).run_opt(**opt_params)
    return Summarize(
        charge_and_multiplicity=(charge, spin_multiplicity),
        additional_fields={"name": "ASE Relax"}
        | (additional_fields or {}),
    ).opt(dyn)


def ase_calc_single_point_job(
    calc,
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
    additional_fields=None,
    copy_files=None,
):
    """
    Carry out a single point calculation via ASE utilities.

    Parameters
    ----------
    atoms
        Atoms object
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    additional_fields
        Any additional fields to supply to the summarizer.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.

    Returns
    -------
    OptSchema
        Dictionary of results
    """

    final_atoms = Runner(atoms, calc, copy_files=copy_files).run_calc()

    return Summarize(
        charge_and_multiplicity=(charge, spin_multiplicity),
        additional_fields={"name": "ASE Single Point"}
        | (additional_fields or {}),
    ).run(final_atoms, atoms)


