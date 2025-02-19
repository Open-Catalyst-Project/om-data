from typing import Optional, Union, List, Literal
from pathlib import Path
from ase.atoms import Atoms
from quacc.recipes.orca.core import ase_relax_job
from quacc.schemas.calc import CalcSchema
from quacc.recipes.orca._base import prep_calculator

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from monty.dev import requires

from quacc import job
from quacc.runners.ase import Runner
from quacc.schemas.ase import Summarize, VibSummarize
from quacc.utils.dicts import recursive_dict_merge

from omdata.orca.calc import (
    OPT_PARAMETERS,
    ORCA_BASIS,
    ORCA_BLOCKS,
    NBO_FLAGS,
    ORCA_FUNCTIONAL,
    ORCA_SIMPLE_INPUT,
    Vertical,
    get_symm_break_block,
)

import json

@job
def ase_freq_job(
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
    xc=ORCA_FUNCTIONAL,
    basis=ORCA_BASIS,
    orcasimpleinput: list[str] | None = None,
    orcablocks: list[str] | None = None,
    nprocs: int | Literal["max"] = "max",
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    additional_fields: dict[str, Any] | None = None,
) -> RunSchema:
    """
    Carry out a vibrational frequency analysis via ASE utilities.

    Parameters
    ----------
    atoms
        Atoms object
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    xc
        Exchange-correlation functional
    basis
        Basis set
    orcasimpleinput
        List of `orcasimpleinput` swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name. Refer to the
        [ase.calculators.orca.ORCA][] calculator for details on `orcasimpleinput`.
    orcablocks
        List of `orcablocks` swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name. Refer to the
        [ase.calculators.orca.ORCA][] calculator for details on `orcablocks`.
    nprocs
        Number of processors to use. Defaults to the number of physical cores.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    additional_fields
        Additional fields to add to the results dictionary.

    Returns
    -------
    RunSchema
        Dictionary of results
    """
    nprocs = (psutil.cpu_count(logical=False) if nprocs == "max" else nprocs) or 1
    default_inputs = [xc, basis, "normalprint"]
    default_blocks = [f"%pal nprocs {nprocs} end"]

    vib_kwargs = vib_kwargs or {}

    calc_defaults = {"method": method}
    calc_flags = recursive_dict_merge(calc_defaults, calc_kwargs)

    calc = prep_calculator(
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        default_inputs=default_inputs,
        default_blocks=default_blocks,
        input_swaps=input_swaps,
        block_swaps=block_swaps,
        **calc_kwargs,
    )

    vib = Runner(atoms, calc).run_vib(vib_kwargs=vib_kwargs)
    return VibSummarize(
        vib,
        additional_fields={"name": "TBLite Frequency and Thermo"}
        | (additional_fields or {}),
    ).vib_and_thermo(
        "ideal_gas", energy=energy, temperature=temperature, pressure=pressure
    )


def double_ase_opt_freq_orca(
    atoms: Atoms,
    initial_charge: int,
    initial_spin_multiplicity: int,
    new_charge: int,
    new_spin_multiplicity: int,
    xc=ORCA_FUNCTIONAL,
    basis=ORCA_BASIS,
    orcasimpleinput: Optional[List[str]] = None,
    orcablocks: Optional[List[str]] = None,
    nprocs: Union[int, Literal["max"]] = "max",
    outputdir=os.getcwd(),
    dump_json=True,
    vertical=Vertical.Default,
    copy_files: Optional[Union[str, Path, dict[str, str]]] = None,
    additional_fields: Optional[dict] = None,
) -> List[CalcSchema]:
    """
    Args:
        atoms (Atoms): ASE Atoms object
        initial_charge (int): Initial charge state
        initial_spin_multiplicity (int): Initial spin multiplicity
        new_charge (int): New charge state
        new_spin_multiplicity (int): New spin multiplicity
        xc (str): Exchange-correlation functional
        basis (str): Basis set
        orcasimpleinput (List[str], optional): Additional ORCA simple input parameters
        orcablocks (List[str], optional): Additional ORCA block parameters
        nprocs (Union[int, "max"]): Number of processors to use
        copy_files: Files to copy to working directory
        additional_fields (dict, optional): Additional fields to include in output
        
    Returns:
        List[CalcSchema]: List containing results for both charge states
    """
    from quacc import SETTINGS

    SETTINGS.RESULTS_DIR = outputdir

    if orcasimpleinput is None:
        orcasimpleinput = ORCA_SIMPLE_INPUT.copy()
    if orcablocks is None:
        orcablocks = ORCA_BLOCKS.copy()
    if opt_params is None:
        opt_params = OPT_PARAMETERS.copy()
    if vertical == Vertical.MetalOrganics and spin_multiplicity == 1:
        orcasimpleinput.append("UKS")
        orcablocks.append(get_symm_break_block(atoms, charge))
    orcasimpleinput.extend(["NONBO", "NONPA"])

    nprocs = psutil.cpu_count(logical=False) if nprocs == "max" else nprocs
    default_inputs = [xc, basis, "engrad"]
    default_blocks = [f"%pal nprocs {nprocs} end"]

    results = []
    
    # First optimization at initial charge
    opt1 = ase_relax_job(
        atoms=atoms,
        charge=initial_charge,
        spin_multiplicity=initial_spin_multiplicity,
        xc=xc,
        basis=basis,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        nprocs=nprocs,
        copy_files=copy_files,
        additional_fields={"name": f"ORCA Opt Initial Charge {initial_charge} Spin {initial_spin_multiplicity}"} | (additional_fields or {}),
    )
    results.append(opt1)
    
    # Frequency calculation at initial charge
    freq1 = ase_freq_job(
        atoms=opt1["atoms"],
        charge=initial_charge,
        spin_multiplicity=initial_spin_multiplicity,
        xc=xc,
        basis=basis,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        nprocs=nprocs,
        copy_files=copy_files,
        additional_fields={"name": f"ORCA Freq Initial Charge {initial_charge} Spin {initial_spin_multiplicity}"} | (additional_fields or {}),
    )
    results.append(freq1)
    
    # Second optimization at new charge/spin
    new_charge = initial_charge + charge_change
    opt2 = ase_relax_job(
        atoms=opt1["atoms"],  # Start from optimized geometry
        charge=new_charge,
        spin_multiplicity=new_spin_multiplicity,
        xc=xc,
        basis=basis,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        nprocs=nprocs,
        copy_files=copy_files,
        additional_fields={"name": f"ORCA Opt New Charge {new_charge} Spin {new_spin_multiplicity}"} | (additional_fields or {}),
    )
    results.append(opt2)
    
    # Frequency calculation at new charge/spin
    freq2 = ase_freq_job(
        atoms=opt2["atoms"],
        charge=new_charge,
        spin_multiplicity=new_spin_multiplicity,
        xc=xc,
        basis=basis,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        nprocs=nprocs,
        copy_files=copy_files,
        additional_fields={"name": f"ORCA Freq New Charge {new_charge} Spin {new_spin_multiplicity}"} | (additional_fields or {}),
    )
    results.append(freq2)

    if dump_json:
        with open("test.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


