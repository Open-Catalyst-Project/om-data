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

@job
def ase_calc_freq_job(
    calc,
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
    temperature=298.15,
    pressure=1.0,
    vib_kwargs=None,
    copy_files=None,
    additional_fields=None,
):
    """
    Carry out a vibrational frequency calculation + analysis via ASE utilities.

    Parameters
    ----------
    calc
        Calculator object
    atoms
        Atoms object
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    temperature
        Temperature in K
    pressure
        Pressure in bar
    vib_kwargs
        Dictionary of vibrational frequency analysis parameters
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    additional_fields
        Additional fields to add to the results dictionary.

    Returns
    -------
    RunSchema
        Dictionary of results
    """

    vib_kwargs = vib_kwargs or {}

    vib = Runner(atoms, calc, copy_files=copy_files).run_vib(vib_kwargs=vib_kwargs)
    return VibSummarize(
        vib,
        charge_and_multiplicity=(charge, spin_multiplicity),
        additional_fields={"name": "ASE Frequency and Thermo"}
        | (additional_fields or {}),
    ).vib_and_thermo(
        "ideal_gas", energy=0.0, temperature=temperature, pressure=pressure
    )


@job
def ase_calc_relax_job(
    calc,
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
    opt_defaults=None,
    opt_params=None,
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
    opt_defaults
        Default arguments for the ASE optimizer.
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

    opt_flags = recursive_dict_merge(opt_defaults, opt_params)
    dyn = Runner(atoms, calc, copy_files=copy_files).run_opt(**opt_flags)
    return Summarize(
        charge_and_multiplicity=(charge, spin_multiplicity),
        additional_fields={"name": "ASE Relax"}
        | (additional_fields or {}),
    ).opt(dyn)


def ase_calc_double_opt_freq(
    calc1,
    calc2,
    atoms: Atoms,
    initial_charge: int,
    initial_spin_multiplicity: int,
    new_charge: int,
    new_spin_multiplicity: int,
    opt_params=EVAL_OPT_PARAMETERS,
    outputdir=os.getcwd(),
    json_name="test.json",
    copy_files=None,
    temperature=298.15,
    pressure=1.0,
    additional_fields=None,
):
    """
    Args:
        atoms (Atoms): ASE Atoms object
        initial_charge (int): Initial charge state
        initial_spin_multiplicity (int): Initial spin multiplicity
        new_charge (int): New charge state
        new_spin_multiplicity (int): New spin multiplicity
        opt_params: Dictionary of optimizer parameters
        copy_files: Files to copy to working directory
        additional_fields (dict, optional): Additional fields to include in output
        
    Returns:
        List[CalcSchema]: List containing results for both charge states
    """
    results = []

    # First optimization at initial charge
    opt1 = ase_calc_relax_job(
        calc=calc1,
        atoms=atoms,
        charge=initial_charge,
        spin_multiplicity=initial_spin_multiplicity,
        opt_params=opt_params,
        copy_files=copy_files,
        additional_fields={"name": f"Opt Initial Charge {initial_charge} Spin {initial_spin_multiplicity}"} | (additional_fields or {}),
    )
    results.append(opt1)
    
    # Frequency calculation on optimized geometry at initial charge
    freq1 = ase_calc_freq_job(
        calc=calc1,
        atoms=opt1["atoms"],
        charge=initial_charge,
        spin_multiplicity=initial_spin_multiplicity,
        temperature=temperature,
        pressure=pressure,
        copy_files=copy_files,
        additional_fields={"name": f"Freq Initial Charge {initial_charge} Spin {initial_spin_multiplicity}"} | (additional_fields or {}),
    )
    results.append(freq1)
    
    # Second optimization at new charge/spin
    opt2 = ase_calc_relax_job(
        calc=calc2,
        atoms=opt1["atoms"],  # Start from optimized geometry
        charge=new_charge,
        spin_multiplicity=new_spin_multiplicity,
        opt_params=opt_params,
        copy_files=copy_files,
        additional_fields={"name": f"Opt New Charge {new_charge} Spin {new_spin_multiplicity}"} | (additional_fields or {}),
    )
    results.append(opt2)
    
    # Frequency calculation on optimized geometry at new charge/spin
    freq2 = ase_calc_freq_job(
        calc=calc2,
        atoms=opt2["atoms"],
        charge=new_charge,
        spin_multiplicity=new_spin_multiplicity,
        temperature=temperature,
        pressure=pressure,
        copy_files=copy_files,
        additional_fields={"name": f"Freq New Charge {new_charge} Spin {new_spin_multiplicity}"} | (additional_fields or {}),
    )
    results.append(freq2)

    dumpfn(results, json_name)
    
    return results


def ase_calc_ts_freq_fb_qirc_freq(
    calc,
    atoms: Atoms,
    charge: int,
    spin_multiplicity: int,
    perturb_magnitude=0.6,
    outputdir=os.getcwd(),
    json_name="test.json",
    copy_files=None,
    temperature=298.15,
    pressure=1.0,
    additional_fields=None,
):
    results = []

    ts_opt = ase_calc_relax_job(
        calc=calc,
        atoms=atoms,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        opt_params=EVAL_TS_PARAMETERS,
        copy_files=copy_files,
        additional_fields={"name": f"TS Opt"} | (additional_fields or {}),
    )
    results.append(ts_opt)

    ts_freq = ase_calc_freq_job(
        calc=calc,
        atoms=ts_opt["atoms"],
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        temperature=temperature,
        pressure=pressure,
        copy_files=copy_files,
        additional_fields={"name": f"TS Freq"} | (additional_fields or {}),
    )
    results.append(ts_freq)

    forward_qirc_opt = ase_calc_relax_job(
        calc=calc,
        atoms=perturb(ts_opt["atoms"], ts_opt["results"]["imag_vib_freqs"][0], perturb_magnitude),
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        opt_params=EVAL_OPT_PARAMETERS,
        copy_files=copy_files,
        additional_fields={"name": f"Forward QIRC Opt"} | (additional_fields or {}),
    )
    results.append(forward_qirc_opt)

    forward_qirc_freq = ase_calc_freq_job(
        calc=calc,
        atoms=forward_qirc_opt["atoms"],
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        temperature=temperature,
        pressure=pressure,
        copy_files=copy_files,
        additional_fields={"name": f"Forward QIRC Freq"} | (additional_fields or {}),
    )
    results.append(forward_qirc_freq)

    backward_qirc_opt = ase_calc_relax_job(
        calc=calc,
        atoms=perturb(ts_opt["atoms"], ts_opt["results"]["imag_vib_freqs"][0], perturb_magnitude * -1),
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        opt_params=EVAL_OPT_PARAMETERS,
        copy_files=copy_files,
        additional_fields={"name": f"Backward QIRC Opt"} | (additional_fields or {}),
    )
    results.append(backward_qirc_opt)

    backward_qirc_freq = ase_calc_freq_job(
        calc=calc,
        atoms=backward_qirc_opt["atoms"],
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        temperature=temperature,
        pressure=pressure,
        copy_files=copy_files,
        additional_fields={"name": f"Backward QIRC Freq"} | (additional_fields or {}),
    )
    results.append(backward_qirc_freq)

    dumpfn(results, json_name)

    return results


def orca_double_opt_freq(
    atoms: Atoms,
    initial_charge: int,
    initial_spin_multiplicity: int,
    new_charge: int,
    new_spin_multiplicity: int,
    xc=ORCA_FUNCTIONAL,
    basis=ORCA_BASIS,
    orcasimpleinput=None,
    orcablocks=None,
    nprocs="max",
    opt_params=EVAL_OPT_PARAMETERS,
    outputdir=os.getcwd(),
    json_name="test.json",
    vertical=Vertical.Default,
    copy_files=None,
    temperature=298.15,
    pressure=1.0,
    additional_fields=None,
):
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
        opt_params: Dictionary of optimizer parameters
        outputdir: Directory to save output
        json_name: Name of the JSON file to save
        vertical: Vertical calculation type
        copy_files: Files to copy to working directory
        temperature: Temperature in K
        pressure: Pressure in atm
        additional_fields (dict, optional): Additional fields to include in output
        
    Returns:
        List[CalcSchema]: List containing results for both charge states
    """
    if orcasimpleinput is None:
        orcasimpleinput1 = ORCA_SIMPLE_INPUT.copy()
        orcasimpleinput2 = ORCA_SIMPLE_INPUT.copy()
    if orcablocks is None:
        orcablocks1 = ORCA_BLOCKS.copy()
        orcablocks2 = ORCA_BLOCKS.copy()
    if vertical == Vertical.MetalOrganics and initial_spin_multiplicity == 1:
        orcasimpleinput1.append("UKS")
        orcablocks1.append(get_symm_break_block(atoms, initial_charge))
    if vertical == Vertical.MetalOrganics and new_spin_multiplicity == 1:
        orcasimpleinput2.append("UKS")
        orcablocks2.append(get_symm_break_block(atoms, new_charge))
    orcasimpleinput1.extend(["NONBO", "NONPA"])
    orcasimpleinput2.extend(["NONBO", "NONPA"])

    nprocs = psutil.cpu_count(logical=False) if nprocs == "max" else nprocs

    default_inputs = [xc, basis, "engrad"]
    default_blocks = [f"%pal nprocs {nprocs} end"]

    calc1 = prep_calculator(
        charge=initial_charge,
        spin_multiplicity=initial_spin_multiplicity,
        default_inputs=default_inputs,
        default_blocks=default_blocks,
        input_swaps=orcasimpleinput1,
        block_swaps=orcablocks1,
    )

    calc2 = prep_calculator(
        charge=new_charge,
        spin_multiplicity=new_spin_multiplicity,
        default_inputs=default_inputs,
        default_blocks=default_blocks,
        input_swaps=orcasimpleinput2,
        block_swaps=orcablocks2,
    )

    return ase_calc_double_opt_freq(calc1,
                                    calc2,
                                    atoms,
                                    initial_charge,
                                    initial_spin_multiplicity,
                                    new_charge,
                                    new_spin_multiplicity,
                                    opt_params,
                                    outputdir,
                                    json_name,
                                    copy_files,
                                    temperature,
                                    pressure,
                                    additional_fields
                                    )