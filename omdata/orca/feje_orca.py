from __future__ import annotations

import asyncio
import logging
import sys
from threading import Semaphore, Thread
from typing import TYPE_CHECKING

from ase.atoms import Atoms
from ase.calculators.orca import ORCA, OrcaTemplate
from quacc.runners.ase import run_opt
from quacc.schemas.cclib import summarize_cclib_opt_run
from quacc.utils.dicts import recursive_dict_merge
from quacc.utils.lists import merge_list_params

if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms

    from quacc.schemas._aliases.cclib import cclibASEOptSchema
    from quacc.utils.files import Filenames, SourceDirectory

from feje.scripts.collect_results import collect_results_ as collect_results
from feje.scripts.submit_job import submit_job_ as submit_job

_LABEL = OrcaTemplate()._label  # skipcq: PYL-W0212
LOG_FILE = f"{_LABEL}.out"
GEOM_FILE = f"{_LABEL}.xyz"

log = logging.getLogger(__name__)


class _AsyncLoopInThread(Thread):
    """
    Helper class that creates an event loop in its own thread,
    """

    def __init__(self) -> None:
        super().__init__(daemon=False)
        self.loop: asyncio.events.AbstractEventLoop = asyncio.new_event_loop()
        self.start()

    def run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()


# Start an asyncio io event loop to handle all feje calls from this module
submit_thread = _AsyncLoopInThread()
collect_thread = _AsyncLoopInThread()
active_thread_semaphore = Semaphore(value=1)


async def feje_submit(calc_folder, feje_project, calculation_type="vasp"):
    await submit_job(
        input_folder=calc_folder,
        project=feje_project,  # Fix this to use properties, use test_0 or workflows_dev
        calculation_type=calculation_type,  #
        recursive=False,
        re_submit="REPLACE",
        s3_profile="NONE",
        s3_bucket="opencatalysisdata",
        priority=None,
        output_submissions_file=None,
    )
    await asyncio.sleep(5)
    return


async def feje_collect(calc_folder, feje_project, calculation_type="vasp"):
    await collect_results(
        output_folder=calc_folder,
        timeout=None,  # no timeout!
        wait=150,  # was DEFAULT_WAIT_TIME_S
        delete_from_s3=False,
        s3_profile="NONE",
        s3_bucket="opencatalysisdata",
        skip_download=False,
        force_redownload=True,
        s3_index_key="jobs/index.gz",
        use_local_cache=False,  # Feels like this should be true, but when I tried that this collect always hung. Maybe something about local_cache default?
        feje_job_path=None,
        job_id=None,
    )
    return


import quacc.recipes.orca._base
from ase.calculators.orca import ORCA, OrcaTemplate


class FejeORCA(ORCA):
    """
    Override the ASE Orca calculator to run Orca via Feje.
    """

    def __init__(self, *args, feje_project: str, **kwargs):
        self.feje_project = feje_project
        super().__init__(*args, **kwargs)

    def calculate(self, atoms, properties, system_changes):
        self.write_inputfiles(atoms, properties)

        active_thread_semaphore.release()
        try:
            submit_future = asyncio.run_coroutine_threadsafe(
                feje_submit(
                    self.directory,
                    calculation_type="orca",
                    feje_project=self.feje_project,
                ),
                submit_thread.loop,
            )
            submit_future.result()
            collect_future = asyncio.run_coroutine_threadsafe(
                feje_collect(
                    self.directory,
                    calculation_type="orca",
                    feje_project=self.feje_project,
                ),
                collect_thread.loop,
            )
            collect_future.result()
            return 0
        except (
            Exception
        ) as err:  # need to do proper error handling or let feje_submit/collect raise!
            sys.stderr.write(f"Submission failed: {err}")
        finally:
            self.results = self.template.read_results(self.directory)
            # Acquire a semaphore since we're done with async and moving back to ASE script!
            active_thread_semaphore.acquire()

        return 0


def feje_run_and_summarize_opt(
    atoms: Atoms,
    charge: int = 0,
    spin_multiplicity: int = 1,
    default_inputs: list[str] | None = None,
    default_blocks: list[str] | None = None,
    input_swaps: list[str] | None = None,
    block_swaps: list[str] | None = None,
    calc_kwargs: dict[str, Any] | None = None,
    opt_defaults: dict[str, Any] | None = None,
    opt_params: dict[str, Any] | None = None,
    additional_fields: dict[str, Any] | None = None,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
) -> cclibASEOptSchema:
    """
    Base job function for ORCA recipes with ASE optimizer.

    Parameters
    ----------
    atoms
        Atoms object
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    default_inputs
        Default input parameters.
    default_blocks
        Default block input parameters.
    input_swaps
        List of orcasimpleinput swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    block_swaps
        List of orcablock swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    opt_defaults
        Default arguments for the ASE optimizer.
    opt_params
        Dictionary of custom kwargs for [quacc.runners.ase.run_opt][]
    additional_fields
        Any additional fields to supply to the summarizer.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.

    Returns
    -------
    cclibASEOptSchema
        Dictionary of results
    """
    atoms.calc = _prep_feje_calculator(
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        default_inputs=default_inputs,
        default_blocks=default_blocks,
        input_swaps=input_swaps,
        block_swaps=block_swaps,
        calc_kwargs=calc_kwargs,
    )

    opt_flags = recursive_dict_merge(opt_defaults, opt_params)
    dyn = run_opt(atoms, copy_files=copy_files, **opt_flags)
    return summarize_cclib_opt_run(dyn, LOG_FILE, additional_fields=additional_fields)


def _prep_feje_calculator(
    charge: int = 0,
    spin_multiplicity: int = 1,
    default_inputs: list[str] | None = None,
    default_blocks: list[str] | None = None,
    input_swaps: list[str] | None = None,
    block_swaps: list[str] | None = None,
    calc_kwargs: dict[str, Any] | None = None,
) -> FejeORCA:
    """
    Prepare the Feje ORCA calculator.

    Parameters
    ----------
    charge
        Charge of the system.
    spin_multiplicity
        Multiplicity of the system.
    default_inputs
        Default input parameters.
    default_blocks
        Default block input parameters.
    input_swaps
        List of orcasimpleinput swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    block_swaps
        List of orcablock swaps for the calculator. To remove entries
        from the defaults, put a `#` in front of the name.
    calc_kwargs
        Custom kwargs for the FejeOrca calculator

    Returns
    -------
    ORCA
        The ORCA calculator
    """
    inputs = merge_list_params(default_inputs, input_swaps)
    blocks = merge_list_params(default_blocks, block_swaps)
    if "xyzfile" not in inputs:
        inputs.append("xyzfile")
    orcasimpleinput = " ".join(inputs)
    orcablocks = "\n".join(blocks)

    return FejeORCA(
        profile="orca",
        charge=charge,
        mult=spin_multiplicity,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        **calc_kwargs,
    )


quacc.recipes.orca._base.run_and_summarize_opt = feje_run_and_summarize_opt
