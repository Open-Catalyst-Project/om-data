import os
from shutil import which

from ase.calculators.orca import ORCA, OrcaProfile
from sella import Sella

ORCA_FUNCTIONAL = "wB97M-V"
ORCA_BASIS = "def2-TZVPD"
ORCA_COMMON_INPUT = [
    "RIJCOSX",
    "def2/J",
    "NoUseSym",
    "DIIS",
    "NOSOSCF",
    "NormalConv",
]
ORCA_INIT_INPUT = [
    "SP",
    "DEFGRID2",
    "NoPop",
    "NoTRAH",
    "SCNL",
]
ORCA_FINAL_INPUT = [
    "EnGrad",
    "DEFGRID3",
    "ALLPOP",
    "NBO",
]
ORCA_COMMON_BLOCKS = [
    "%scf Convergence Tight maxiter 300 end",
]
ORCA_INIT_BLOCKS = [
    "%elprop Dipole false end",
    '%output Print[P_OrbEn] 0 end', 
]
ORCA_FINAL_BLOCKS = [
    "%elprop Dipole true Quadrupole true end",
    '%nbo NBOKEYLIST = "$NBO NPA NBO E2PERT 0.1 $END" end',
    '%output Print[P_ReducedOrbPopMO_L] 1 Print[P_ReducedOrbPopMO_M] 1 Print[P_BondOrder_L] 1 Print[P_BondOrder_M] 1 end',
]
ORCA_ASE_SIMPLE_INPUT = " ".join([ORCA_FUNCTIONAL] + [ORCA_BASIS] + ORCA_COMMON_INPUT + ORCA_FINAL_INPUT)
OPT_PARAMETERS = {
    "optimizer": Sella,
    "store_intermediate_results": True,
    "fmax": 0.05,
    "max_steps": 100,
    "optimizer_kwargs": {
        "order": 0,
        "internal": True,
    },
}


def write_orca_inputs(
    atoms,
    output_directory,
    charge=0,
    mult=1,
    orcasimpleinput=ORCA_ASE_SIMPLE_INPUT,
    orcablocks=" ".join(ORCA_COMMON_BLOCKS + ORCA_FINAL_BLOCKS),
):
    """
    One-off method to be used if you wanted to write inputs for an arbitrary
    system. Primarily used for debugging.
    """

    MyOrcaProfile = OrcaProfile([which("orca")])
    calc = ORCA(
        charge=charge,
        mult=mult,
        profile=MyOrcaProfile,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        directory=output_directory,
    )
    calc.write_inputfiles(atoms, ["energy", "forces"])

def write_xyz(fh, atoms, charge, mult):
    """
    Function stolen from ASE to write Orca xyz
    """
    fh.write('*xyz')
    fh.write(" %d" % charge)
    fh.write(" %d \n" % mult)
    for atom in atoms:
        if atom.tag == 71:  # 71 is ascii G (Ghost)
            symbol = atom.symbol + ' : '
        else:
            symbol = atom.symbol + '   '
        fh.write(symbol +
                 str(atom.position[0]) + ' ' +
                 str(atom.position[1]) + ' ' +
                 str(atom.position[2]) + '\n')
    fh.write('*\n')

def write_multigrid_input(
    atoms,
    output_directory,
    charge=0,
    mult=1,
    nprocs=1,
    orcainitinput=" ".join([ORCA_FUNCTIONAL] + [ORCA_BASIS] + ORCA_COMMON_INPUT + ORCA_INIT_INPUT),
    orcainitblocks="\n".join(ORCA_COMMON_BLOCKS + ORCA_INIT_BLOCKS),
    orcafinalinput=ORCA_ASE_SIMPLE_INPUT,
    orcafinalblocks="\n".join(ORCA_COMMON_BLOCKS + ORCA_FINAL_BLOCKS),
):

    with open(os.path.join(output_directory, 'orca.inp'), 'w') as fh:
        fh.write(f'%pal nprocs {nprocs} end\n')
        fh.write(f'! {orcainitinput}\n')
        fh.write(orcainitblocks + '\n')
        write_xyz(fh, atoms, charge, mult)  
  
        fh.write('$new_job\n')
        fh.write(f'! {orcafinalinput}\n')
        fh.write(orcafinalblocks + '\n')     
        write_xyz(fh, atoms, charge, mult)  
        fh.write(f'%pal nprocs {nprocs} end')
