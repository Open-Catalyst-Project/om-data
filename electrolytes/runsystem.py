import argparse
import csv
import glob
import os

import openmm.app as app
from openmm import *
from openmm.unit import bar, kelvin, nanometer, picosecond


def main(row_idx: int, job_dir: str):
    """
    Main job driver

    :param row_idx: Row number in the `elytes.csv` that is to be run
    :param job_dir: Directory where job files are stored and run
    """
    # Read the temperature from the CSV file
    with open("elytes.csv", "r") as f:
        systems = list(csv.reader(f))
    temp = float(systems[row_idx][4])

    dt = 0.002  # ps
    t_final = 500*1000 # ps, which is 500 ns
    frames = 1000
    runtime = int(t_final / dt)

    cwd = os.getcwd()
    os.chdir(os.path.join(job_dir, str(row_idx)))
    pdb = app.PDBFile("system_equil.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    forcefield = app.ForceField("system.xml")
    rdist = 1.0 * nanometer
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=rdist,
        constraints=None,
        switchDistance=0.9 * rdist,
    )

    system.addForce(MonteCarloBarostat(1.0 * bar, temp * kelvin, 100))
    integrator = LangevinMiddleIntegrator(
        temp * kelvin,  # Temperate of head bath
        1 / picosecond,  # Friction coefficient
        dt * picosecond,
    )  # Time step
    simulation = app.Simulation(modeller.topology, system, integrator)
    rate = max(1, int(runtime / frames))
    if os.path.exists("md.chk"):
        simulation.loadCheckpoint("md.chk")
        restart = True
    else:
        simulation.loadState("equilsystem.state")
        simulation.loadCheckpoint("equilsystem.checkpoint")
        # Assuming you have a Simulation object named `simulation`
        state = simulation.context.getState()
        # Get the box vectors (3x3 matrix representing the vectors defining the periodic box)
        box_vectors = state.getPeriodicBoxVectors()
        # Extract the box dimensions (lengths of the box vectors)
        box_lengths = [box_vectors[i][i].value_in_unit(unit.nanometer) for i in range(3)]
        print("Box dimensions (nm):", box_lengths)
        simulation.minimizeEnergy()
        restart = False

    # Get name for PDBReporter (OpenMM cannot add to an existing .pdb file for restarts)
    output_pdb_basename = "system_output"
    other_name = sorted(glob.glob(output_pdb_basename + "*.pdb"))
    if other_name and other_name[-1] != output_pdb_basename + ".pdb":
        last_name = other_name[-1].replace(".pdb", "")
        count = int(last_name.split("_")[-1]) + 1
    else:
        count = 0
    output_name = f"{output_pdb_basename}_{count}.pdb"

    simulation.reporters.append(
        app.PDBReporter(output_name, rate, enforcePeriodicBox=True)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            "data.csv",
            rate,
            progress=True,
            temperature=True,
            potentialEnergy=True,
            density=True,
            totalSteps=runtime,
            speed=True,
            append=restart,
        )
    )
    simulation.reporters.append(app.CheckpointReporter("md.chk", rate))
    simulation.step(
        runtime - simulation.currentStep - 10
    )  # starts at 10 for some reason, equilibration?
    # Assuming you have a Simulation object named `simulation`
    state = simulation.context.getState()
    # Get the box vectors (3x3 matrix representing the vectors defining the periodic box)
    box_vectors = state.getPeriodicBoxVectors()
    # Extract the box dimensions (lengths of the box vectors)
    box_lengths = [box_vectors[i][i].value_in_unit(unit.nanometer) for i in range(3)]
    print("Final Box dimensions (nm):", box_lengths)
    os.chdir(cwd)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameters for OMol24 Electrolytes MD"
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        required=True,
        help="Directory containing input electrolyte directories/where job files will be stored",
    )
    parser.add_argument(
        "--row_idx", type=int, help="Job specified in elytes.csv to be run"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.row_idx, args.job_dir)
