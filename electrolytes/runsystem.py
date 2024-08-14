import csv
import glob
import os
import sys

import openmm.app as app
from openmm import *
from openmm.unit import nanometer, bar, kelvin, picosecond

# Read the temperature from the CSV file
row_idx = int(sys.argv[1])
with open("elytes.csv", "r") as f:
    systems = list(csv.reader(f))
Temp = float(systems[row_idx][4])

dt = 0.002  # ps
t_final = 500000  # ps, which is 500 ns
frames = 1000
runtime = int(t_final / dt)

os.chdir(str(row_idx))
pdb = app.PDBFile("system_equil.pdb")
modeller = app.Modeller(pdb.topology, pdb.positions)
forcefield = app.ForceField("system.xml")
rdist = 1.0*nanometer
system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=rdist, constraints=None,switchDistance=0.9*rdist)

system.addForce(MonteCarloBarostat(1.0 * bar, Temp * kelvin, 100))
integrator = LangevinMiddleIntegrator(
    Temp * kelvin,  # Temperate of head bath
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

simulation.reporters.append(app.PDBReporter(output_name, rate, enforcePeriodicBox=True))
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
