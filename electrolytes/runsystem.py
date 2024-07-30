import csv
import glob
import os

from openmm import *
from openmm.app import *
from openmm.unit import *

# Read the temperature from the CSV file
row_idx = int(sys.argv[1])
with open("elytes.csv", "r") as f:
    systems = list(csv.reader(f))
Temp = float(systems[row_idx + 1][4])

dt = 0.002  # ps
t_final = 500000  # ps, which is 500 ns
frames = 1000
runtime = int(t_final / dt)

os.chdir(str(row_idx))
pdb = app.PDBFile("system_equil.pdb")
modeller = app.Modeller(pdb.topology, pdb.positions)
forcefield = app.ForceField("system.xml")
system = forcefield.createSystem(
    modeller.topology, nonbondedMethod=PME, nonbondedCutoff=0.5, constraints=None
)
system.addForce(MonteCarloBarostat(1.0 * bar, Temp * kelvin, 1))
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

simulation.reporters.append(PDBReporter(output_name, rate, enforcePeriodicBox=True))
simulation.reporters.append(
    StateDataReporter(
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
simulation.reporters.append(CheckpointReporter("md.chk", rate))
simulation.step(
    runtime - simulation.currentStep - 10
)  # starts at 10 for some reason, equilibration?
