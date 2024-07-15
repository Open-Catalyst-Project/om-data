from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout, exit
import csv

row_idx  = int(sys.argv[1]) + 1
# Load the CSV file containing systems to simulate
with open("../elytes.csv", "r") as f:
    systems = list(csv.reader(f))
Temp = float(systems[row_idx][4])

pdb = app.PDBFile('system_init.pdb')
modeller = app.Modeller(pdb.topology, pdb.positions)
forcefield = app.ForceField('system.xml')
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=0.5, constraints=None)
system.addForce(MonteCarloBarostat(1.0*bar, Temp*kelvin, 1))
integrator = VariableLangevinIntegrator(Temp*kelvin,   # Temperate of head bath
                                      1/picosecond, # Friction coefficient
                                      1e-6) # Time step
simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()
steps = 1000
simulation.reporters.append(StateDataReporter('equildata.txt', 1, progress=True, density=True,totalSteps=steps,speed=True))
simulation.step(steps)
simulation.saveState('equilsystem.state')
simulation.saveCheckpoint('equilsystem.checkpoint')
