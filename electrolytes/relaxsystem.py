from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout, exit
import csv

row_idx  = int(sys.argv[1]) 
# Load the CSV file containing systems to simulate
with open("../elytes.csv", "r") as f:
    systems = list(csv.reader(f))
Temp = float(systems[row_idx][4])

steps = 2000
pdb = app.PDBFile('system_init.pdb')
modeller = app.Modeller(pdb.topology, pdb.positions)
forcefield = app.ForceField('system.xml')
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=0.5, constraints=None)
system.addForce(MonteCarloBarostat(1.0*bar, Temp*kelvin, 1))
integrator = LangevinMiddleIntegrator(Temp*kelvin,   # Temperate of head bath
                                      1/picosecond, # Friction coefficient
                                      0.001*picosecond) # Tolerance value
simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.reporters.append(StateDataReporter('relaxdata.txt', 10, progress=True, density=True,totalSteps=steps,speed=True))
simulation.minimizeEnergy()
simulation.step(steps)

simulation.saveState('equilsystem.state')
simulation.saveCheckpoint('equilsystem.checkpoint')
# Get the final state
final_state = simulation.context.getState(getPositions=True)
# Save the final frame to a PDB file
with open('system_equil.pdb', 'w') as f:
    PDBFile.writeFile(pdb.topology, final_state.getPositions(), f, keepIds=True)
