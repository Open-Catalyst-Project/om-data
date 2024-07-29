from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout, exit
import csv

#Read the temperature from the CSV file
row_idx  = int(sys.argv[1]) 
with open("../elytes.csv", "r") as f:
    systems = list(csv.reader(f))
Temp = float(systems[row_idx][4])

dt = 0.002 #ps
t_final = 5000 #ps, which is 500 ns
frames = 1000
runtime = int(t_final/dt)

pdb = app.PDBFile('system_equil.pdb')
modeller = app.Modeller(pdb.topology, pdb.positions)
forcefield = app.ForceField('system.xml')
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=0.5, constraints=None)
system.addForce(MonteCarloBarostat(1.0*bar, Temp*kelvin, 1))
integrator = LangevinMiddleIntegrator(Temp*kelvin,   # Temperate of head bath
                                      1/picosecond, # Friction coefficient
                                      dt*picosecond) # Time step
simulation = app.Simulation(modeller.topology, system, integrator)
simulation.loadState('equilsystem.state')
simulation.loadCheckpoint('equilsystem.checkpoint')
simulation.minimizeEnergy()

rate = int(runtime/frames)
if rate < 1:
    rate = 1
simulation.reporters.append(PDBReporter('system_output.pdb', rate, enforcePeriodicBox=True))
simulation.reporters.append(StateDataReporter('data.txt', rate, progress=True, temperature=True, potentialEnergy=True, density=True,totalSteps=runtime,speed=True))
simulation.step(runtime)
