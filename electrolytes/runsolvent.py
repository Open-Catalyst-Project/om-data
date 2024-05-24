from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout, exit


#TO-DO: Read temperature from CSV file
pdb = app.PDBFile('solvent_init.pdb')
modeller = app.Modeller(pdb.topology, pdb.positions)
forcefield = app.ForceField('solvent.xml')
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=0.5, constraints=None)
system.addForce(MonteCarloBarostat(1.0*bar, 293*kelvin, 50))
integrator = LangevinMiddleIntegrator(293.00*kelvin,   # Temperate of head bath
                                      1/picosecond, # Friction coefficient
                                      0.002*picoseconds) # Time step
simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()
frames = 1000
runtime = 100
rate = int(runtime/frames)
if rate == 0:
    rate = 1

#simulation.reporters.append(PDBReporter('solvent_output.pdb', rate, enforcePeriodicBox=True))
simulation.reporters.append(StateDataReporter('solventdata.txt', rate, progress=True, temperature=True, potentialEnergy=True, density=True,totalSteps=runtime,speed=True))
simulation.step(runtime)
