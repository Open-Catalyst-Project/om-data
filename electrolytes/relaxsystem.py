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

finalT = 1000  #ps. 
dt = 2.0 #fs
dt = dt/1000.0
steps = int(finalT/dt)#10000
frames = 500
rate = max(1,steps/frames)

pdb = app.PDBFile('system_init.pdb')
modeller = app.Modeller(pdb.topology, pdb.positions)
forcefield = app.ForceField('system.xml')
rdist = 2.0*nanometer
system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff)#, nonbondedCutoff=rdist, constraints=None,switchDistance=0.9*rdist)
#system.addForce(MonteCarloBarostat(1.0*bar, Temp*kelvin, 20))
integrator = LangevinMiddleIntegrator(Temp*kelvin,   # Temperate of head bath
                                      1/picosecond, # Friction coefficient
                                      dt*picosecond) # Tolerance value
simulation = app.Simulation(modeller.topology, system, integrator)#, platform)
simulation.context.setPositions(modeller.positions)
simulation.reporters.append(StateDataReporter('relaxdata.txt', rate, progress=True, density=True,totalSteps=steps,speed=True))
simulation.minimizeEnergy()
try:
    #simulation.step(steps)

    simulation.saveState('equilsystem.state')
    simulation.saveCheckpoint('equilsystem.checkpoint')
    # Get the final state
    final_state = simulation.context.getState(getPositions=True)
    # Save the final frame to a PDB file
    with open('system_equil.pdb', 'w') as f:
        PDBFile.writeFile(pdb.topology, final_state.getPositions(), f, keepIds=True)
except Exception as e:
    print(e)
    #WHen the box size becomes smaller than the cutoff, it's not a problem really so we can output this. 
    if "box size has decreased" in str(e):
        simulation.saveState('equilsystem.state')
        simulation.saveCheckpoint('equilsystem.checkpoint')
        # Get the final state
        final_state = simulation.context.getState(getPositions=True)
        # Save the final frame to a PDB file
        with open('system_equil.pdb', 'w') as f:
            PDBFile.writeFile(pdb.topology, final_state.getPositions(), f, keepIds=True)
