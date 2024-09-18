from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout, exit
from pathlib import Path
import csv 

#TO-DO: Read temperature from CSV file
pdb_initfile = 'solvent_init.pdb'
ff_xml = 'solvent.xml'
if Path(pdb_initfile).is_file() and Path(ff_xml).is_file():
    row_idx  = int(sys.argv[1]) 
    # Load the CSV file containing systems to simulate
    with open("../elytes.csv", "r") as f:
        systems = list(csv.reader(f))
    Temp = float(systems[row_idx][4])
    pdb = app.PDBFile(pdb_initfile)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    forcefield = app.ForceField(ff_xml)
    rdist = 1.0*nanometer
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=rdist, constraints=None,switchDistance=0.9*rdist)
    system.addForce(MonteCarloBarostat(1.0*bar, Temp*kelvin, 50))
    dt = 2.0 #fs
    dt = dt/1000.0
    integrator = LangevinMiddleIntegrator(Temp*kelvin,   # Temperate of head bath
                                          1/picosecond, # Friction coefficient
                                          dt*picoseconds) # Time step
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    frames = 500
    runtime = 1000 #ps
    runtime /= dt
    rate = int(runtime/frames)
    if rate == 0:
        rate = 1

    simulation.reporters.append(StateDataReporter('solventdata.txt', rate, progress=True, temperature=True, potentialEnergy=True, density=True,totalSteps=runtime,speed=True))
    simulation.step(runtime)
    # Get the final state
    final_state = simulation.context.getState(getPositions=True)
    # Save the final frame to a PDB file
    with open('solvent_output.pdb', 'w') as f:
        PDBFile.writeFile(pdb.topology, final_state.getPositions(), f, keepIds=True)
else:
    print(f'Either {pdb_initfile} or {ff_xml} are missing. Not an error but check if the system is molten salt/ionic liquid or if concentrations are specified by molalities')
