from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout, exit
from pathlib import Path

#TO-DO: Read temperature from CSV file
pdb_initfile = 'solvent_init.pdb'
ff_xml = 'solvent.xml'
if Path(pdb_initfile).is_file() and Path(ff_xml).is_file():
    pdb = app.PDBFile(pdb_initfile)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    forcefield = app.ForceField(ff_xml)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=0.5, constraints=None)
    system.addForce(MonteCarloBarostat(1.0*bar, 293*kelvin, 50))
    integrator = LangevinMiddleIntegrator(293.00*kelvin,   # Temperate of head bath
                                          1/picosecond, # Friction coefficient
                                          0.002*picoseconds) # Time step
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    frames = 200
    runtime = 500000
    rate = int(runtime/frames)
    if rate == 0:
        rate = 1

    simulation.reporters.append(StateDataReporter('solventdata.txt', rate, progress=True, temperature=True, potentialEnergy=True, density=True,totalSteps=runtime,speed=True))
    simulation.step(runtime)
else:
    print(f'Either {pdb_initfile} or {ff_xml} are missing. Not an error but check if the system is molten salt/ionic liquid')
