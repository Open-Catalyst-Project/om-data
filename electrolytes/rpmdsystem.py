import csv
import glob
import os

import openmm.app as app
from openmm import *
from openmm.unit import nanometer, bar, kelvin, picosecond
import numpy as np

class RPMDPDBReporter(object):
    """RPMDPDBReporter outputs a series of frames from an RPMD Simulation to a PDB file.

    To use it, create a RPMDPDBReporter, then add it to the Simulation's list of reporters.
    """

    def __init__(self, file_prefix, reportInterval, enforcePeriodicBox=None,nbeads=1):
        """Create an RPMDPDBReporter.

        Parameters
        ----------
        file : string
            The file to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        enforcePeriodicBox: bool
            Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.  If None (the default), it will automatically decide whether
            to translate molecules based on whether the system being simulated uses periodic boundary
            conditions.
        nbeads: int
            How many beads we want to output from the simulation. Defaults to one bead (the first one).
        """
        self._file_prefix = file_prefix
        self._reportInterval = reportInterval
        self._enforcePeriodicBox = enforcePeriodicBox
        self.nbeads = nbeads
        self._nextModel_beads= self.nbeads*[0] 
        self._out_beads = []
        for i in range(self.nbeads):
            self._out_beads.append(open(f"{file_prefix}_bead_{i+1}.pdb","w"))
        self._topology = None

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, self._enforcePeriodicBox)

    def report(self, simulation,state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation. In practice this is not used but is needed
            for compatibility. 
        """
        integrator = simulation.integrator
        if not isinstance(integrator, RPMDIntegrator):
            raise TypeError('RPMDPDBReporter only works with RPMDIntegrator.')
        
        for bead, f in enumerate(self._out_beads):
            state = integrator.getState(bead,getPositions=True,enforcePeriodicBox=self._enforcePeriodicBox)
            positions = state.getPositions()
            if self._nextModel_beads[bead]  == 0:
                app.PDBFile.writeHeader(simulation.topology, f)#f)
                self._topology = simulation.topology
                self._nextModel_beads[bead] += 1
            app.PDBFile.writeModel(simulation.topology, positions, f, self._nextModel_beads[bead])
            self._nextModel_beads[bead] += 1
            if hasattr(f, 'flush') and callable(f.flush):
                f.flush()

    def __del__(self):
        if self._topology is not None:
            for f in self._out_beads:
                app.PDBFile.writeFooter(self._topology, f)#self._out)
                f.close()


# Read the temperature from the CSV file
row_idx = int(sys.argv[1])
with open("elytes.csv", "r") as f:
    systems = list(csv.reader(f))
Temp = float(systems[row_idx][4])

dt = 0.002  # ps
t_final = 50  # ps, which is 500 ns
frames = 1000
runtime = int(t_final / dt)
nbeads = 32 #Number of beads

os.chdir(str(row_idx))
pdb = app.PDBFile("system_equil.pdb")
modeller = app.Modeller(pdb.topology, pdb.positions)
forcefield = app.ForceField("system.xml")
rdist = 1.0*nanometer
system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=rdist, constraints=None,switchDistance=0.9*rdist)

system.addForce(RPMDMonteCarloBarostat(1.0 * bar, 100))
integrator = RPMDIntegrator(
    nbeads, #number of beads
    Temp * kelvin,  # Temperate of head bath
    1 / picosecond,  # Friction coefficient
    dt * picosecond
)  # Time step

simulation = app.Simulation(modeller.topology, system, integrator)
rate = max(1, int(runtime / frames))
if os.path.exists("md.chk"):
    simulation.loadCheckpoint("md.chk")
    restart = True
else:
    #If the initial relaxation run is not using RPMD, then
    #We cannot use the generated checkpoints from before 
    #And set the positions manually
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    restart = False

# Get name for PDBReporter (OpenMM cannot add to an existing .pdb file for restarts)
output_pdb_basename = "system_output"
other_name = sorted(glob.glob(output_pdb_basename + "*.pdb"))
if other_name and other_name[-1] != output_pdb_basename + ".pdb":
    last_name = other_name[-1].replace(".pdb", "")
    count = int(last_name.split("_")[-3]) + 1
else:
    count = 0
output_name = f"{output_pdb_basename}_{count}"
simulation.reporters.append(RPMDPDBReporter(output_name, rate, enforcePeriodicBox=True,nbeads=nbeads))
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
