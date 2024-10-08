import argparse
import csv
import glob
import os
import sys

import openmm.app as app
from openmm import *
from openmm.unit import bar, kelvin, nanometer, picosecond

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



def main(row_idx: int, job_dir: str, rpmd: bool, nbeads: int):
    """
    Main job driver

    :param row_idx: Row number in the `elytes.csv` that is to be run
    :param job_dir: Directory where job files are stored and run
    """
    # Read the temperature from the CSV file
    with open("elytes.csv", "r") as f:
        systems = list(csv.reader(f))
    temp = float(systems[row_idx][4])

    dt = 0.001  # ps
    t_final = 1000 # ps, which is 500 ns
    frames = 1000
    runtime = int(t_final / dt)

    cwd = os.getcwd()
    os.chdir(os.path.join(job_dir, str(row_idx)))

    if not os.path.exists("solvent_init.pdb"):
        print("Solvent does not exist. Not an error, but check if system is a pure moltent salt/ionic liquid.")
        sys.exit()
    pdb = app.PDBFile("solvent_init.pdb")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    forcefield = app.ForceField("solvent.xml")
    rdist = 1.0 * nanometer
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=rdist,
        constraints=None,
        switchDistance=0.9 * rdist,
    )
    integrator = None
    if rpmd:
        integrator = RPMDIntegrator(
            nbeads, #number of beads
            temp * kelvin,  # Temperate of head bath
            1 / picosecond,  # Friction coefficient
            dt * picosecond
        )  # Time step
        print("hello!")
    else:
        system.addForce(MonteCarloBarostat(1.0 * bar, temp * kelvin, 100))
        integrator = LangevinMiddleIntegrator(
            temp * kelvin,  # Temperate of head bath
            1 / picosecond,  # Friction coefficient
            dt * picosecond,
        )  # Time step

    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    rate = max(1, int(runtime / frames))
    
    # Assuming you have a Simulation object named `simulation`
    state = simulation.context.getState()
    # Get the box vectors (3x3 matrix representing the vectors defining the periodic box)
    box_vectors = state.getPeriodicBoxVectors()
    # Extract the box dimensions (lengths of the box vectors)
    box_lengths = [box_vectors[i][i].value_in_unit(unit.nanometer) for i in range(3)]
    print("Box dimensions (nm):", box_lengths)
    
    simulation.minimizeEnergy()
    restart = False
    if rpmd:
        system.addForce(RPMDMonteCarloBarostat(1*bar, 100))
        simulation.context.reinitialize(preserveState=True)
    # Get name for PDBReporter (OpenMM cannot add to an existing .pdb file for restarts)
    output_pdb_basename = "solvent_output"
    other_name = sorted(glob.glob(output_pdb_basename + "*.pdb"))
    if other_name and other_name[-1] != output_pdb_basename + ".pdb":
        last_name = other_name[-1].replace(".pdb", "")
        if rpmd:
            count = int(last_name.split("_")[-3]) + 1
        else:
            count = int(last_name.split("_")[-1]) + 1
    else:
        count = 0
    output_name = f"{output_pdb_basename}_{count}"
    

    if rpmd:
        simulation.reporters.append(RPMDPDBReporter(output_name, rate, enforcePeriodicBox=True,nbeads=nbeads))
    else:
        simulation.reporters.append(
            app.PDBReporter(output_name+".pdb", rate, enforcePeriodicBox=True)
        )
    simulation.reporters.append(app.StateDataReporter('solventdata.txt', rate, progress=True, temperature=True, potentialEnergy=True, density=True,totalSteps=runtime,speed=True))
    simulation.step(
        runtime - simulation.currentStep - 10
    )  # starts at 10 for some reason, equilibration?
    # Assuming you have a Simulation object named `simulation`
    state = simulation.context.getState()
    # Get the box vectors (3x3 matrix representing the vectors defining the periodic box)
    box_vectors = state.getPeriodicBoxVectors()
    # Extract the box dimensions (lengths of the box vectors)
    box_lengths = [box_vectors[i][i].value_in_unit(unit.nanometer) for i in range(3)]
    print("Final Box dimensions (nm):", box_lengths)
    os.chdir(cwd)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameters for OMol24 Electrolytes MD"
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        required=True,
        help="Directory containing input electrolyte directories/where job files will be stored",
    )
    parser.add_argument(
        "--row_idx", type=int, help="Job specified in elytes.csv to be run"
    )
    parser.add_argument(
        "--nbeads", type=int, help="Number of beads in an RPMD simulation. Turn on --rpmd flag to use."
    )
    parser.add_argument(
        "--rpmd", action="store_true", help="Do RPMD simulation"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.row_idx, args.job_dir, args.rpmd, args.nbeads)
