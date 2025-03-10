"""test_openmm.py
A script for testing OpenMM simulations with proper periodic boundary conditions
and barostat control.
"""

from openmm import *
from openmm.app import *
from openmm.unit import *
import os
import argparse
from typing import Optional, Tuple
from openmm import NonbondedForce, LangevinMiddleIntegrator, RPMDIntegrator
from openmm.app import PME
import string
import numpy as np
import sys  # Add this import if not already present
import time  # Add this import at the top
import glob
import json
from math import sqrt, pi

# Add this class before TestSimulation class
class RPMDPDBReporter(object):
    """RPMDPDBReporter outputs a series of frames from an RPMD Simulation to a PDB file."""

    def __init__(self, file_prefix, reportInterval, enforcePeriodicBox=None, nbeads=1):
        """Create an RPMDPDBReporter.
        
        Parameters
        ----------
        file_prefix : string
            The prefix for output files
        reportInterval : int
            The interval (in time steps) at which to write frames
        enforcePeriodicBox: bool
            Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.
        nbeads: int
            How many beads we want to output from the simulation. Defaults to one bead.
        """
        self._file_prefix = file_prefix
        self._reportInterval = reportInterval
        self._enforcePeriodicBox = enforcePeriodicBox
        self.nbeads = nbeads
        self._nextModel_beads = self.nbeads * [0]
        self._out_beads = []
        for i in range(self.nbeads):
            self._out_beads.append(open(f"{file_prefix}_bead_{i+1}.pdb", "w"))
        self._topology = None

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate."""
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, self._enforcePeriodicBox)

    def report(self, simulation, state):
        """Generate a report."""
        integrator = simulation.integrator
        if not isinstance(integrator, RPMDIntegrator):
            raise TypeError('RPMDPDBReporter only works with RPMDIntegrator.')
        
        for bead, f in enumerate(self._out_beads):
            state = integrator.getState(bead, getPositions=True, enforcePeriodicBox=self._enforcePeriodicBox)
            positions = state.getPositions()
            if self._nextModel_beads[bead] == 0:
                PDBFile.writeHeader(simulation.topology, f)
                self._topology = simulation.topology
                self._nextModel_beads[bead] += 1
            PDBFile.writeModel(simulation.topology, positions, f, self._nextModel_beads[bead])
            self._nextModel_beads[bead] += 1
            if hasattr(f, 'flush') and callable(f.flush):
                f.flush()

    def __del__(self):
        if self._topology is not None:
            for f in self._out_beads:
                PDBFile.writeFooter(self._topology, f)
                f.close()

class FormattedReporter:
    """Custom reporter for formatted progress output."""
    
    def _format_time(self, seconds):
        """Convert seconds to hh:mm:ss format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, True, False, False)
    
    def report(self, simulation, state):
        # Get progress
        progress = 100 * simulation.currentStep / self._totalSteps
        
        # Get simulation time in ps
        sim_time = simulation.currentStep * self._dt
        
        # Calculate speed in ns/day
        current_time = time.time()
        elapsed_time = current_time - self._lastStepTime
        elapsed_steps = simulation.currentStep - self._lastSteps
        
        if elapsed_time > 0:
            speed = (elapsed_steps * self._dt / 1000) * 86400 / elapsed_time
        else:
            speed = 0
            
        self._lastStepTime = current_time
        self._lastSteps = simulation.currentStep
        
        # Calculate remaining time in seconds
        if speed > 0:
            remaining_steps = self._totalSteps - simulation.currentStep
            # Convert to seconds:
            # (steps * dt[ps]) * (1ns/1000ps) / (speed[ns/day]) * (86400s/day)
            remaining_seconds = (remaining_steps * self._dt / 1000) / speed * 86400
            remaining = self._format_time(remaining_seconds)
        else:
            remaining = "--:--:--"
        
        # Format output
        print("{:6.0f}%     {:8.2f}    {:8.1f}     {:>8s}".format(
            progress, sim_time, speed, remaining
        ), file=self._file)

    def __init__(self, file, reportInterval, totalSteps, dt):
        self._file = file
        self._reportInterval = reportInterval
        self._totalSteps = totalSteps
        self._lastStepTime = time.time()
        self._lastSteps = 0
        self._dt = dt
        
        # Print header with proper alignment
        print("\n{:6s}     {:8s}    {:8s}     {:8s}".format(
            "Progress", "Time(ps)", "Ns/day", "Remaining"
        ), file=self._file)
        print("-" * 45, file=self._file)

def generate_molres(length):
    """Generate systematic residue names (AAA, BBB, etc.) for the given number of molecules.
    
    Args:
        length (int): Number of residue names to generate
        
    Returns:
        list: List of 3-letter residue names
    """
    molres = []
    alphabet = string.ascii_uppercase
    num_alphabet = len(alphabet)
    
    for i in range(length):
        if i < num_alphabet:
            letter = alphabet[i]
            molres.append(letter * 3)
        else:
            number = i - num_alphabet + 1
            molres.append(str(number) * 3)
    
    return molres

def calculate_geometry(positions):
    """Calculate bond lengths and angle for water molecule."""
    # Convert positions to numpy arrays
    o_pos = np.array([positions[0][i].value_in_unit(nanometer) for i in range(3)])
    h1_pos = np.array([positions[1][i].value_in_unit(nanometer) for i in range(3)])
    h2_pos = np.array([positions[2][i].value_in_unit(nanometer) for i in range(3)])
    
    # Calculate bond lengths
    r1 = np.linalg.norm(h1_pos - o_pos)
    r2 = np.linalg.norm(h2_pos - o_pos)
    
    # Calculate angle
    v1 = h1_pos - o_pos
    v2 = h2_pos - o_pos
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return r1, r2, angle

class TestSimulation:
    """Class to handle test molecular dynamics simulations."""
    
    def __init__(self, 
                 pdb_file: str,
                 xml_file: str,
                 output_dir: str = "./sim_output",
                 temperature: float = 300.0,
                 pressure: float = 1.0,
                 equil_steps: int = 5000,
                 prod_steps: int = 10000,
                 rpmd: bool = False,
                 num_replicas: int = 32,
                 t_final: float = 50.0,
                 n_frames: int = 1000,
                 dt: float = 0.001,
                 is_droplet: bool = False):
        """Initialize the test simulation.
        
        Args:
            pdb_file: Path to input PDB file
            xml_file: Path to force field XML file
            output_dir: Directory for simulation outputs
            temperature: Temperature in Kelvin
            pressure: Pressure in bar
            equil_steps: Number of equilibration steps
            prod_steps: Number of production steps
            rpmd: Whether to run RPMD simulation
            num_replicas: Number of ring polymer beads for RPMD
            t_final: Final simulation time in picoseconds (default: 50.0 ps)
            n_frames: Number of frames to output (default: 1000)
            dt: Timestep in picoseconds (default: 0.001 ps)
            is_droplet: Whether this is a droplet simulation (no PBC)
        """
        self.pdb_file = pdb_file
        self.xml_file = xml_file
        self.output_dir = output_dir
        self.temperature = temperature * kelvin
        self.pressure = pressure * bar
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps
        self.rpmd = rpmd
        self.num_replicas = num_replicas
        self.t_final = t_final
        self.n_frames = n_frames
        self.dt = dt
        self.is_droplet = is_droplet
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Checkpoint file path
        self.checkpoint_file = os.path.join(output_dir, "md.chk")
        
    def _get_restart_count(self):
        """Get the restart count based on existing PDB files."""
        if self.rpmd:
            pattern = os.path.join(self.output_dir, "rpmd_*_bead_1.pdb")
            files = sorted(glob.glob(pattern))
            if files:
                # Extract the highest number from rpmd_X_bead_1.pdb files
                numbers = [int(f.split('_')[-3]) for f in files]
                return max(numbers) + 1
            return 0
        else:
            pattern = os.path.join(self.output_dir, "trajectory_*.pdb")
            files = sorted(glob.glob(pattern))
            if files:
                # Extract the highest number from trajectory_X.pdb files
                numbers = [int(f.split('_')[-1].replace('.pdb', '')) for f in files]
                return max(numbers) + 1
            return 0
    
    def setup_system(self) -> Tuple[Simulation, bool]:
        """Set up the OpenMM simulation system."""
        print("\nLoading PDB file...")
        pdb = PDBFile(self.pdb_file)
        
        # Immediately verify positions
        if not pdb.positions:
            raise ValueError("No positions in PDB file!")
        print(f"Loaded {len(pdb.positions)} positions from PDB")
        
        # Print residue information and verify templates exist
        print("\nResidue information:")
        residues = set()
        for res in pdb.topology.residues():
            residues.add(res.name)
        print(f"Found residues: {', '.join(sorted(residues))}")
        
        print(f"\nLoading force field from: {self.xml_file}")
        forcefield = ForceField(self.xml_file)
        
        # Verify force field templates
        print("\nForce field templates:")
        for res_name in residues:
            template = forcefield._templates.get(res_name)
            if template:
                print(f"Template {res_name}: {len(template.atoms)} atoms, {len(template.bonds)} bonds")
            else:
                raise ValueError(f"No template found for residue {res_name}")
        
        print("\nSetting up modeller...")
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addExtraParticles(forcefield)
        
        # Choose nonbonded method based on droplet mode
        nonbonded_method = NoCutoff if self.is_droplet else PME
        print(f"\nUsing nonbonded method: {'NoCutoff' if self.is_droplet else 'PME'}")
        
        # Create system with appropriate nonbonded method
        if nonbonded_method == NoCutoff:
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=nonbonded_method,  # Use NoCutoff for droplet mode
                constraints=None,
                rigidWater=False
            )
        else:
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=nonbonded_method,  # Use PME for periodic system
                nonbondedCutoff=1.0 * nanometer,
                constraints=None,
                rigidWater=False,
                switchDistance=0.9 * nanometer
            )

        if self.is_droplet:
            print("\nSetting up droplet forces...")
            for force in system.getForces():
                if hasattr(force, 'setUsesPeriodicBoundaryConditions'):
                    force.setUsesPeriodicBoundaryConditions(False)
                if isinstance(force, NonbondedForce):
                    force.setNonbondedMethod(NonbondedForce.NoCutoff)
            self._add_adaptive_container_force(system)

        if self.rpmd:
            print(f"\nSetting up RPMD simulation with {self.num_replicas} beads...")
            
            # Print and set force groups
            print("\nForce groups:")
            for i, force in enumerate(system.getForces()):
                force_type = force.__class__.__name__
                print(f"Force {i}: {force_type}")
                
                if not self.is_droplet and force_type == "NonbondedForce":
                    # Split NonbondedForce for periodic systems:
                    # Direct space (short-range) -> group 1 (3 beads)
                    # Reciprocal space (long-range) -> group 2 (1 bead)
                    force.setReciprocalSpaceForceGroup(2)
                    force.setForceGroup(1)
                    print(f"  -> Direct space set to group 1 (contracted to 3 beads)")
                    print(f"  -> Reciprocal space set to group 2 (contracted to 1 bead)")
                else:
                    # All other forces use full number of beads
                    force.setForceGroup(0)
                    print(f"  -> Set to group 0 (using all {self.num_replicas} beads)")
            
            # First minimize with a regular simulation
            print("\nPerforming energy minimization...")
            min_integrator = LangevinMiddleIntegrator(
                self.temperature,
                1.0/picosecond,
                self.dt*picoseconds
            )
            min_simulation = Simulation(modeller.topology, system, min_integrator)
            min_simulation.context.setPositions(modeller.positions)
            min_simulation.minimizeEnergy(maxIterations=15000, tolerance=1.0*kilojoule/mole/nanometer)
            
            # Get minimized positions
            state = min_simulation.context.getState(getPositions=True, getEnergy=True)
            minimized_positions = state.getPositions()
            print(f"Final potential energy: {state.getPotentialEnergy()}")
            
            # Clean up minimization objects
            del min_simulation
            del min_integrator
            
            # Now create RPMD integrator and simulation
            print("\nCreating RPMD simulation...")
            rpmd_integrator = RPMDIntegrator(
                self.num_replicas,
                self.temperature,
                1.0 / picosecond,
                self.dt * picosecond,
                {0: self.num_replicas, 1: 3, 2: 1} if not self.is_droplet else {0: self.num_replicas}
            )
            
            # Add appropriate barostat for RPMD
            if not self.is_droplet:
                print("\nAdding RPMD barostat...")
                system.addForce(RPMDMonteCarloBarostat(self.pressure, 100))
            
            # Create RPMD simulation
            simulation = Simulation(modeller.topology, system, rpmd_integrator)
            
            # Check for checkpoint file
            is_restart = os.path.exists(self.checkpoint_file)
            if is_restart:
                print(f"\nFound checkpoint file: {self.checkpoint_file}")
                print("Loading simulation state from checkpoint...")
                simulation.loadCheckpoint(self.checkpoint_file)
            else:
                # Set minimized positions for all beads
                print("\nInitializing RPMD beads with minimized positions...")
                for copy in range(self.num_replicas):
                    rpmd_integrator.setPositions(copy, minimized_positions)
                    print(f"Set positions for bead {copy + 1}/{self.num_replicas}")
        
        else:
            # Regular MD setup
            if not self.is_droplet:
                print("\nAdding MonteCarloBarostat for pressure coupling...")
                system.addForce(MonteCarloBarostat(self.pressure, self.temperature, 100))
            
            integrator = LangevinMiddleIntegrator(
                self.temperature,
                1.0/picosecond,
                self.dt*picoseconds
            )
            
            simulation = Simulation(modeller.topology, system, integrator)
            
            # Check for checkpoint file
            is_restart = os.path.exists(self.checkpoint_file)
            if is_restart:
                print(f"\nFound checkpoint file: {self.checkpoint_file}")
                print("Loading simulation state from checkpoint...")
                simulation.loadCheckpoint(self.checkpoint_file)
            else:
                # Set positions and minimize
                simulation.context.setPositions(modeller.positions)
                print("\nPerforming energy minimization...")
                simulation.minimizeEnergy(maxIterations=15000, tolerance=1.0*kilojoule/mole/nanometer)
                # Get and explicitly set minimized positions
                state = simulation.context.getState(getPositions=True, getEnergy=True)
                minimized_positions = state.getPositions()
                simulation.context.setPositions(minimized_positions)
                print(f"Final potential energy: {state.getPotentialEnergy()}")

        return simulation, is_restart
    
    def _add_adaptive_container_force(self, system: System) -> None:
        """Add a spherical container force with a static force constant."""
        
        # Read droplet radius from metadata
        metadata_file = os.path.join(os.path.dirname(self.pdb_file), "electrolyte_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if 'box_size_ang' in metadata:
                    R = metadata['box_size_ang'] / 20.0  # Convert Å to nm and divide by 2
                    print(f"\nUsing radius from metadata: {R} nm")
                else:
                    R = 2.0  # Default radius in nm
                    print(f"\nNo box_size_ang in metadata, using default radius: {R} nm")
        else:
            R = 2.0  # Default radius in nm
            print(f"\nNo metadata.json found, using default radius: {R} nm")

        # Create a custom force with static force constant
        energy_expression = "100*step(r-R)*((r-R)^2); r=sqrt(x*x+y*y+z*z)"
        force = CustomExternalForce(energy_expression)
        
        # Add radius parameter
        force.addGlobalParameter('R', R)
        
        print(f"Container radius R: {R} nm")
        print("Using static force constant k = 100 kJ/mol/nm²")
        
        # Add the force to all particles
        n_particles = system.getNumParticles()
        for i in range(n_particles):
            force.addParticle(i, [])
        print(f"Added container force to {n_particles} particles")

        # Add force to system
        system.addForce(force)
    
    def run(self) -> None:
        try:
            simulation, is_restart = self.setup_system()
            
            # Calculate report interval based on desired number of frames
            report_interval = max(1, int(self.prod_steps / self.n_frames))
            
            print(f"\nSimulation parameters:")
            print(f"Total simulation time: {self.t_final} ps")
            print(f"Timestep: {self.dt} ps")
            print(f"Total steps: {self.prod_steps}")
            print(f"Output frequency: every {report_interval} steps")
            
            # Get restart count for PDB file naming
            restart_count = self._get_restart_count()
            
            # Calculate remaining steps
            current_step = simulation.currentStep
            remaining_steps = self.prod_steps - current_step
            
            if remaining_steps <= 0:
                print("\nSimulation has already completed the requested steps.")
                return

            print("\nStarting production...")
            
            # Add reporters after announcing start
            if self.rpmd:
                simulation.reporters.append(
                    RPMDPDBReporter(
                        os.path.join(self.output_dir, f"rpmd_{restart_count}"),
                        report_interval,
                        enforcePeriodicBox=True,
                        nbeads=self.num_replicas
                    )
                )
            else:
                simulation.reporters.append(
                    PDBReporter(
                        os.path.join(self.output_dir, f"trajectory_{restart_count}.pdb"),
                        report_interval,
                        enforcePeriodicBox=True
                    )
                )
            
            # Progress reporter for both RPMD and regular MD
            simulation.reporters.append(
                FormattedReporter(
                    sys.stdout,
                    report_interval,
                    self.prod_steps,
                    self.dt
                )
            )
            
            # Data reporter for both RPMD and regular MD
            simulation.reporters.append(
                StateDataReporter(
                    os.path.join(self.output_dir, "data.csv"),
                    report_interval,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    kineticEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    volume=True,
                    density=True,
                    speed=True,
                    separator=",",
                    append=is_restart
                )
            )
            
            # Add checkpoint reporter
            simulation.reporters.append(
                CheckpointReporter(
                    self.checkpoint_file,
                    report_interval
                )
            )
            
            simulation.step(remaining_steps)  # Only run the remaining steps
            
            # Save final state
            if self.checkpoint_file:
                simulation.saveState(self.checkpoint_file)
            
        except Exception as e:
            print(f"\nError during simulation: {str(e)}")
            raise

def run_simulation(pdb_file: str, xml_file: str, output_dir: str, 
                  temperature: float = 300.0, pressure: float = 1.0,
                  equil_steps: int = 5000, prod_steps: int = 5000,
                  rpmd: bool = False, num_replicas: int = 32,
                  t_final: float = 50.0, n_frames: int = 5000,
                  dt: float = 0.001, is_droplet: bool = False) -> None:
    """Run OpenMM simulation with specified parameters."""
    
    # Calculate production steps from t_final and dt if provided
    if t_final is not None:
        prod_steps = int(t_final / dt)
        print(f"\nCalculated {prod_steps} steps from t_final={t_final} ps and dt={dt} ps")
    
    test = TestSimulation(
        pdb_file, xml_file, output_dir,
        temperature=temperature,
        pressure=pressure,
        equil_steps=equil_steps,
        prod_steps=prod_steps,
        rpmd=rpmd,
        num_replicas=num_replicas,
        t_final=t_final,
        n_frames=n_frames,
        dt=dt,
        is_droplet=is_droplet
    )
    
    simulation, is_restart = test.setup_system()
    
    # Calculate report interval based on desired number of frames
    report_interval = max(1, prod_steps // n_frames)
    print(f"\nOutput settings:")
    print(f"Total steps: {prod_steps}")
    print(f"Number of frames requested: {n_frames}")
    print(f"Saving trajectory every {report_interval} steps")
    
    print("\nStarting production...")
    
    # Get restart count for PDB file naming
    restart_count = test._get_restart_count()
    
    # Add reporters after announcing start
    if rpmd:
        simulation.reporters.append(
            RPMDPDBReporter(
                os.path.join(output_dir, f"rpmd_{restart_count}"),
                report_interval,
                enforcePeriodicBox=not is_droplet,
                nbeads=num_replicas
            )
        )
    else:
        simulation.reporters.append(
            PDBReporter(
                os.path.join(output_dir, f"trajectory_{restart_count}.pdb"),
                report_interval,
                enforcePeriodicBox=not is_droplet
            )
        )
    
    # Progress reporter
    simulation.reporters.append(
        FormattedReporter(
            sys.stdout,
            report_interval,
            prod_steps,
            dt
        )
    )
    
    # Data reporter
    simulation.reporters.append(
        StateDataReporter(
            os.path.join(output_dir, "data.csv"),
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            separator=",",
            append=is_restart
        )
    )
    
    # Add checkpoint reporter
    simulation.reporters.append(
        CheckpointReporter(
            test.checkpoint_file,
            report_interval
        )
    )
    
    simulation.step(prod_steps)
    
    # Save final checkpoint
    simulation.saveCheckpoint(test.checkpoint_file)
    print("\nSimulation completed!")

def main():
    parser = argparse.ArgumentParser(description='Run test MD simulation with OpenMM')
    parser.add_argument('--pdb', required=True, help='Path to PDB file')
    parser.add_argument('--xml', required=True, help='Path to XML force field file')
    parser.add_argument('--output-dir', help='Output directory (default: ./sim_output)')  # Make optional
    parser.add_argument('--temperature', type=float, default=300.0, help='Temperature in Kelvin')
    parser.add_argument('--pressure', type=float, default=1.0, help='Pressure in bar')
    parser.add_argument('--eq-steps', type=int, default=5000, help='Number of equilibration steps')
    parser.add_argument('--prod-steps', type=int, default=10000, help='Number of production steps')
    parser.add_argument('--dt', type=float, default=0.001, help='Timestep in picoseconds (default: 0.001 ps)')
    
    # Add RPMD-related arguments
    parser.add_argument('--rpmd', action='store_true', 
                       help='Run Ring Polymer Molecular Dynamics simulation')
    parser.add_argument('--num-replicas', type=int, default=32,
                       help='Number of ring polymer beads for RPMD (default: 32)')
    
    parser.add_argument('--t-final', type=float, default=50.0,
                       help='Final simulation time in ps')
    parser.add_argument('--n-frames', type=int, default=1000,
                       help='Number of frames to output')
    
    # Add droplet mode argument
    parser.add_argument('--droplet', action='store_true',
                       help='Run simulation in droplet mode (no periodic boundary conditions)')
    
    args = parser.parse_args()
    
    return run_simulation(
        args.pdb,
        args.xml,
        args.output_dir,  # This will be None if not specified
        args.temperature,
        args.pressure,
        args.eq_steps,
        args.prod_steps,
        rpmd=args.rpmd,
        num_replicas=args.num_replicas,
        t_final=args.t_final,
        n_frames=args.n_frames,
        dt=args.dt,
        is_droplet=args.droplet  # Pass the droplet mode parameter
    )

if __name__ == "__main__":
    main() 