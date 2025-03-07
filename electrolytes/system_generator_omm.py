"""system_generator_omm.py
Author: Muhammad R. Hasyim

A unified script for generating both pure solvent and electrolyte system configurations.
The script supports:
1. Pure solvent system generation
2. Electrolyte system generation (solvent + ions)
3. Different concentration units (mass, volume, number)
4. OpenMM simulation preparation
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pulp import LpProblem, LpVariable, lpSum, LpMinimize
import molbuilder as mb
import argparse
import textwrap
from pathlib import Path
import xml.etree.ElementTree as ET
import subprocess
import os
import string
import re
import MDAnalysis as mda
from molbuilder import calculate_mw
import json

class SystemBuilder(ABC):
    """Abstract base class for building molecular systems."""
    
    AVOGADRO = 6.023e23
    DEFAULT_DENSITY = 0.5  # g/mL
    
    def __init__(self, output_dir: str, target_atoms: int = 5000, density: float = None):
        """Initialize the system builder.
        
        Args:
            output_dir: Directory to save the generated system.
            target_atoms: Target number of atoms in the system.
            density: System density in g/mL (defaults to DEFAULT_DENSITY if None)
        """
        self.output_dir = output_dir
        self.target_atoms = target_atoms
        self.density = density if density is not None else self.DEFAULT_DENSITY
        self.box_size = None
        
    def _check_charge_balance(self) -> bool:
        """Check if the system has balanced charges.
        
        Returns:
            bool: True if charges are balanced, False otherwise
        """
        # Load ion charges from database
        cations_df = pd.read_csv('cations.csv')
        anions_df = pd.read_csv('anions.csv')
        
        total_charge = 0
        
        # Check salt charges
        for cat, num in zip(self.cations, self.n_molecules[:len(self.cations)]):
            charge = cations_df[cations_df['formula'] == cat].iloc[0]['charge']
            total_charge += charge * num
            
        for an, num in zip(self.anions, self.n_molecules[len(self.cations):len(self.cations + self.anions)]):
            charge = anions_df[anions_df['formula'] == an].iloc[0]['charge']
            total_charge += charge * num
            
        # Check solvent charges (if any are charged)
        solv_start_idx = len(self.cations + self.anions)
        for solv, num in zip(self.solvents, self.n_molecules[solv_start_idx:]):
            try:
                charge = cations_df[cations_df['formula'] == solv].iloc[0]['charge']
                total_charge += charge * num
            except:
                try:
                    charge = anions_df[anions_df['formula'] == solv].iloc[0]['charge']
                    total_charge += charge * num
                except:
                    continue  # Solvent not found in either database, assume neutral
                    
        return abs(total_charge) < 1e-13
        
    @abstractmethod
    def calculate_system_size(self) -> None:
        """Calculate the number of molecules and box size."""
        pass
    
    def generate_system(self) -> None:
        """Generate the system configuration and prepare for OpenMM."""
        self.calculate_system_size()
        
        # Check charge balance before proceeding
        if not self._check_charge_balance():
            print("Warning: System charges are not balanced!")
            print("Attempting to fix charge balance...")
            if hasattr(self, '_solve_charge_balance'):
                n_ions = self.n_molecules[:len(self.cations + self.anions)]
                self.n_molecules[:len(self.cations + self.anions)] = self._solve_charge_balance(n_ions)
                if not self._check_charge_balance():
                    raise ValueError("Failed to balance system charges!")
            else:
                raise ValueError("System charges are not balanced and no charge balancing method is available!")
        
        # Determine prefix based on system type
        prefix = 'solvent' if isinstance(self, SolventSystem) else 'system'
        
        self._create_configuration()  # This should create the LAMMPS data file
        
        self._prepare_openmm()
        self._write_metadata()  # Add metadata writing at the end

    def _create_configuration(self) -> None:
        """Create initial configuration using packmol."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Collect all species (cations, anions, solvents)
        all_species = []
        solute_count = 0
        if hasattr(self, 'cations'):
            all_species.extend(self.cations)
            all_species.extend(self.anions)
            solute_count = len(self.cations) + len(self.anions)
        all_species.extend(self.solvents)
        
        # Look for PDB files for all species and store bond information
        pdb_files = []
        molecule_bonds = []  # Store bonds for each molecule type
        for species in all_species:
            pdb_paths = [
                os.path.join('ff', f'{species}.pdb'),
                os.path.join('./ff', f'{species}.pdb'),
                os.path.join('..', 'ff', f'{species}.pdb'),
                f'{species}.pdb'
            ]
            
            pdb_path = None
            for path in pdb_paths:
                if os.path.exists(path):
                    pdb_path = path
                    break
            
            if pdb_path is None:
                raise FileNotFoundError(f"Required PDB file not found for {species}")
            
            pdb_files.append(pdb_path)
            
            # Read bonds from original PDB and store them
            bonds = []
            with open(pdb_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                if line.startswith('CONECT'):
                    atoms = [int(line[6:11])]
                    for j in range(11, len(line.rstrip()), 5):
                        if line[j:j+5].strip():
                            atoms.append(int(line[j:j+5]))
                    for k in range(1, len(atoms)):
                        bonds.append((atoms[0]-1, atoms[k]-1))
                        
            molecule_bonds.append(bonds)

        # Generate unique residue names for all species
        molres = self.generate_molres(len(all_species))
        
        # Pack molecules with specified density
        prefix = 'solvent' if isinstance(self, SolventSystem) else 'system'
        self.output_pdb = os.path.join(self.output_dir, f"{prefix}.pdb")
        
        # Calculate box volume in Å³
        box_volume = self.box_size ** 3
        box_volume_L = box_volume * 1e-27  # Convert Å³ to L
        
        # Calculate number of molecules based on box size and concentrations
        if not hasattr(self, 'n_molecules') or self.n_molecules is None:
            if isinstance(self, SolventSystem):
                # For pure solvent system:
                # n = (density * volume * NA) / MW
                total_mass = self.density * 1000 * box_volume_L  # g
                mw_weighted = sum(calculate_mw(solv) * ratio for solv, ratio in zip(self.solvents, self.solvent_ratios))
                total_moles = total_mass / mw_weighted
                self.n_molecules = np.round(total_moles * self.AVOGADRO * np.array(self.solvent_ratios)).astype(int)
            else:
                # For electrolyte system
                if self.units == 'volume':
                    # First calculate number of salt molecules from concentration
                    n_salt = np.round(self.salt_conc * box_volume_L * self.AVOGADRO).astype(int)
                    
                    # Calculate mass of ions
                    ion_mass = sum(
                        n * calculate_mw(ion) 
                        for n, ion in zip(n_salt, self.cations + self.anions)
                    )  # g/mol
                    
                    # Calculate remaining mass available for solvent 
                    # total_mass = density * volume
                    total_mass = self.density * 1000 * box_volume_L  # g
                    solvent_mass = total_mass - (ion_mass * n_salt[0] / self.AVOGADRO)
                    
                    # Calculate number of solvent molecules
                    mw_weighted = sum(calculate_mw(solv) * ratio for solv, ratio in zip(self.solvents, self.solvent_ratios))
                    solvent_moles = solvent_mass / mw_weighted
                    n_solv = np.round(solvent_moles * self.AVOGADRO * np.array(self.solvent_ratios)).astype(int)
                    
                    self.n_molecules = np.concatenate([n_salt, n_solv])
                else:
                    # Use existing n_molecules calculated in calculate_system_size
                    pass
        
        # Ensure minimum number of molecules
        min_molecules = 2
        self.n_molecules = np.maximum(self.n_molecules, min_molecules)

        # Convert molecule numbers to integers
        self.n_molecules = np.round(self.n_molecules).astype(int)

        # Write packmol input file in the output directory
        packmol_input = os.path.join(self.output_dir, 'pack.inp')
        with open(packmol_input, 'w') as f:
            f.write(f'tolerance 2.0\n')
            f.write(f'filetype pdb\n')
            f.write(f'output {self.output_pdb}\n\n')
            
            # Write structure section for each molecule type
            for i, (pdb_file, n_mol) in enumerate(zip(pdb_files, self.n_molecules)):
                f.write(f'structure {pdb_file}\n')
                f.write(f'  number {int(n_mol)}\n')
                f.write(f'  inside box 0. 0. 0. {self.box_size} {self.box_size} {self.box_size}\n')
                f.write('end structure\n\n')
                
        # Run packmol with input file from output directory
        try:
            subprocess.run([f'packmol < {packmol_input}'], shell=True, check=True)
        finally:
            # Clean up temporary file
            if os.path.exists(packmol_input):
                os.remove(packmol_input)
        
        # Load the PACKMOL-generated structure instead of individual files
        print("\nLoading packed structure:")
        u = mda.Universe(self.output_pdb)
        
        # Update residue names and chain IDs
        molres = self.generate_molres(len(self.n_molecules))  # Get list of residue names
        
        # Create mapping of molecule counts to residue names
        residue_mapping = []
        for i, n_mol in enumerate(self.n_molecules):
            residue_mapping.extend([molres[i]] * int(n_mol))  # Convert n_mol to integer
        
        # Create mapping of molecule counts to residue names and chain IDs
        chain_mapping = []
        for i, n_mol in enumerate(self.n_molecules):
            chain_mapping.extend(['A'] * int(n_mol))  # Solutes
            chain_mapping.extend(['B'] * int(n_mol))  # Solvents
            
        # Update residues and chain IDs in order
        for i, (residue, new_name, chain_id) in enumerate(zip(u.residues, residue_mapping, chain_mapping)):
            residue.resname = new_name
            residue.resid = i + 1
            residue.segment.segid = chain_id  # MDAnalysis uses segid for PDB chain ID

        # Add topology attributes to suppress warnings
        Natoms = len(u.atoms)
        u.dimensions = [self.box_size, self.box_size, self.box_size, 90, 90, 90]
        u.add_TopologyAttr('altLocs',['']*Natoms)
        u.add_TopologyAttr('icodes',['']*len(u.residues))
        u.add_TopologyAttr('tempfactors',[0]*Natoms)
        u.add_TopologyAttr('occupancies',[1.0]*Natoms)
        u.add_TopologyAttr('formalcharges',[0.0]*Natoms)  # Set formal charges to zero

        # Write updated structure
        with mda.Writer(self.output_pdb) as writer:
            writer.write(u.atoms)
            
        # Read the file content
        with open(self.output_pdb, 'r') as f:
            content = f.readlines()
            
        # Create CRYST1 record and ensure it's the first line
        cryst_line = f"CRYST1{self.box_size:9.3f}{self.box_size:9.3f}{self.box_size:9.3f}  90.00  90.00  90.00 P 1           1\n"
        
        # Remove any existing CRYST1 records
        content = [line for line in content if not line.startswith('CRYST1')]
        
        # Insert CRYST1 as the first line
        content.insert(0, cryst_line)
        
        # Propagate bonds for all molecules
        all_bonds = []
        atom_offset = 0
        
        # Iterate through each molecule type
        for mol_type, (bonds, n_mol) in enumerate(zip(molecule_bonds, self.n_molecules)):
            # Get number of atoms in this molecule type
            u_orig = mda.Universe(pdb_files[mol_type])
            n_atoms_per_mol = len(u_orig.atoms)
            
            # Add bonds for each molecule of this type
            for mol_idx in range(int(n_mol)):
                mol_offset = atom_offset + mol_idx * n_atoms_per_mol
                for bond in bonds:
                    # Convert to 1-based indexing and add offset
                    idx1 = bond[0] + mol_offset + 1
                    idx2 = bond[1] + mol_offset + 1
                    all_bonds.append((idx1, idx2))
            
            atom_offset += int(n_mol) * n_atoms_per_mol
            
        # Write CONECT records
        for bond in all_bonds:
            conect_line = f"CONECT{bond[0]:5d}{bond[1]:5d}\n"
            content.append(conect_line)
            
        # Add END record
        content.append("END\n")
        
        # Write the complete file
        with open(self.output_pdb, 'w') as f:
            f.writelines(content)

    def _prepare_openmm(self) -> None:
        """Prepare the system for OpenMM simulation using individual XML files."""
        # Determine prefix based on system type
        prefix = 'solvent' if isinstance(self, SolventSystem) else 'system'
        
        # Create dictionary of XML files
        xml_files = {}
        
        # Define possible XML file locations
        xml_paths = [
            self.output_dir,  # First check output directory
            './ff',       # Then check ./output
            '.',             # Then check current directory
            '../ff',     # Then check parent's output directory
            '..'             # Finally check parent directory
        ]
        
        def find_xml_file(name):
            """Helper function to find XML file in possible locations"""
            for path in xml_paths:
                xml_path = Path(path) / f"{name}.xml"
                if xml_path.exists():
                    return str(xml_path)
            raise FileNotFoundError(f"XML file not found for {name} in any of these locations: {[str(p) for p in xml_paths]}")
        
        # Get XML files for each component with unique residue names
        xml_contents = []
        
        # Handle both solvent and electrolyte systems the same way - combine XML files
        all_species = self.solvents if isinstance(self, SolventSystem) else (self.cations + self.anions + self.solvents)
        
        # Combine XML files for all species
        for i, species in enumerate(all_species):
            xml_file = find_xml_file(species)
            if xml_file is None:
                raise FileNotFoundError(f"No XML file found for {species}")
            
            # Read the XML and update residue names to match
            with open(xml_file, 'r') as f:
                content = f.read()
            
            # Replace residue name with the correct one (AAA, BBB, etc.)
            res_name = self.generate_molres(len(all_species))[i]
            content = re.sub(r'<Residue name="[A-Z]{3}">', f'<Residue name="{res_name}">', content)
            xml_contents.append(content)
            
        # Combine force fields with appropriate name based on system type
        output_name = "solvent.xml" if isinstance(self, SolventSystem) else "system.xml"
        output_xml = os.path.join(self.output_dir, output_name)
        self.combine_xml_forcefields(xml_contents, output_xml)

    def _write_metadata(self) -> None:
        """Write metadata about the system with only required fields."""
        # First build charge mappings from XML files
        charge_mappings = {}
        all_species = self.cations + self.anions + self.solvents
        
        for species in all_species:
            xml_path = None
            # Look for XML file in same locations as PDB files
            xml_paths = [
                os.path.join('ff', f'{species}.xml'),
                os.path.join('./ff', f'{species}.xml'),
                os.path.join('..', 'ff', f'{species}.xml'),
                f'{species}.xml'
            ]
            for path in xml_paths:
                if os.path.exists(path):
                    xml_path = path
                    break
            
            if xml_path is None:
                raise FileNotFoundError(f"XML file not found for {species}")
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # First build a mapping of atom types to charges from NonbondedForce
            charge_map = {}
            nonbonded = root.find(".//NonbondedForce")
            if nonbonded is not None:
                for atom in nonbonded.findall('Atom'):
                    type_name = atom.get('type')
                    charge = float(atom.get('charge', 0.0))
                    charge_map[type_name] = charge
            
            # Create mapping of atom names to charges for this residue
            atom_charge_map = {}
            residue = root.find(".//Residue")
            if residue is not None:
                for atom in residue.findall('Atom'):
                    type_name = atom.get('type')
                    atom_name = atom.get('name')
                    charge = charge_map.get(type_name, 0.0)
                    atom_charge_map[atom_name] = charge
            
            charge_mappings[species] = atom_charge_map
        
        # Create mapping between residue names and species
        residue_names = self.generate_molres(len(all_species))
        residue_to_species = dict(zip(residue_names, all_species))
        
        # Get partial charges from PDB
        partial_charges = []
        with open(self.output_pdb, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    resname = line[17:20].strip()  # Residue name (e.g., "AAA")
                    atom_name = line[12:16].strip()  # Atom name
                    
                    species = residue_to_species[resname]
                    charge = charge_mappings[species].get(atom_name, 0.0)
                    partial_charges.append(charge)
        
        # Create solute/solvent list
        solute_or_solvent = (
            ["solute"] * len(self.cations) +  # Cations are solutes
            ["solute"] * len(self.anions) +   # Anions are solutes
            ["solvent"] * len(self.solvents)  # Rest are solvents
        )
        
        # Create metadata with only required fields
        metadata = {
            "residue": residue_names,
            "species": all_species,
            "solute_or_solvent": solute_or_solvent,
            "partial_charges": partial_charges
        }
        
        # Write to JSON file
        json_path = os.path.join(self.output_dir, "metadata.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

    @staticmethod
    def combine_xml_forcefields(xml_contents, output_file):
        """Combine multiple OpenMM XML force field files.
        
        Args:
            xml_contents (list): List of XML content strings to combine
            output_file (str): Path to output combined XML file
        """
        # Parse the first XML file to get the base structure
        root = ET.fromstring(xml_contents[0])
        
        # For each additional XML file
        for content in xml_contents[1:]:
            other = ET.fromstring(content)
            
            # Merge AtomTypes
            atom_types = root.find('AtomTypes')
            if atom_types is not None and other.find('AtomTypes') is not None:
                for atom in other.find('AtomTypes'):
                    # Check if this atom type already exists
                    if not any(a.get('name') == atom.get('name') for a in atom_types):
                        atom_types.append(atom)
            
            # Merge Residues
            residues = root.find('Residues')
            if residues is not None and other.find('Residues') is not None:
                for residue in other.find('Residues'):
                    residues.append(residue)
            
            # Helper function to merge a force field section
            def merge_force_field(tag_name):
                force = root.find(tag_name)
                other_force = other.find(tag_name)
                if other_force is not None:
                    if force is None:
                        # Create force field if it doesn't exist
                        force = ET.SubElement(root, tag_name)
                        # Copy all attributes from the other force field
                        for attr, value in other_force.attrib.items():
                            force.set(attr, value)
                    # Copy all child elements
                    for element in other_force:
                        force.append(element)
            
            # Merge standard force fields
            standard_forces = [
                'HarmonicBondForce',
                'HarmonicAngleForce',
                'PeriodicTorsionForce',
                'NonbondedForce',
                'CustomTorsionForce',
                'CustomBondForce',
                'CustomAngleForce',
                'CustomNonbondedForce'
            ]
            
            for force_name in standard_forces:
                merge_force_field(force_name)
            
            # Check for and merge any other force fields that might be present
            for child in other:
                if child.tag not in ['AtomTypes', 'Residues'] + standard_forces:
                    # If this is a new type of force field, check if it exists in root
                    existing = root.find(child.tag)
                    if existing is None:
                        # If it doesn't exist, create it and copy all attributes and elements
                        new_force = ET.SubElement(root, child.tag)
                        # Copy attributes
                        for attr, value in child.attrib.items():
                            new_force.set(attr, value)
                        # Copy child elements
                        for element in child:
                            new_force.append(element)
                    else:
                        # If it exists, copy attributes if they don't exist
                        for attr, value in child.attrib.items():
                            if attr not in existing.attrib:
                                existing.set(attr, value)
                        # And append all elements
                        for element in child:
                            existing.append(element)
        
        # Write the combined force field
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)

    @staticmethod
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

class SolventSystem(SystemBuilder):
    """Class for generating pure solvent systems."""
    
    def __init__(self, solvents: List[str], solvent_ratios: List[float], 
                 output_dir: str, target_atoms: int = 5000, density: float = None):
        """Initialize the solvent system.
        
        Args:
            solvents: List of solvent species.
            solvent_ratios: List of solvent molar ratios.
            output_dir: Directory to save the generated system.
            target_atoms: Target number of atoms in the system.
            density: System density in g/mL (defaults to DEFAULT_DENSITY if None)
        """
        super().__init__(output_dir, target_atoms, density)
        self.solvents = solvents
        
        # Convert ratios to numpy array and normalize
        self.solvent_ratios = np.array(solvent_ratios, dtype=float)
        self.solvent_ratios = self.solvent_ratios / np.sum(self.solvent_ratios)
        
        # Set up PDB file paths
        self.pdb_files = []
        for solvent in solvents:
            pdb_path = os.path.join('ff', f'{solvent}.pdb')  # Changed from 'ff' to 'output'
            if not os.path.exists(pdb_path):
                raise FileNotFoundError(f"PDB file not found: {pdb_path}")
            self.pdb_files.append(pdb_path)
        
        # Set the first PDB as the reference for initial configuration
        self.pdb_file = self.pdb_files[0]
        self.cations = []
        self.anions = []
        self.n_molecules = None

        
    def _check_charge_balance(self) -> bool:
        """Override charge balance check for neutral solvents."""
        return True  # Solvents are assumed to be neutral

    def calculate_system_size(self) -> None:
        """Calculate number of solvent molecules and box size."""
        # Calculate molecular weights and total mass
        mw_list = [mb.calculate_mw(solv) for solv in self.solvents]  # Get MW for each solvent
        mw_avg = sum(mw * ratio for mw, ratio in zip(mw_list, self.solvent_ratios))  # Weighted avg MW

        # Convert density from g/cc to g/L (1 g/cc = 1000 g/L)
        density_gl = self.density * 1000  

        # Calculate atoms per molecule
        atoms_per_molecule = [
            sum(mb.extract_elements_and_counts(solv)[1]) for solv in self.solvents
        ]
        avg_atoms_per_mol = sum(n * ratio for n, ratio in zip(atoms_per_molecule, self.solvent_ratios))

        # Compute total molecules based on target atom count
        total_molecules = self.target_atoms / avg_atoms_per_mol  

        # Compute total mass in grams
        total_mass = (total_molecules * mw_avg) / self.AVOGADRO  

        # Compute volume in liters and convert to Å³
        volume_l = total_mass / density_gl  
        volume_ang3 = volume_l * 1e27  # Convert L → Å³

        # Compute cubic box length in Å
        self.box_size = volume_ang3 ** (1/3)  # Å
        # Compute number of molecules for each solvent
        self.n_molecules = np.round(total_molecules * self.solvent_ratios).astype(int)

        # Print system information
        print("\nSystem size calculation:")
        print(f"Total molecules: {sum(self.n_molecules)}")
        print(f"Box size: {self.box_size/10:.3f} nm")  # Convert Å → nm
        for solv, n_mol, ratio in zip(self.solvents, self.n_molecules, self.solvent_ratios):
            print(f"{solv}: {n_mol} molecules ({ratio*100:.1f}%)")

    def _write_metadata(self) -> None:
        """Write metadata about the solvent system."""
        # First build charge mappings from XML files
        charge_mappings = {}
        for species in self.solvents:
            xml_path = None
            # Look for XML file in same locations as PDB files
            xml_paths = [
                os.path.join('ff', f'{species}.xml'),
                os.path.join('./ff', f'{species}.xml'),
                os.path.join('..', 'ff', f'{species}.xml'),
                f'{species}.xml'
            ]
            for path in xml_paths:
                if os.path.exists(path):
                    xml_path = path
                    break
            
            if xml_path is None:
                raise FileNotFoundError(f"XML file not found for {species}")
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # First build a mapping of atom types to charges from NonbondedForce
            charge_map = {}
            nonbonded = root.find(".//NonbondedForce")
            if nonbonded is not None:
                for atom in nonbonded.findall('Atom'):
                    type_name = atom.get('type')
                    charge = float(atom.get('charge', 0.0))
                    charge_map[type_name] = charge
            
            # Create mapping of atom names to charges for this residue
            atom_charge_map = {}
            residue = root.find(".//Residue")
            if residue is not None:
                for atom in residue.findall('Atom'):
                    type_name = atom.get('type')
                    atom_name = atom.get('name')
                    charge = charge_map.get(type_name, 0.0)
                    atom_charge_map[atom_name] = charge
            
            # Store mapping for this species
            charge_mappings[species] = atom_charge_map
        
        # Create mapping between residue names and species
        residue_to_species = dict(zip(self.generate_molres(len(self.solvents)), self.solvents))
        
        # Now read through the system PDB and assign charges
        partial_charges = []
        with open(self.output_pdb, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    resname = line[17:20].strip()  # Residue name (e.g., "AAA")
                    atom_name = line[12:16].strip()  # Atom name
                    
                    # Use the mapping to get the correct species
                    species = residue_to_species[resname]
                    charge = charge_mappings[species].get(atom_name, 0.0)
                    partial_charges.append(charge)
        
        metadata = {
            "system_type": "pure_solvent",
            "species": self.solvents,
            "residue": self.generate_molres(len(self.solvents)),
            "solute_or_solvent": ["solvent"] * len(self.solvents),  # All species are solvents
            "composition": [int(n) for n in self.n_molecules],  # Convert np.int64 to int
            "molar_ratios": [float(r) for r in self.solvent_ratios],  # Convert np.float64 to float
            "density": float(self.density),
            "box_size_ang": float(self.box_size),
            "total_molecules": int(sum(self.n_molecules)),
            "partial_charges": partial_charges
        }
        
        # Write to JSON file
        json_path = os.path.join(self.output_dir, "solvent_metadata.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

class ElectrolyteSystem(SystemBuilder):
    """Class for generating electrolyte systems."""
    
    def __init__(self, cations: List[str], anions: List[str], solvents: List[str],
                 concentrations: Dict[str, List[float]], units: str,
                 output_dir: str, target_atoms: int = 5000, density: float = None):
        """Initialize the electrolyte system.
        
        Args:
            cations: List of cation species.
            anions: List of anion species.
            solvents: List of solvent species.
            concentrations: Dictionary with 'salt' and 'solvent' concentrations.
            units: Concentration units ('mass', 'volume', or 'number').
            output_dir: Directory to save the generated system.
            target_atoms: Target number of atoms in the system.
            density: System density in g/mL (defaults to DEFAULT_DENSITY if None)
        """
        super().__init__(output_dir, target_atoms, density)
        self.cations = cations
        self.anions = anions
        self.solvents = solvents
        self.salt_conc = np.array(concentrations['salt'])
        self.solvent_ratios = np.array(concentrations['solvent'])
        self.units = units.lower()
        self.charges = self._load_ion_charges()
        self.n_molecules = None

    def _write_metadata(self) -> None:
        """Write metadata about the electrolyte system."""
        # First build charge mappings from XML files
        charge_mappings = {}
        for species in self.cations + self.anions + self.solvents:
            xml_path = None
            # Look for XML file in same locations as PDB files
            xml_paths = [
                os.path.join('ff', f'{species}.xml'),
                os.path.join('./ff', f'{species}.xml'),
                os.path.join('..', 'ff', f'{species}.xml'),
                f'{species}.xml'
            ]
            for path in xml_paths:
                if os.path.exists(path):
                    xml_path = path
                    break
            
            if xml_path is None:
                raise FileNotFoundError(f"XML file not found for {species}")
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # First build a mapping of atom types to charges from NonbondedForce
            charge_map = {}
            nonbonded = root.find(".//NonbondedForce")
            if nonbonded is not None:
                for atom in nonbonded.findall('Atom'):
                    type_name = atom.get('type')
                    charge = float(atom.get('charge', 0.0))
                    charge_map[type_name] = charge
            
            # Create mapping of atom names to charges for this residue
            atom_charge_map = {}
            residue = root.find(".//Residue")
            if residue is not None:
                for atom in residue.findall('Atom'):
                    type_name = atom.get('type')
                    atom_name = atom.get('name')
                    charge = charge_map.get(type_name, 0.0)
                    atom_charge_map[atom_name] = charge
            
            # Store mapping for this species
            charge_mappings[species] = atom_charge_map
        
        # Create mapping between residue names and species
        all_species = self.cations + self.anions + self.solvents
        residue_to_species = dict(zip(self.generate_molres(len(all_species)), all_species))
        
        # Now read through the system PDB and assign charges
        partial_charges = []
        with open(self.output_pdb, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    resname = line[17:20].strip()  # Residue name (e.g., "AAA")
                    atom_name = line[12:16].strip()  # Atom name
                    
                    # Use the mapping to get the correct species
                    species = residue_to_species[resname]
                    charge = charge_mappings[species].get(atom_name, 0.0)
                    partial_charges.append(charge)
        
        # Create solute/solvent list
        solute_or_solvent = (
            ["solute"] * len(self.cations) +  # Cations are solutes
            ["solute"] * len(self.anions) +   # Anions are solutes
            ["solvent"] * len(self.solvents)  # Rest are solvents
        )
        
        metadata = {
            "system_type": "electrolyte",
            "cations": self.cations,
            "anions": self.anions,
            "solvents": self.solvents,
            "species": self.cations + self.anions + self.solvents,
            "residue": self.generate_molres(len(self.cations + self.anions + self.solvents)),
            "solute_or_solvent": solute_or_solvent,
            "composition": [int(n) for n in self.n_molecules],  # Convert np.int64 to int
            "concentration_units": self.units,
            "salt_concentrations": [float(c) for c in self.salt_conc],  # Convert np.float64 to float
            "solvent_ratios": [float(r) for r in self.solvent_ratios],  # Convert np.float64 to float
            "density": float(self.density),
            "box_size_ang": float(self.box_size),
            "total_molecules": int(sum(self.n_molecules)),
            "partial_charges": partial_charges  # Partial charges of all atoms
        }
        
        # Write to JSON file
        json_path = os.path.join(self.output_dir, "electrolyte_metadata.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)
                
    def _load_ion_charges(self) -> List[float]:
        """Load ion charges from database."""
        cations_df = pd.read_csv('cations.csv')
        anions_df = pd.read_csv('anions.csv')
        
        charges = []
        for cat in self.cations:
            charges.append(cations_df[cations_df['formula'] == cat].iloc[0]['charge'])
        for an in self.anions:
            charges.append(anions_df[anions_df['formula'] == an].iloc[0]['charge'])
        return charges
    
    def _solve_charge_balance(self, n_ions: np.ndarray, tolerance: int = 1) -> np.ndarray:
        """Solve for charge-neutral ion numbers."""
        prob = LpProblem("charge_balance", LpMinimize)
        x = []
        for i, n in enumerate(n_ions):
            lower_bound = 1 if n - tolerance <= 0 else n - tolerance
            x.append(LpVariable(f"x{i}", lower_bound, n + tolerance, cat='Integer'))
        
        prob += lpSum(c * v for c, v in zip(self.charges, x)) == 0
        prob.solve()
        return np.array([int(v.varValue) for v in prob.variables()])
    
    def calculate_system_size(self) -> None:
        """Calculate numbers of ions and solvent molecules and box size."""
        if self.units == 'volume':
            self._calculate_volume_based()
        elif self.units == 'mass':
            self._calculate_mass_based()
        else:  # number
            self._calculate_number_based()
            
        # Ensure charge neutrality
        n_ions = self.n_molecules[:len(self.cations + self.anions)]
        if abs(sum(np.array(self.charges) * n_ions)) > 0.1 or any(n == 0 for n in n_ions):
            self.n_molecules[:len(self.cations + self.anions)] = self._solve_charge_balance(n_ions)

        # Print system information
        print("\nSystem size calculation:")
        print(f"Total molecules: {sum(self.n_molecules)}")
        print(f"Box size: {self.box_size/10:.3f} nm")  # Convert Å → nm
        
        # Print ion information
        for cat, n_mol in zip(self.cations, self.n_molecules[:len(self.cations)]):
            print(f"Cation {cat}: {n_mol} molecules")
        for an, n_mol in zip(self.anions, self.n_molecules[len(self.cations):len(self.cations + self.anions)]):
            print(f"Anion {an}: {n_mol} molecules")
            
        # Print solvent information
        solv_start = len(self.cations + self.anions)
        for solv, n_mol, ratio in zip(self.solvents, self.n_molecules[solv_start:], self.solvent_ratios):
            print(f"Solvent {solv}: {n_mol} molecules ({ratio*100:.1f}%)")

    def _calculate_volume_based(self) -> None:
        """Calculate system size for volume-based concentrations (mol/L).
        
        This method:
        1. Uses pure solvent density from MD simulation
        2. Calculates initial number of salt molecules based on molar concentration
        3. Determines box size that satisfies concentration and minimum size requirements
        4. Calculates number of solvent molecules based on remaining volume
        
        Units:
        - salt_conc: mol/L
        - density: g/mL → converted to g/L
        - boxsize: nm
        - volume: L (after conversion from nm³)
        - box_size: Å (final output for OpenMM)
        """
        MIN_BOX_SIZE = 4  # nm
        MIN_MOLECULES = 2  # minimum number of molecules per species
        
        # Calculate solvent molecular weight (g/mol)
        solv_mw = sum(calculate_mw(solv) * ratio 
                      for solv, ratio in zip(self.solvents, self.solvent_ratios))
        
        # Get solvent density from previous MD run
        solvent_data_file = f'{self.output_dir}/solventdata.txt'
        if not Path(solvent_data_file).exists():
            raise FileNotFoundError(
                "\nError: Missing required density data file for volume-based concentration calculation.\n"
                f"Expected file: {solvent_data_file}\n"
                "\nFor volume-based concentrations, you need to:\n"
                "1. First run a pure solvent simulation\n"
                "2. Generate a solventdata.txt file containing density data\n"
                "3. Then run the electrolyte system generation\n"
                "\nAlternatively, consider using mass-based or number-based concentration units."
            )
            
        data = np.loadtxt(solvent_data_file, skiprows=1, usecols=3, delimiter=',')
        rho = np.mean(data[-10:]) * 1000  # Convert g/mL to g/L
        mol_rho = rho / solv_mw  # Convert to mol/L
        
        # Calculate initial number of salt molecules
        numsalt = self.salt_conc / min(self.salt_conc) * MIN_MOLECULES
        total_mol = np.sum(numsalt)
        conc = 0.6022 * sum(self.salt_conc)  # Convert mol/L to molecules/nm³ (0.6022 = NA/1e24)
        
        # Adjust box size to meet minimum requirement
        boxsize = (total_mol / conc) ** (1/3)  # Calculate box size in nm
        while boxsize < MIN_BOX_SIZE:
            MIN_MOLECULES += 1
            numsalt = self.salt_conc / min(self.salt_conc) * MIN_MOLECULES
            total_mol = np.sum(numsalt)
            boxsize = (total_mol / conc) ** (1/3)
        
        # Calculate final volumes and numbers
        volume = boxsize**3 * 1e-24  # Convert nm³ to L
        num_solv = rho / solv_mw * volume * self.AVOGADRO  # Calculate number of solvent molecules
        numsolv = np.round(num_solv * self.solvent_ratios).astype(int)
        
        self.n_molecules = np.concatenate([numsalt, numsolv])
        self.box_size = boxsize * 10  # Convert nm to Å for OpenMM

    def _calculate_mass_based(self) -> None:
        """Calculate system size for mass-based (molal) concentrations.
        
        This method:
        1. Calculates initial number of salt molecules based on molal concentration
        2. Determines solvent mass needed for given molality
        3. Calculates number of solvent molecules
        4. Scales system to meet target number of atoms
        
        Units:
        - salt_conc: mol/kg solvent (molal)
        - mass: kg solvent
        - solv_mw: g/mol
        - num_solv: number of molecules
        """
        MIN_MOLECULES = 2
        
        # Calculate initial number of salt molecules
        numsalt = self.salt_conc / min(self.salt_conc) * MIN_MOLECULES
        
        # Calculate solvent molecular weight and number
        solv_mw = sum(calculate_mw(solv) * ratio 
                      for solv, ratio in zip(self.solvents, self.solvent_ratios))
        
        # For molal concentration (mol/kg solvent):
        mass = numsalt[0] / self.AVOGADRO / self.salt_conc[0]  # Convert to kg solvent
        num_solv = 1000 * mass / solv_mw * self.AVOGADRO  # Convert kg→g, then to molecules
        numsolv = np.round(num_solv * self.solvent_ratios).astype(int)
        
        # Scale system to target number of atoms
        total_atoms = self._calculate_total_atoms(numsalt, numsolv)
        scale_factor = self.target_atoms / total_atoms
        if scale_factor > 1:
            numsolv = (numsolv * int(scale_factor)).astype(int)
            numsalt = (numsalt * int(scale_factor)).astype(int)
        
        self.n_molecules = np.concatenate([numsalt, numsolv])
        self._calculate_box_size()

    def _calculate_number_based(self) -> None:
        """Calculate system size for number-based concentrations (molten salts)."""
        MIN_MOLECULES = 2
        
        # Calculate initial numbers
        numsalt = self.salt_conc / min(self.salt_conc) * MIN_MOLECULES
        numsolv = self.solvent_ratios * MIN_MOLECULES
        
        # Scale system to target number of atoms
        total_atoms = self._calculate_total_atoms(numsalt, numsolv)
        scale_factor = self.target_atoms / total_atoms
        if scale_factor > 1:
            numsolv = (numsolv * int(scale_factor)).astype(int)
            numsalt = (numsalt * int(scale_factor)).astype(int)
        
        self.n_molecules = np.concatenate([numsalt, numsolv])
        self._calculate_box_size()

    def _calculate_total_atoms(self, numsalt: np.ndarray, numsolv: np.ndarray) -> int:
        """Helper method to calculate total number of atoms in the system."""
        natoms = 0
        for spec, num in zip(self.cations + self.anions, numsalt):
            elements, counts = mb.extract_elements_and_counts(spec)
            natoms += sum(counts) * num
        for spec, num in zip(self.solvents, numsolv):
            elements, counts = mb.extract_elements_and_counts(spec)
            natoms += sum(counts) * num
        return natoms

    def _calculate_box_size(self) -> None:
        """Calculate box size based on target density and number of molecules.
        
        For number-based systems (e.g., molten salts), uses salt molecular weight.
        For other systems, uses solvent molecular weight.
        
        Units:
        - density: g/mL → converted to g/L
        - mol_rho: mol/L
        - volume: Å³ (after conversion from L)
        - box_size: Å
        """
        if self.units in ['number', 'Number']:
            # For molten salts, use salt molecular weight
            salt_mw = sum(calculate_mw(salt) * conc 
                         for salt, conc in zip(self.cations + self.anions, 
                                             self.salt_conc / min(self.salt_conc)))
            rho = self.density * 1000  # Convert g/mL to g/L
            mol_rho = rho / salt_mw  # Convert to mol/L
            # Convert number of molecules to volume in Å³
            volume = sum(self.n_molecules[:len(self.cations + self.anions)]) / mol_rho * 1e27 / self.AVOGADRO
        else:
            # For solutions, use solvent molecular weight
            solv_mw = sum(calculate_mw(solv) * ratio 
                          for solv, ratio in zip(self.solvents, self.solvent_ratios))
            rho = self.density * 1000  # Convert g/mL to g/L
            mol_rho = rho / solv_mw  # Convert to mol/L
            # Convert number of molecules to volume in Å³
            volume = sum(self.n_molecules[len(self.cations + self.anions):]) / mol_rho * 1e27 / self.AVOGADRO
        
        self.box_size = volume ** (1/3)  # Calculate box length in Å

def generate_solvent_system(solvents: List[str], ratios: List[float], output: str, 
                          target_atoms: int = 5000, density: float = None) -> None:
    """Generate a pure solvent system.
        
        Args:
        solvents: List of solvent species names
        ratios: List of molar ratios for each solvent
        output: Output file prefix
        target_atoms: Target number of atoms in the system
        density: System density in g/mL (defaults to 0.5 if None)
    """
    # Normalize ratios
    ratios = np.array(ratios)
    ratios = ratios / np.sum(ratios)
    
    # Create system builder using the correct parameter names from SolventSystem class
    system = SolventSystem(
        solvents=solvents,
        solvent_ratios=ratios,
        output_dir=output,
        target_atoms=target_atoms,
        density=density
    )
    
    # Generate the system
    system.generate_system()

def generate_electrolyte_system(cations: List[str], anions: List[str], 
                              solvents: List[str], salt_conc: List[float],
                              solvent_ratios: List[float], units: str,
                              output: str, target_atoms: int = 5000,
                              density: float = None) -> None:
    """Generate an electrolyte system.
    
    Args:
        cations: List of cation species names
        anions: List of anion species names
        solvents: List of solvent species names
        salt_conc: List of salt concentrations
        solvent_ratios: List of molar ratios for solvents
        units: Concentration units ('mass', 'volume', or 'number')
        output: Output file prefix
        target_atoms: Target number of atoms in the system
        density: System density in g/mL (defaults to 0.5 if None)
    """
    # Normalize solvent ratios
    solvent_ratios = np.array(solvent_ratios)
    solvent_ratios = solvent_ratios / np.sum(solvent_ratios)
    
    # Create system builder
    system = ElectrolyteSystem(
        cations=cations,
        anions=anions,
        solvents=solvents,
        concentrations={'salt': salt_conc, 'solvent': solvent_ratios},
        units=units,
        output_dir=output,
        target_atoms=target_atoms,
        density=density
    )
    
    # Generate the system
    system.generate_system()

def process_csv_row(row, row_idx, density=None, output_dir=None):
    """Process a single row from the CSV file.
    
    Args:
        row: Row from CSV file
        row_idx: Index of the row being processed
        density: Optional system density in g/mL (defaults to 0.5 if None)
        output_dir: Optional output directory (defaults to str(row_idx))
    """
    # Use row index as output directory if not specified
    output_dir = output_dir if output_dir is not None else str(row_idx)
    
    # Extract cations and their concentrations
    cations = []
    salt_concentrations = []
    for i in range(1, 5):  # cation1 through cation4
        cation = row[f'cation{i}']
        conc = row[f'cation{i}_conc']
        if pd.notna(cation) and pd.notna(conc):
            cations.append(str(cation))
            salt_concentrations.append(float(conc))
    
    # Extract anions and their concentrations
    anions = []
    anion_concentrations = []
    for i in range(1, 5):  # anion1 through anion4
        anion = row[f'anion{i}']
        conc = row[f'anion{i}_conc']
        if pd.notna(anion) and pd.notna(conc):
            anions.append(str(anion))
            anion_concentrations.append(float(conc))
    
    # Combine salt concentrations in the correct order
    salt_concentrations = salt_concentrations + anion_concentrations
    
    # Extract solvents and their ratios
    solvents = []
    solvent_ratios = []
    for i in range(1, 5):  # solvent1 through solvent4
        solvent = row[f'solvent{i}']
        ratio = row[f'solvent{i}_ratio']
        if pd.notna(solvent) and pd.notna(ratio):
            solvents.append(str(solvent))
            solvent_ratios.append(float(ratio))
    
    # Normalize solvent ratios to sum to 1
    if solvent_ratios:
        solvent_ratios = np.array(solvent_ratios) / sum(solvent_ratios)
    
    print("\nExtracted parameters:")
    print(f"Cations: {cations}")
    print(f"Anions: {anions}")
    print(f"Solvents: {solvents}")
    print(f"Salt concentrations: {salt_concentrations}")
    print(f"Solvent ratios: {solvent_ratios}")
    print(f"Units: {row['units']}")
    
    # Package concentrations in the required dictionary format
    concentrations = {
        'salt': salt_concentrations,
        'solvent': solvent_ratios
    }
    
    # Create the system with configurable density and output directory
    system = ElectrolyteSystem(
        cations=cations,
        anions=anions,
        solvents=solvents,
        concentrations=concentrations,
        units=row['units'],
        output_dir=output_dir,  # Use the output_dir parameter
        target_atoms=5000,
        density=density
    )

    system.generate_system()

def handle_csv_mode(args):
    """Handle CSV mode by processing specified row."""
    df = pd.read_csv(args.file)
    row = df.iloc[args.row]
    process_csv_row(row, args.row, density=args.density, output_dir=args.output)

def main(mode=None, **kwargs):
    """Main function to generate molecular systems.
    
    Args:
        mode: 'csv', 'solvent', or 'electrolyte'
        **kwargs: Arguments specific to each mode:
            CSV mode:
                file: str - Path to CSV file
                row: int - Row index to process
                density: float - System density in g/mL (optional)
            
            Solvent mode:
                solvents: List[str] - List of solvent species
                ratios: List[float] - Molar ratios of solvents
                output: str - Output directory
                target_atoms: int - Target number of atoms (default: 5000)
                density: float - System density in g/mL (optional)
            
            Electrolyte mode:
                cations: List[str] - List of cation species
                anions: List[str] - List of anion species
                solvents: List[str] - List of solvent species
                salt_conc: List[float] - Salt concentrations
                solvent_ratios: List[float] - Molar ratios of solvents
                units: str - Concentration units ('mass', 'volume', or 'number')
                output: str - Output directory
                target_atoms: int - Target number of atoms (default: 5000)
                density: float - System density in g/mL (optional)
    """
    # If no arguments provided, use command line
    if mode is None:
        parser = argparse.ArgumentParser(
            description='Generate molecular systems (pure solvents or electrolytes)',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent('''
                Examples:
                    # Generate from CSV file:
                    python system_generator.py csv --file rpmd_elytes.csv --row 0
                    
                    # Generate pure solvent system:
                    python system_generator.py solvent --solvents H2O CH3OH --ratios 0.7 0.3 --output solvent_test
                    
                    # Generate electrolyte system:
                    python system_generator.py electrolyte \\
                        --cations "Li+" "Na+" \\
                        --anions "Cl-" \\
                        --solvents H2O \\
                        --salt-conc 0.5 0.3 \\
                        --solvent-ratios 1.0 \\
                        --units mass \\
                        --output electrolyte_test
                '''))

        subparsers = parser.add_subparsers(dest='mode', required=True,
                                          help='Operation mode')

        # CSV subcommand
        csv_parser = subparsers.add_parser('csv', help='Generate system from CSV file')
        csv_parser.add_argument('--file', type=str, default='rpmd_elytes.csv',
                               help='CSV file containing system specifications')
        csv_parser.add_argument('--row', type=int, required=True,
                               help='Row index in CSV file')
        csv_parser.add_argument('--density', type=float, default=None,
                               help='System density in g/mL (default: 0.5)')
        csv_parser.add_argument('--output', type=str, default=None,
                               help='Output directory (default: row number)')

        # Solvent subcommand
        solvent_parser = subparsers.add_parser('solvent', help='Generate pure solvent system')
        solvent_parser.add_argument('--solvents', nargs='+', required=True,
                                   help='List of solvent species')
        solvent_parser.add_argument('--ratios', nargs='+', type=float, required=True,
                                   help='Molar ratios of solvents')
        solvent_parser.add_argument('--output', type=str, required=True,
                                   help='Output file prefix')
        solvent_parser.add_argument('--target-atoms', type=int, default=5000,
                                   help='Target number of atoms (default: 5000)')
        solvent_parser.add_argument('--density', type=float, default=None,
                                   help='System density in g/mL (default: 0.5)')

        # Electrolyte subcommand
        elyte_parser = subparsers.add_parser('electrolyte', help='Generate electrolyte system')
        elyte_parser.add_argument('--cations', nargs='+', required=True,
                                 help='List of cation species')
        elyte_parser.add_argument('--anions', nargs='+', required=True,
                                 help='List of anion species')
        elyte_parser.add_argument('--solvents', nargs='+', required=True,
                                 help='List of solvent species')
        elyte_parser.add_argument('--salt-conc', nargs='+', type=float, required=True,
                                 help='Salt concentrations')
        elyte_parser.add_argument('--solvent-ratios', nargs='+', type=float, required=True,
                                 help='Molar ratios of solvents')
        elyte_parser.add_argument('--units', choices=['mass', 'volume', 'number'], required=True,
                                 help='Units for salt concentration')
        elyte_parser.add_argument('--output', type=str, required=True,
                                 help='Output file prefix')
        elyte_parser.add_argument('--target-atoms', type=int, default=5000,
                                 help='Target number of atoms (default: 5000)')
        elyte_parser.add_argument('--density', type=float, default=None,
                                   help='System density in g/mL (default: 0.5)')

        args = parser.parse_args()
        
        # Process arguments based on mode
        if args.mode == 'csv':
            handle_csv_mode(args)
        elif args.mode == 'solvent':
            if len(args.solvents) != len(args.ratios):
                parser.error("Number of solvents must match number of ratios")
            generate_solvent_system(args.solvents, args.ratios, args.output, 
                                  args.target_atoms, args.density)
        elif args.mode == 'electrolyte':
            if len(args.solvents) != len(args.solvent_ratios):
                parser.error("Number of solvents must match number of solvent ratios")
            generate_electrolyte_system(args.cations, args.anions, args.solvents,
                                      args.salt_conc, args.solvent_ratios,
                                      args.units, args.output, args.target_atoms, args.density)
        
        return args
    else:
        # Create args namespace from kwargs
        class Args:
            pass
        args = Args()
        args.mode = mode
        
        if mode == 'csv':
            args.file = kwargs.get('file')
            args.row = kwargs.get('row')
            args.density = kwargs.get('density')
            args.output = kwargs.get('output')  # Will be None if not specified
            if args.file is None or args.row is None:
                raise ValueError("CSV mode requires 'file' and 'row' arguments")
            
        elif mode == 'solvent':
            args.solvents = kwargs.get('solvents')
            args.ratios = kwargs.get('ratios')
            args.output = kwargs.get('output')
            args.target_atoms = kwargs.get('target_atoms', 5000)
            args.density = kwargs.get('density')
            if not all([args.solvents, args.ratios, args.output]):
                raise ValueError("Solvent mode requires 'solvents', 'ratios', and 'output' arguments")
            
        elif mode == 'electrolyte':
            args.cations = kwargs.get('cations')
            args.anions = kwargs.get('anions')
            args.solvents = kwargs.get('solvents')
            args.salt_conc = kwargs.get('salt_conc')
            args.solvent_ratios = kwargs.get('solvent_ratios')
            args.units = kwargs.get('units')
            args.output = kwargs.get('output')
            args.target_atoms = kwargs.get('target_atoms', 5000)
            args.density = kwargs.get('density')
            if not all([args.cations, args.anions, args.solvents, args.salt_conc, 
                       args.solvent_ratios, args.units, args.output]):
                raise ValueError("Electrolyte mode requires 'cations', 'anions', 'solvents', "
                               "'salt_conc', 'solvent_ratios', 'units', and 'output' arguments")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Process arguments based on mode
    if args.mode == 'csv':
        handle_csv_mode(args)
    elif args.mode == 'solvent':
        if len(args.solvents) != len(args.ratios):
            raise ValueError("Number of solvents must match number of ratios")
        generate_solvent_system(args.solvents, args.ratios, args.output, 
                              args.target_atoms, args.density)
    elif args.mode == 'electrolyte':
        if len(args.solvents) != len(args.solvent_ratios):
            raise ValueError("Number of solvents must match number of solvent ratios")
        generate_electrolyte_system(args.cations, args.anions, args.solvents,
                                  args.salt_conc, args.solvent_ratios,
                                  args.units, args.output, args.target_atoms, args.density)
    
    return args

if __name__ == "__main__":
    main()
else:
    # This allows the module to be called directly
    def __call__(*args):
        return main(list(args))

    