"""lt2omm.py

A script to convert LAMMPS Template (LT) files to OpenMM XML force field files.
This script processes LT files and generates corresponding OpenMM XML force field files with:
1. Atom types and masses
2. Residue templates (validated against PDB)
3. Bond parameters
4. Angle parameters
5. Dihedral parameters (OPLS and Fourier styles)
6. Improper parameters (CVFF style)
7. Non-bonded parameters

Author: Muhammad Risyad Hasyim, e-mail: mh7373@nyu.edu
"""

import xml.etree.ElementTree as ET
import sys
import re
import os
from pathlib import Path
import argparse
from rdkit.Chem import GetPeriodicTable
import math
import MDAnalysis as mda
import sys
import glob
import numpy as np
import shutil

# Conversion constants
kcal2kj = 4.184          # 1 kcal = 4.184 kj
ang2nm = 0.1            # 1 Angstrom = 0.1 nanometers
degree2radian = 0.0174533  # 1 degree = 0.0174533 radians

def find_elem_by_mass(target, tol=0.1):
    """Find element symbol based on atomic mass.
    
    Args:
        target (float): Target atomic mass
        tol (float): Tolerance for mass matching
        
    Returns:
        str: Element symbol or None if no match found
    """
    pt = GetPeriodicTable()
    closest_elem = None
    closest_diff = float('inf')
    
    for num in range(1, 119):
        symbol = pt.GetElementSymbol(num)
        mass = pt.GetAtomicWeight(num)
        diff = abs(mass - target)
        if diff <= closest_diff and diff <= tol:
            closest_diff = diff
            closest_elem = symbol
    return closest_elem

def parse_lt_file(lt_content):
    """Parse a LAMMPS template file and extract force field parameters.
    
    Args:
        lt_content (str): Content of the LT file to parse
        
    Returns:
        dict: Dictionary containing parsed parameters including:
            - atoms: List of atom definitions
            - bonds: List of bond definitions  
            - angles: List of angle definitions
            - dihedrals: List of dihedral definitions
            - impropers: List of improper definitions
            - masses: Dict mapping atom types to masses
            - nonbonded_params: Dict of nonbonded parameters
            - bond_coeffs: Dict of bond coefficients
            - angle_coeffs: Dict of angle coefficients
            - dihedral_coeffs: Dict of dihedral coefficients
            - improper_coeffs: Dict of improper coefficients
    """
    params = {
        'atoms': [],
        'bonds': [],
        'angles': [],
        'dihedrals': [],
        'impropers': [],
        'masses': {},
        'nonbonded_params': {},
        'bond_coeffs': {},
        'angle_coeffs': {},
        'dihedral_coeffs': {},
        'improper_coeffs': {}
    }
    
    print("\nStarting file parsing...")
    lines = lt_content.split('\n') if isinstance(lt_content, str) else lt_content
    current_section = None
    brace_level = 0
    atom_counter = 1
    
    try:
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Track braces
            brace_level += line.count('{') - line.count('}')
                
            # Track sections
            if 'write("Data Atoms")' in line:
                current_section = 'atoms'
                continue
            elif 'write_once("Data Masses")' in line:
                current_section = 'masses'
                continue
            elif 'write_once("In Settings")' in line:
                current_section = 'settings'
                continue
            elif 'write("Data Impropers")' in line:
                current_section = 'impropers'
                continue
            elif 'write("Data Dihedrals")' in line:
                current_section = 'dihedrals'
                continue
            elif 'write("Data Angles")' in line:
                current_section = 'angles'
                continue
            elif 'write("Data Bonds")' in line:
                current_section = 'bonds'
                continue
            elif brace_level == 0:
                current_section = None
                
            # Parse masses
            if current_section == 'masses' and line.startswith('@atom:'):
                parts = line.split()
                atom_type = parts[0].replace('@atom:', '')
                mass = float(parts[1])
                element = find_elem_by_mass(mass)
                params['masses'][atom_type] = {'mass': mass, 'element': element}
                print(f"Found mass and element: {atom_type} = {mass} ({element})")
                print(f"LT line: {line}")
                
            # Parse atoms
            elif current_section == 'atoms' and line.startswith('$atom:'):
                parts = line.split()
                atom_id = parts[0].replace('$atom:', '')
                atom_type = next(part.replace('@atom:', '') for part in parts if part.startswith('@atom:'))
                charge = float(parts[3])
                
                # Get mass and element from previously parsed data
                mass_data = params['masses'].get(atom_type, {'mass': 0.0, 'element': 'X'})
                
                atom = {
                    'id': str(atom_counter),
                    'orig_id': atom_id,
                    'type': atom_type,
                    'charge': charge,
                    'mass': mass_data['mass'],
                    'element': mass_data['element']
                }
                params['atoms'].append(atom)
                print(f"Added atom {atom_counter}: {atom}")
                print(f"LT line: {line}")
                atom_counter += 1

            elif current_section == 'settings' and line.startswith('angle_coeff'):
                parts = line.split()
                angle_type = parts[1].split(':')[1]
                k = float(parts[2])
                theta = float(parts[3])
                params['angle_coeffs'][angle_type] = {'k': k, 'angle': theta}
                print(f"Found angle coefficient: {angle_type} k={k} theta={theta}")
                print(f"LT line: {line}")
                
            elif current_section == 'settings' and line.startswith('bond_coeff'):
                parts = line.split()
                bond_type = parts[1].split(':')[1]
                k = float(parts[2])
                length = float(parts[3])
                params['bond_coeffs'][bond_type] = {'k': k, 'length': length}
                print(f"Found bond coefficient: {bond_type} k={k} length={length}")
                print(f"LT line: {line}")

            # Parse dihedral coefficients
            elif current_section == 'settings' and line.startswith('dihedral_coeff'):
                parts = line.split()
                dihedral_type = parts[1].replace('@dihedral:', '')
                
                # Check if we are in hybrid mode by looking for style keyword
                if parts[2] in ['opls', 'fourier']:
                    style = parts[2]
                    shift = 1
                else:
                    # No style specified, infer from format
                    shift = 0
                    # If next value is number of terms, it's fourier style
                    try:
                        int(parts[2])
                        style = 'fourier'
                    except ValueError:
                        # Otherwise assume OPLS with 4 coefficients
                        style = 'opls'
                
                if style == 'fourier':
                    nterms = int(parts[2+shift])
                    coeffs = {'style': style, 'terms': []}
                    
                    # Parse each fourier term following LAMMPS format
                    for i in range(nterms):
                        k = float(parts[3*i+3+shift]) * kcal2kj  # Convert to kJ/mol
                        n = int(parts[3*i+4+shift])  # Periodicity must be integer
                        d = float(parts[3*i+5+shift])  # Phase in degrees
                        coeffs['terms'].append({
                            'k': k,
                            'periodicity': n,
                            'phase': d
                        })
                    params['dihedral_coeffs'][dihedral_type] = coeffs
                    print(f"Added fourier dihedral coefficient: {dihedral_type} with {nterms} terms")
                    print(f"LT line: {line}")
                
                elif style == 'opls':
                    # OPLS has 4 terms with coefficients K1-K4 that need to be halved
                    coeffs = {'style': style}
                    for i in range(4):
                        k = float(parts[2+i+shift])/2 * kcal2kj  # Convert to kJ/mol
                        coeffs[f'K{i+1}'] = k
                    params['dihedral_coeffs'][dihedral_type] = coeffs 
                    print(f"Added OPLS dihedral coefficient: {dihedral_type}")
                    print(f"LT line: {line}")

            # Parse improper coefficients
            elif current_section == 'settings' and line.startswith('improper_coeff'):
                parts = line.split()
                improper_type = parts[1].replace('@improper:', '')
                k = float(parts[2])
                d = float(parts[3])
                n = float(parts[4])

                
                # Convert energy to kJ/mol and determine phase based on d
                k_kj = k * kcal2kj
                phase = math.pi if d < 0 else 0.0
                
                params['improper_coeffs'][improper_type] = {
                    'k': k_kj,
                    'periodicity': n,
                    'phase': phase
                }
                print(f"Added improper coefficient: {improper_type} k={k_kj} n={n} phase={phase}")
                print(f"LT line: {line}")
                
            elif current_section == 'bonds' and line.startswith('$bond:'):
                print("Bond section")
                parts = line.split()
                bond_id = parts[0].split(':')[1]
                bond_type = parts[1].split(':')[1]
                atom1_id = parts[2].replace('$atom:', '')
                atom2_id = parts[3].replace('$atom:', '')

                # Map atom IDs to their corresponding types
                atom1_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom1_id), None)
                atom2_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom2_id), None)
                
                # Get coefficients if available
                k = 0.0
                length = 0.0
                if bond_type in params['bond_coeffs']:
                    k = params['bond_coeffs'][bond_type]['k']
                    length = params['bond_coeffs'][bond_type]['length']
                    print(f"Applied coefficients to {bond_id}: k={k}, length={length}")
                
                bond = {
                    'id': bond_id,
                    'type': bond_type,
                    'atom1': atom1_type,
                    'atom2': atom2_type
                }
                params['bonds'].append(bond)
                print(f"Added bond: {bond}")
                print(f"LT line: {line}")

            elif current_section == 'angles' and line.startswith('$angle:'):
                parts = line.split()
                print(parts)
                angle_id = parts[0].split(':')[1]
                angle_type = parts[1].split(':')[1]
                atom1_id = parts[2].replace('$atom:', '')
                atom2_id = parts[3].replace('$atom:', '')
                atom3_id = parts[4].replace('$atom:', '')
                
                # Map atom IDs to their corresponding types from previously parsed atoms
                atom1_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom1_id), None)
                atom2_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom2_id), None)
                atom3_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom3_id), None)
                #print(atom1_type, atom2_type, atom3_type)
                #sys.exit()
                angle = {
                    'id': angle_id,
                    'type': angle_type,
                    'atom1': atom1_type,
                    'atom2': atom2_type,
                    'atom3': atom3_type
                }
                params['angles'].append(angle)
                print(f"Added angle: {angle}")
                print(f"LT line: {line}")
                
            elif current_section == 'impropers' and line.startswith('$improper:'):
                parts = line.split()
                improper_id = parts[0].split(':')[1]
                improper_type = parts[1].split(':')[1]
                atom1_id = parts[2].replace('$atom:', '')
                atom2_id = parts[3].replace('$atom:', '')
                atom3_id = parts[4].replace('$atom:', '')
                atom4_id = parts[5].replace('$atom:', '')

                # Map atom IDs to their corresponding types
                atom1_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom1_id), None)
                atom2_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom2_id), None)
                atom3_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom3_id), None)
                atom4_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom4_id), None)
                
                improper = {
                    'id': improper_id,
                    'type': improper_type,
                    'atom1': atom1_type,
                    'atom2': atom2_type,
                    'atom3': atom3_type,
                    'atom4': atom4_type
                }
                params['impropers'].append(improper)
                print(f"Added improper: {improper}")
                print(f"LT line: {line}")

            elif current_section == 'dihedrals' and line.startswith('$dihedral:'):
                parts = line.split()
                dihedral_id = parts[0].split(':')[1]
                dihedral_type = parts[1].split(':')[1]
                atom1_id = parts[2].replace('$atom:', '')
                atom2_id = parts[3].replace('$atom:', '')
                atom3_id = parts[4].replace('$atom:', '')
                atom4_id = parts[5].replace('$atom:', '')

                # Map atom IDs to their corresponding types
                atom1_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom1_id), None)
                atom2_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom2_id), None)
                atom3_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom3_id), None)
                atom4_type = next((atom['type'] for atom in params['atoms'] if atom['orig_id'] == atom4_id), None)
                
                dihedral = {
                    'id': dihedral_id,
                    'type': dihedral_type,
                    'atom1': atom1_type,
                    'atom2': atom2_type,
                    'atom3': atom3_type,
                    'atom4': atom4_type
                }
                params['dihedrals'].append(dihedral)
                print(f"Added dihedral: {dihedral}")
                print(f"LT line: {line}")
            
            # Parse pair coefficients
            elif current_section == 'settings' and line.startswith('pair_coeff'):
                parts = line.split()
                atom_type = parts[1].replace('@atom:', '')
                epsilon = float(parts[3])
                sigma = float(parts[4])
                params['nonbonded_params'][atom_type] = {
                    'epsilon': epsilon,
                    'sigma': sigma
                }
                print(f"Added nonbonded params: {atom_type} epsilon={epsilon} sigma={sigma}")
                print(f"LT line: {line}")
                
    except Exception as e:
        print(f"Error in parse_lt_file: {str(e)}")
        raise
        
    print("\n=== Parse Summary ===")
    print(f"Number of atoms: {len(params['atoms'])}")
    print(f"Atoms: {params['atoms']}")
    print(f"Improper coefficients: {params['improper_coeffs']}")
    print(f"Impropers: {params['impropers']}")
    
    return params

def parse_pdb_structure(pdb_file):
    """Parse PDB file to extract structure information including CONECT records.
    
    Args:
        pdb_file (str): Path to PDB file to parse
        
    Returns:
        dict: Dictionary containing:
            - atoms: List of atom information
            - bonds: Set of bond tuples
            - residues: Dict mapping residue names to atoms and bonds
    
    Raises:
        FileNotFoundError: If PDB file does not exist
    """
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
    try:
        u = mda.Universe(pdb_file)
        
        # Guess elements if not present
        u.guess_TopologyAttrs(context='default', to_guess=['elements'])
        
        structure = {
            'atoms': [],
            'bonds': set(),  # Using set to avoid duplicates
            'residues': {}
        }
        
        # Parse atoms and their coordinates
        for atom in u.atoms:
            structure['atoms'].append({
                'name': atom.name,
                'id': atom.id,
                'element': atom.element if hasattr(atom, 'element') else atom.name[0],
                'coords': atom.position.tolist(),  # Convert to list for JSON serialization
                'resname': atom.resname,  # Add residue name
                'resid': atom.resid  # Add residue ID
            })
            
            # Initialize residue entry if not exists
            if atom.resname not in structure['residues']:
                structure['residues'][atom.resname] = {
                    'atoms': [],
                    'bonds': set()
                }
            structure['residues'][atom.resname]['atoms'].append(atom.id)
        
        # Parse CONECT records directly from PDB file
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('CONECT'):
                    try:
                        atoms = [int(line[6:11])]  # First atom
                        # Get all connected atoms
                        for i in range(11, len(line.strip()), 5):
                            try:
                                atoms.append(int(line[i:i+5]))
                            except ValueError:
                                break
                        # Add bonds (both directions)
                        for connected in atoms[1:]:
                            bond = tuple(sorted([atoms[0], connected]))
                            structure['bonds'].add(bond)
                            
                            # Add bond to residue
                            atom1 = next(a for a in structure['atoms'] if a['id'] == bond[0])
                            atom2 = next(a for a in structure['atoms'] if a['id'] == bond[1])
                            if atom1['resname'] == atom2['resname']:
                                structure['residues'][atom1['resname']]['bonds'].add(bond)
                            
                    except Exception as e:
                        print(f"Warning: Error parsing CONECT record: {line}")
                        print(f"Error: {str(e)}")
                        continue
        
        return structure
    except Exception as e:
        print(f"Error parsing PDB file: {pdb_file}")
        print(f"Error: {str(e)}")
        return None

def validate_residue_template(lt_structure, pdb_structure):
    """Validate residue template against PDB structure.
    
    Args:
        lt_structure (dict): Structure data from LT file
        pdb_structure (dict): Structure data from PDB file
        
    Returns:
        tuple: (bool, str) indicating success/failure and message
    """
    # Check number of atoms
    if len(lt_structure['atoms']) != len(pdb_structure['atoms']):
        return False, f"Atom count mismatch: LT={len(lt_structure['atoms'])} PDB={len(pdb_structure['atoms'])}"
    
    # Check coordinates (within tolerance)
    tolerance = 0.1  # Angstroms
    for lt_atom, pdb_atom in zip(lt_structure['atoms'], pdb_structure['atoms']):
        dist = np.linalg.norm(np.array(lt_atom['coords']) - pdb_atom['coords'])
        if dist > tolerance:
            return False, f"Coordinate mismatch for atom {lt_atom['id']}: distance={dist:.3f}"
    
    # Check bonds
    lt_bonds = set(tuple(sorted([int(b['atom1']), int(b['atom2'])])) for b in lt_structure['bonds'])
    if lt_bonds != pdb_structure['bonds']:
        return False, "Bond connectivity mismatch"
    
    return True, "Validation successful"

def write_xml_forcefield(params, output_file, pdb_structure):
    """Write OpenMM XML force field file.
    
    Args:
        params (dict): Force field parameters from LT file
        output_file (str): Path to output XML file
        pdb_structure (dict): Structure data from PDB file
    """
    print("\nWriting XML forcefield:")
    
    # Debug print atom types
    print("\nAtom types dictionary:")
    for atom in params['atoms']:
        print(f"  {atom['id']} ({atom.get('orig_id', 'N/A')}) -> {atom['type']}")
    
    print("\nImpropers to write:")
    for imp in params['impropers']:
        print(f"  {imp}")
    
    root = ET.Element('ForceField')
    
    # Create a mapping of atom IDs to their types
    atom_types = {atom['id']: atom['type'] for atom in params['atoms']}
    print("\nAtom ID to Type mapping:")
    for atom_id, atom_type in atom_types.items():
        print(f"  {atom_id} -> {atom_type}")
        
    # Add AtomTypes section
    atoms_force = ET.SubElement(root, 'AtomTypes')
    for atom in params['atoms']:
        atom_type = ET.SubElement(atoms_force, 'Type')
        atom_type.set('name', atom['type'])
        atom_type.set('class', atom['type'])
        atom_type.set('element', params['masses'][atom['type']]['element'])
        atom_type.set('mass', str(params['masses'][atom['type']]['mass']))
        print(f"Added atom type: {atom['type']} ({atom['element']})")
    
    # Add Residues section
    residues = ET.SubElement(root, 'Residues')
    for resname, res_data in pdb_structure['residues'].items():
        residue = ET.SubElement(residues, 'Residue')
        residue.set('name', resname)
        # Add atoms
        for atom_id in res_data['atoms']:
            atom_data = next(a for a in pdb_structure['atoms'] if a['id'] == atom_id)
            atom = ET.SubElement(residue, 'Atom')
            atom.set('name', atom_data['name'])
            atom.set('type', atom_types[str(atom_id)])
            
        # Add bonds
        for bond in res_data['bonds']:
            atom1_data = next(a for a in pdb_structure['atoms'] if a['id'] == bond[0])
            atom2_data = next(a for a in pdb_structure['atoms'] if a['id'] == bond[1])
            bond_elem = ET.SubElement(residue, 'Bond')
            bond_elem.set('atomName1', atom1_data['name'])
            bond_elem.set('atomName2', atom2_data['name'])
    
    # Add HarmonicBondForce section
    print("\nBonds to write:")
    if not params['bonds']:
        print("  No bonds found")
    else:
        print(f"  {params['bonds']}")
    if params['bonds']:
        print(f"\nNumber of bonds to write: {len(params['bonds'])}")
        bond_force = ET.SubElement(root, 'HarmonicBondForce')
        for bond in params['bonds']:
           
            bond_entry = ET.SubElement(bond_force, 'Bond')
            bond_entry.set('type1', bond['atom1'])
            bond_entry.set('type2', bond['atom2'])
            # Convert k from kcal/mol/Å² to kJ/mol/nm²
            k_kj = 2*float(params['bond_coeffs'][bond['type']]['k']) * kcal2kj * 100  # *100 for Å² to nm²
            bond_entry.set('k', f"{k_kj:.1f}")
            # Convert length from Å to nm
            length_nm = float(params['bond_coeffs'][bond['type']]['length']) * ang2nm
            bond_entry.set('length', f"{length_nm:.6f}")
            print(f"  Added bond: {bond['atom1']}-{bond['atom2']} "
                  f"({bond['type']}-{bond['type']})")
    
    # Add HarmonicAngleForce section
    if params['angles']:
        print(f"\nNumber of angles to write: {len(params['angles'])}")
        angle_force = ET.SubElement(root, 'HarmonicAngleForce')
        for angle in params['angles']:
            
            angle_entry = ET.SubElement(angle_force, 'Angle')
            angle_entry.set('type1', angle['atom1'])
            angle_entry.set('type2', angle['atom2']) 
            angle_entry.set('type3', angle['atom3'])
            # Convert k from kcal/mol/rad² to kJ/mol/rad²
            k_kj = 2*float(params['angle_coeffs'][angle['type']]['k']) * kcal2kj 
            angle_entry.set('k', f"{k_kj:.1f}")
            # Convert angle from degrees to radians
            theta_rad = float(params['angle_coeffs'][angle['type']]['angle']) * degree2radian
            angle_entry.set('angle', f"{theta_rad:.6f}")
            print(f"  Added angle: {angle['atom1']}-{angle['atom2']}-{angle['atom3']}")
    # Add PeriodicTorsionForce section for dihedrals and impropers

    if params['dihedrals'] or params['impropers']:
        periodic_force = ET.SubElement(root, 'PeriodicTorsionForce')
        
        # Write dihedrals
        for dihedral in params['dihedrals']:
            proper = ET.SubElement(periodic_force, 'Proper')
            proper.set('type1', dihedral['atom1'])
            proper.set('type2', dihedral['atom2'])
            proper.set('type3', dihedral['atom3']) 
            proper.set('type4', dihedral['atom4'])
            
            # Handle dihedral coefficients
            if dihedral['type'] in params['dihedral_coeffs']:
                coeff = params['dihedral_coeffs'][dihedral['type']]
                if coeff['style'] == 'opls':
                    # OPLS style has K1-K4 terms
                    for i in range(1,5):
                        k = coeff[f'K{i}']
                        if abs(k) > 1e-6:  # Only write non-zero terms
                            proper.set(f'k{i}', f"{k:.6f}")
                            proper.set(f'periodicity{i}', f"{i}")
                            phase = "3.141592653589793" if i % 2 == 0 else "0.0"
                            proper.set(f'phase{i}', phase)
                            
                elif coeff['style'] == 'fourier':
                    # Fourier style has multiple terms with k, periodicity, phase
                    for i, term in enumerate(coeff['terms'], start=1):
                        if abs(term['k']) > 1e-6:  # Only write non-zero terms
                            proper.set(f'k{i}', f"{term['k']:.6f}")
                            if term['periodicity'] < 0:
                                proper.set(f'periodicity{i}', f"{-term['periodicity']}")
                                proper.set(f'phase{i}', f"{-term['phase']}")
                            else:
                                proper.set(f'periodicity{i}', f"{term['periodicity']}")
                                proper.set(f'phase{i}', f"{term['phase']}")
        
        # Write impropers
        for improper in params['impropers']:
            improper_elem = ET.SubElement(periodic_force, 'Improper')
            # Print for debugging
            print(f"Improper atom1: {improper['atom1']}")
            print(f"Atom types: {atom_types}")
            
            # Get the atom type from the ID mapping
            atom1_type = atom_types.get(improper['atom1'], improper['atom1'])
            atom2_type = atom_types.get(improper['atom2'], improper['atom2']) 
            atom3_type = atom_types.get(improper['atom3'], improper['atom3'])
            atom4_type = atom_types.get(improper['atom4'], improper['atom4'])
            
            improper_elem.set('type1', atom1_type)
            improper_elem.set('type2', atom2_type)
            improper_elem.set('type3', atom3_type)
            improper_elem.set('type4', atom4_type)
            
            if improper['type'] in params['improper_coeffs']:
                coeff = params['improper_coeffs'][improper['type']]
                improper_elem.set('k1', f"{coeff['k']:.6f}")
                improper_elem.set('periodicity1', str(int(coeff['periodicity'])))
                improper_elem.set('phase1', f"{coeff['phase']:.6f}")

                # Add NonbondedForce section
    nb_force = ET.SubElement(root, 'NonbondedForce')
    nb_force.set('coulomb14scale', '0.5')
    nb_force.set('lj14scale', '0.5')
    for atom in params['atoms']:
        atom_type = ET.SubElement(nb_force, 'Atom')
        atom_type.set('type', atom['type'])
        atom_type.set('charge', str(atom['charge']))
        # Convert epsilon from kcal/mol to kJ/mol

        epsilon_kj = params['nonbonded_params'][atom['type']]['epsilon'] * kcal2kj
        # Convert sigma from Angstroms to nanometers
        sigma_nm = params['nonbonded_params'][atom['type']]['sigma'] * ang2nm
        atom_type.set('sigma', f"{sigma_nm:.6f}")
        atom_type.set('epsilon', f"{epsilon_kj:.6f}")
        print(f"Added nonbonded params for {atom['type']}: charge={atom['charge']}, "
              f"epsilon={epsilon_kj:.6f}, sigma={sigma_nm:.6f}")
    
    # Write the XML file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"\nWritten XML file: {output_file}")

def process_lt_folder(input_folder, output_folder):
    lt_files = glob.glob(os.path.join(input_folder, "*.lt"))
    print(f"\nFound {len(lt_files)} LT files to process\n")
    
    os.makedirs(output_folder, exist_ok=True)
    
    for lt_path in lt_files:
        molecule_name = os.path.basename(lt_path).replace('.lt', '')
        print(f"\nProcessing {molecule_name}...")
        
        try:
            with open(lt_path, 'r') as f:
                lt_content = f.read()
                print(f"Read {len(lt_content)} bytes from {lt_path}")
                print("First 100 characters:")
                print(lt_content[:100])
            
            params = parse_lt_file(lt_content)
            
            if params.get('atoms'):
                xml_output = os.path.join(output_folder, f"{molecule_name}.xml")
                pdb_path = lt_path.replace('.lt', '.pdb')
                
                if os.path.exists(pdb_path):
                    # Parse PDB structure first
                    pdb_structure = parse_pdb_structure(pdb_path)
                    if not pdb_structure:
                        print(f"Warning: Failed to parse PDB file: {pdb_path}, skipping...")
                        continue
                        
                    write_xml_forcefield(params, xml_output, pdb_structure)
                    
                    pdb_output = os.path.join(output_folder, f"{molecule_name}.pdb")
                    shutil.copy2(pdb_path, pdb_output)
                    print(f"Copied PDB file: {pdb_output}")
                else:
                    print(f"Warning: PDB file not found: {pdb_path}, skipping...")
            else:
                print(f"Warning: No atoms found in {lt_path}, skipping...")
                
        except Exception as e:
            print(f"Error processing {lt_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

def main():
    parser = argparse.ArgumentParser(description='Convert LT files to OpenMM XML force field files')
    parser.add_argument('input', help='Input LT file or folder containing LT files')
    parser.add_argument('-o', '--output', help='Output XML file or folder for XML files')
    parser.add_argument('--batch', action='store_true', help='Process all LT files in input folder')
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.output:
            print("Error: Output folder must be specified in batch mode")
            return
        process_lt_folder(args.input, args.output)
    else:
        if not args.output:
            print("Error: Output file must be specified")
            return
            
        lt_file = args.input
        base_name = os.path.splitext(os.path.basename(lt_file))[0]
        pdb_file = os.path.join(os.path.dirname(lt_file), f"{base_name}.pdb")
        
        if not os.path.exists(pdb_file):
            print(f"Error: PDB file not found: {pdb_file}")
            return
            
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        output_pdb = os.path.join(output_dir if output_dir else '.', os.path.basename(pdb_file))
        
        try:
            # Parse PDB file first
            pdb_structure = parse_pdb_structure(pdb_file)
            if not pdb_structure:
                print(f"Error: Failed to parse PDB file: {pdb_file}")
                return
                
            # Then parse LT file
            with open(lt_file, 'r') as f:
                lt_content = f.read()
            params = parse_lt_file(lt_content)
            params['lt_file'] = lt_file
            params['pdb_file'] = pdb_file
            
            # Pass both to write_xml_forcefield
            write_xml_forcefield(params, args.output, pdb_structure)
            if os.path.exists(pdb_file):
                shutil.copy2(pdb_file, output_pdb)
                print(f"Successfully wrote {args.output} and copied {output_pdb}")
        except Exception as e:
            print(f"Error processing {lt_file}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()