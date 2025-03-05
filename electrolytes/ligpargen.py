import requests
from bs4 import BeautifulSoup
import os
import time
import re
import warnings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys
import urllib3
import pandas as pd
import argparse
import xml.etree.ElementTree as ET

# Suppress SSL warnings since we're intentionally using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False
    warnings.warn(
        "RDKit not found. SMILES validation will be disabled. "
        "Install RDKit for better SMILES validation: pip install rdkit"
    )

def validate_smiles(smiles):
    """Validate a SMILES string using RDKit if available.

    Args:
        smiles (str): SMILES string to validate

    Returns:
        tuple: (bool, str) - (is_valid, message)
            - is_valid: True if SMILES is valid, False otherwise
            - message: Validation message or error description

    Note:
        If RDKit is not available, returns True with a warning message
        Performs additional checks for molecule size and supported elements
    """
    if not HAVE_RDKIT:
        return True, "SMILES validation skipped (RDKit not available)"
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES string"
    
    # Additional checks for molecule complexity
    num_atoms = mol.GetNumAtoms()
    if num_atoms > 100:  # This is a reasonable limit for LigParGen
        return False, f"Molecule too large ({num_atoms} atoms). LigParGen works best with smaller molecules."
    
    # Check for unsupported elements and ensure presence of carbon
    supported_elements = {'C', 'H', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Si'}  # Common elements supported by LigParGen
    elements = set(atom.GetSymbol() for atom in mol.GetAtoms())
    
    # Check if molecule contains at least one carbon atom
    if 'C' not in elements:
        return False, "Molecule must contain at least one carbon atom"
    # Check for presence of C-H bonds by looking at explicit and implicit hydrogens
    # For molecules like urea, we need to check both the molecule structure and formula
    has_ch_bond = False
    
    # First check the molecule structure
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and (atom.GetNumExplicitHs() > 0 or atom.GetNumImplicitHs() > 0):
            has_ch_bond = True
            break
    
    # If no C-H bonds found in structure, check molecular formula for H presence
    if not has_ch_bond:
        mol_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        # If formula contains both C and H, assume C-H bonds exist
        if 'C' in mol_formula and 'H' in mol_formula:
            has_ch_bond = True
            
    if not has_ch_bond:
        return False, "Molecule must contain at least one C-H bond for LigParGen"
    # Check for unsupported elements
    unsupported = elements - supported_elements
    if unsupported:
        return False, f"Contains unsupported elements: {', '.join(unsupported)}"
    
    return True, "SMILES validation passed"

def create_session(retries=5, backoff_factor=2.0, verify_ssl=False):
    """Create a requests session with retry logic.

    Args:
        retries (int, optional): Maximum number of retries. Defaults to 5.
        backoff_factor (float, optional): Factor to calculate delay between retries. Defaults to 2.0.
        verify_ssl (bool, optional): Whether to verify SSL certificates. Defaults to False.

    Returns:
        requests.Session: Configured session object with retry logic

    Note:
        Uses exponential backoff for retries
        Handles common HTTP error codes (500, 502, 503, 504)
    """
    session = requests.Session()
    
    if not verify_ssl:
        warnings.warn(
            "SSL certificate verification is disabled. This is insecure and not recommended for production use.",
            UserWarning,
            stacklevel=2
        )
    
    # Configure retry strategy with longer delays and fewer retries
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,  # Increased backoff
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=frozenset(['GET', 'POST']),
        respect_retry_after_header=True,
        raise_on_status=False  # Don't raise exceptions on status, we'll handle them
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.verify = verify_ssl
    
    return session

def check_server_status(session):
    """Check if the LigParGen server is accessible"""
    try:
        # Use the form endpoint directly instead of the main page
        # and reduce timeout to 3 seconds
        response = session.get("https://zarbi.chem.yale.edu/cgi-bin/results_lpg.py", 
                             timeout=3, verify=session.verify)
        return response.status_code != 500  # Consider any non-500 response as available
    except requests.exceptions.RequestException:
        return False

def save_debug_info(response, output_dir, prefix="debug"):
    """Save debug information when server interaction fails"""
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save response content
    content_file = os.path.join(debug_dir, f"{prefix}_response_{timestamp}.html")
    with open(content_file, 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    # Save response headers and info
    info_file = os.path.join(debug_dir, f"{prefix}_info_{timestamp}.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"URL: {response.url}\n")
        f.write(f"Status Code: {response.status_code}\n")
        f.write("Headers:\n")
        for k, v in response.headers.items():
            f.write(f"{k}: {v}\n")
    
    return content_file, info_file

def process_xml_file(xml_path, original_prefix):
    """Process and modify an XML file containing force field parameters.
    
    Converts OPLS atom types to custom types and updates all references throughout the file.
    Also converts numeric atom indices to atom names in bonds.

    Args:
        xml_path (str): Path to the XML file to process
        original_prefix (str): Prefix to use in the new type names

    Returns:
        bool: True if processing was successful, False otherwise

    Note:
        The function modifies the XML file in place, creating backup is recommended before calling.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # First: Create mappings for atom indices to names and create new type names
        idx_to_name = {}
        opls_to_type = {}  # Map OPLS types to new type names
        
        # Create new type names and map atoms
        for idx, atom in enumerate(root.findall(".//Residue/Atom")):
            name = atom.get('name')  # e.g., "C00"
            opls_type = atom.get('type')  # e.g., "opls_800"
            # Extract element symbol by taking characters until we hit a digit
            element = ''.join(c for c in name if not c.isdigit()).lower()  # Get lowercase element symbol
            new_type = f"type{idx+1}_{element.lower()}_{original_prefix}"
            
            idx_to_name[str(idx)] = name
            opls_to_type[opls_type] = new_type
            
            # Update the atom type in Residue
            atom.set('type', new_type)
        
        # Update bonds to use atom names instead of indices
        for bond in root.findall(".//Residue/Bond"):
            if 'from' in bond.attrib and 'to' in bond.attrib:
                from_idx = bond.get('from')
                to_idx = bond.get('to')
                
                # Remove old attributes
                bond.attrib.pop('from')
                bond.attrib.pop('to')
                
                # Add new attributes using atom names
                if from_idx in idx_to_name and to_idx in idx_to_name:
                    bond.set('atomName1', idx_to_name[from_idx])
                    bond.set('atomName2', idx_to_name[to_idx])
        
        # Update AtomTypes
        for atom_type in root.findall(".//AtomTypes/Type"):
            old_type = atom_type.get('name')  # opls_XXX
            if old_type in opls_to_type:
                atom_type.set('name', opls_to_type[old_type])
        
        # Update force field parameters to use new type names
        # HarmonicBondForce
        # for bond in root.findall(".//HarmonicBondForce/Bond"):
        #     if 'class1' in bond.attrib and 'class2' in bond.attrib:
        #         class1 = bond.get('class1')
        #         class2 = bond.get('class2')
        #         # Convert class names to OPLS types
        #         opls_type1 = f"opls_{class1.lower().replace('br', '')}"
        #         opls_type2 = f"opls_{class2.lower().replace('br', '')}"
        #         if opls_type1 in opls_to_type and opls_type2 in opls_to_type:
        #             bond.attrib.pop('class1')
        #             bond.attrib.pop('class2')
        #             bond.set('type1', opls_to_type[opls_type1])
        #             bond.set('type2', opls_to_type[opls_type2])
        
        # # HarmonicAngleForce
        # for angle in root.findall(".//HarmonicAngleForce/Angle"):
        #     for i in range(1, 4):
        #         if f'class{i}' in angle.attrib:
        #             class_val = angle.get(f'class{i}')
        #             opls_type = f"opls_{class_val.lower().replace('br', '')}"
        #             if opls_type in opls_to_type:
        #                 angle.attrib.pop(f'class{i}')
        #                 angle.set(f'type{i}', opls_to_type[opls_type])
        
        # # PeriodicTorsionForce
        # for torsion in root.findall(".//PeriodicTorsionForce/*"):
        #     for i in range(1, 5):
        #         if f'class{i}' in torsion.attrib:
        #             class_val = torsion.get(f'class{i}')
        #             opls_type = f"opls_{class_val.lower().replace('br', '')}"
        #             if opls_type in opls_to_type:
        #                 torsion.attrib.pop(f'class{i}')
        #                 torsion.set(f'type{i}', opls_to_type[opls_type])
        
        # NonbondedForce
        for atom in root.findall(".//NonbondedForce/Atom"):
            old_type = atom.get('type')
            if old_type in opls_to_type:
                atom.set('type', opls_to_type[old_type])
        
        # Write the updated XML
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        return True
        
    except Exception as e:
        print(f"Warning: Failed to process XML file {xml_path}: {e}")
        return False

def get_filename_from_response(response, format_name, name):
    """Extract or construct filename from server response.

    Args:
        response (requests.Response): Server response object
        format_name (str): Format identifier (e.g., 'OpenMM', 'PDB', 'LT')
        name (str): Base name to use if filename cannot be extracted from response

    Returns:
        Union[str, tuple, None]: 
            - str: filename for non-XML files
            - tuple: (filename, prefix) for XML files
            - None: if format is not supported

    Note:
        Handles Content-Disposition headers and constructs reasonable defaults
    """
    # Try to get filename from Content-Disposition header
    if 'Content-Disposition' in response.headers:
        content_disp = response.headers['Content-Disposition']
        filename_match = re.search(r'filename="?([^";]+)"?', content_disp)
        if filename_match:
            orig_filename = filename_match.group(1)
            # Extract the prefix from original filename (e.g., "UNK_234")
            prefix_match = re.match(r'([^\.]+)', orig_filename)
            prefix = prefix_match.group(1).lower() if prefix_match else ''
            
            # Keep only specific file extensions
            extension = os.path.splitext(orig_filename)[1].lower()
            if extension in ['.lt', '.pdb', '.xml']:
                new_filename = f"{name}{extension}"
                # Store the prefix for XML processing
                if extension == '.xml':
                    return new_filename, prefix
                return new_filename
            return None
    
    # If no Content-Disposition, only handle specific formats
    if format_name == 'OpenMM':
        return f"{name}.xml", "unk"
    elif format_name in ['PDB', 'LT']:
        return f"{name}.{format_name.lower()}"
    return None

def submit_to_ligpargen_smiles(smiles, name, charge=0, charge_model="cm1a", 
                             optimization_iterations=0, output_dir="./output", 
                             verify_ssl=False, debug=True):
    """Submit a SMILES string to LigParGen web service and retrieve force field parameters.

    Args:
        smiles (str): SMILES representation of the molecule
        name (str): Name prefix for output files
        charge (int, optional): Molecular charge. Defaults to 0.
        charge_model (str, optional): Charge model to use ('cm1a' or 'cm1a-bcc'). Defaults to "cm1a".
        optimization_iterations (int, optional): Number of geometry optimization iterations. Defaults to 0.
        output_dir (str, optional): Directory to save output files. Defaults to "./output".
        verify_ssl (bool, optional): Whether to verify SSL certificates. Defaults to False.
        debug (bool, optional): Whether to save debug information. Defaults to True.

    Returns:
        dict: Dictionary mapping filenames to their full paths for all downloaded files

    Raises:
        Exception: If there's an error in processing the SMILES or communicating with the server
    """
    session = create_session(retries=5, backoff_factor=2.0, verify_ssl=verify_ssl)
    
    # Set headers to mimic a browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': 'https://zarbi.chem.yale.edu',
        'Referer': 'https://zarbi.chem.yale.edu/ligpargen/'
    })
    
    try:
        # First try SMILES approach
        files = {
            'molpdbfile': ('', '', 'application/octet-stream')
        }

        data = {
            'smiData': smiles,
            'checkopt': f" {optimization_iterations} ",
            'chargetype': charge_model,
            'dropcharge': f" {int(charge)} "
        }

        print(f"Submitting data to LigParGen:")
        print(f"- SMILES: {smiles}")
        print(f"- Charge: {int(charge)}")
        print(f"- Charge model: {charge_model}")
        print(f"- Optimization iterations: {optimization_iterations}")

        response = session.post(
            'https://zarbi.chem.yale.edu/cgi-bin/results_lpg.py',
            files=files,
            data=data,
            verify=verify_ssl,
            timeout=60
        )
        
        if debug:
            save_debug_info(response, output_dir)
            
        soup = BeautifulSoup(response.text, 'html.parser')
        download_forms = soup.find_all('form')
        
        # If no download forms found, try MOL and PDB approaches
        if not download_forms:
            print("SMILES approach failed, trying MOL approach...")
            
            if not HAVE_RDKIT:
                raise Exception("RDKit required for MOL/PDB generation")
                
            # Generate 3D structure using RDKit
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # First try MOL file approach
            temp_mol = os.path.join(output_dir, f"{name}_temp.mol")
            writer = Chem.SDWriter(temp_mol)
            writer.write(mol)
            writer.close()
            
            # Submit MOL file
            with open(temp_mol, 'rb') as f:
                files = {
                    'molpdbfile': (f"{name}.mol", f, 'chemical/x-mdl-molfile')
                }
                data = {
                    'smiData': '',
                    'checkopt': f" {optimization_iterations} ",
                    'chargetype': charge_model,
                    'dropcharge': f" {int(charge)} "
                }
                
                response = session.post(
                    'https://zarbi.chem.yale.edu/cgi-bin/results_lpg.py', 
                    files=files,
                    data=data,
                    verify=verify_ssl,
                    timeout=60
                )
            
            # Clean up temporary MOL file
            os.remove(temp_mol)
            
            if debug:
                save_debug_info(response, output_dir, prefix="mol_approach")
                
            soup = BeautifulSoup(response.text, 'html.parser')
            download_forms = soup.find_all('form')
            
            # If MOL approach failed, try PDB approach
            if not download_forms:
                print("MOL approach failed, trying PDB approach...")
                
                # Save temporary PDB file
                temp_pdb = os.path.join(output_dir, f"{name}_temp.pdb")
                writer = Chem.PDBWriter(temp_pdb)
                writer.write(mol)
                writer.close()
                
                # Submit PDB file
                with open(temp_pdb, 'rb') as f:
                    files = {
                        'molpdbfile': (f"{name}.pdb", f, 'chemical/x-pdb')
                    }
                    data = {
                        'smiData': '',
                        'checkopt': f" {optimization_iterations} ",
                        'chargetype': charge_model,
                        'dropcharge': f" {int(charge)} "
                    }
                    
                    response = session.post(
                        'https://zarbi.chem.yale.edu/cgi-bin/results_lpg.py',
                        files=files,
                        data=data,
                        verify=verify_ssl,
                        timeout=60
                    )
                
                # Clean up temporary PDB
                os.remove(temp_pdb)
                
                if debug:
                    save_debug_info(response, output_dir, prefix="pdb_approach")
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                download_forms = soup.find_all('form')
                
                # If PDB approach failed, try UFF optimization
                if not download_forms:
                    print("PDB approach failed, trying UFF optimization...")
                    
                    # Perform UFF optimization
                    AllChem.UFFOptimizeMolecule(mol)
                    
                    # Try PDB submission again with optimized structure
                    temp_pdb = os.path.join(output_dir, f"{name}_temp_uff.pdb")
                    writer = Chem.PDBWriter(temp_pdb)
                    writer.write(mol)
                    writer.close()
                    
                    with open(temp_pdb, 'rb') as f:
                        files = {
                            'molpdbfile': (f"{name}.pdb", f, 'chemical/x-pdb')
                        }
                        data = {
                            'smiData': '',
                            'checkopt': f" {optimization_iterations} ",
                            'chargetype': charge_model,
                            'dropcharge': f" {int(charge)} "
                        }
                        
                        response = session.post(
                            'https://zarbi.chem.yale.edu/cgi-bin/results_lpg.py',
                            files=files,
                            data=data,
                            verify=verify_ssl,
                            timeout=60
                        )
                    
                    # Clean up temporary PDB
                    os.remove(temp_pdb)
                    
                    if debug:
                        save_debug_info(response, output_dir, prefix="uff_approach")
                        
                    soup = BeautifulSoup(response.text, 'html.parser')
                    download_forms = soup.find_all('form')
                    
                    if not download_forms:
                        raise Exception("All approaches (SMILES, MOL, PDB, UFF) failed")
        
        downloaded_files = {}
        format_patterns = {
            'xml': (r'\.xml$', 'OpenMM'),
            'pdb': (r'\.pdb$', 'PDB'),
            'lt': (r'\.lt$', 'LT')
        }
        
        print(f"Found {len(download_forms)} download forms.")
        
        for form in download_forms:
            try:
                format_name = None
                for input_tag in form.find_all(['input', 'button']):
                    value = input_tag.get('value', '').lower()
                    for ext, (pattern, fmt) in format_patterns.items():
                        if re.search(pattern, value):
                            format_name = fmt
                            break
                    if format_name:
                        break
                
                if not format_name:
                    continue
                
                form_data = {input_tag['name']: input_tag.get('value', '') 
                            for input_tag in form.find_all('input')
                            if 'name' in input_tag.attrs}
                
                form_action = form.get('action', '')
                if not form_action:
                    continue
                    
                if not form_action.startswith('http'):
                    base_url = "https://zarbi.chem.yale.edu"
                if not form_action.startswith('/'):
                    form_action = f'/cgi-bin/{form_action}'
                form_action = base_url + form_action
                        
                download_response = session.post(form_action, data=form_data, verify=verify_ssl)
                        
                if download_response.status_code == 404:
                    print(f"Warning: Download failed for {format_name} - URL not found")
                    continue
                        
                filename_info = get_filename_from_response(download_response, format_name, name)
                if filename_info:
                    if isinstance(filename_info, tuple):
                        filename, prefix = filename_info
                        file_path = os.path.join(output_dir, filename)
                        with open(file_path, 'wb') as f:
                            f.write(download_response.content)
                        if process_xml_file(file_path, prefix):
                            downloaded_files[filename] = file_path
                    else:
                        filename = filename_info
                        file_path = os.path.join(output_dir, filename)
                        with open(file_path, 'wb') as f:
                            f.write(download_response.content)
                        downloaded_files[filename] = file_path
                    
                    print(f"Saved and processed {filename}")
            
            except Exception as e:
                print(f"Warning: Failed to process download form: {e}")
                continue
        
        return downloaded_files
        
    except Exception as e:
        if debug:
            try:
                debug_files = save_debug_info(response, output_dir)
                print(f"Debug files saved to: {debug_files[0]}")
            except:
                pass
        raise Exception(f"Network error: {str(e)}")

def extract_error_message(html_text):
    """Extract error messages from HTML response text.

    Args:
        html_text (str): HTML content to search for error messages

    Returns:
        str: Extracted error message or "Unknown error occurred" if none found

    Note:
        Searches for common error keywords and extracts surrounding text
    """
    # This function would need to be customized based on how errors are displayed
    # For now, a simple implementation that gets text around error keywords
    error_keywords = ["error", "invalid", "exception", "failed"]
    lines = html_text.lower().split('\n')
    
    for line in lines:
        if any(keyword in line.lower() for keyword in error_keywords):
            # Clean up the line
            clean_line = re.sub(r'<[^>]+>', ' ', line).strip()
            if clean_line:
                return clean_line
    
    return "Unknown error occurred"

def identify_format(form, form_data):
    """Identify the force field format from a download form.

    Args:
        form (BeautifulSoup): BeautifulSoup object representing the form
        form_data (dict): Dictionary of form input data

    Returns:
        str: Identified format name (e.g., 'OPLS_AMBER', 'OpenMM') or 'unknown_format'

    Note:
        Supports various force field formats including AMBER, CHARMM, GROMACS, OpenMM, etc.
    """
    # Check form text
    form_text = form.get_text().lower()
    
    # Dictionary of format identifiers and their names
    format_identifiers = {
        'amber': 'OPLS_AMBER',
        'charmm': 'OPLS_CHARMM',
        'gmx': 'OPLS_GMX',
        'gromacs': 'OPLS_GMX',
        'openmm': 'OpenMM',
        'namd': 'NAMD',
        'desmond': 'DESMOND',
        'tinker': 'TINKER',
        'boss': 'BOSS',
        'mcpro': 'MCPRO',
        'lammps': 'LAMMPS',
        'cns': 'CNS',
        'xplor': 'X-PLOR',
        'pqr': 'PQR'
    }
    
    # Check form text and input values for format identifiers
    for identifier, format_name in format_identifiers.items():
        if identifier in form_text:
            return format_name
        
        for value in form_data.values():
            if isinstance(value, str) and identifier in value.lower():
                return format_name
    
    # If form has a button with a specific name, use that
    for input_tag in form.find_all('input', type='submit'):
        button_text = input_tag.get('value', '').lower()
        for identifier, format_name in format_identifiers.items():
            if identifier in button_text:
                return format_name
    
    # Default return a generic name
    return "unknown_format"

# Example usage
if __name__ == "__main__":
    examples = '''
Examples:
    # Process a single SMILES string
    python ligpargen.py --smiles "CC(=O)O" --name "acetic_acid" --charge 0
    
    # Process a single SMILES with custom charge model and optimization
    python ligpargen.py --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --name "caffeine" --charge 0 --charge-model cm1a-bcc --opt-iter 100
    
    # Process compounds from an Excel file
    python ligpargen.py --excel "compounds.xlsx"
    
    # Process compounds from Excel with custom output directory
    python ligpargen.py --excel "compounds.xlsx" --output-dir "./my_parameters"
    '''
    
    parser = argparse.ArgumentParser(
        description='Generate OPLS parameters using LigParGen web service. This script can process either a single SMILES string or multiple compounds from an Excel file.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--excel', help='Process compounds from Excel file', default=None)
    parser.add_argument('--smiles', help='Single SMILES string to process', default=None)
    parser.add_argument('--name', help='Name for the output files (required with --smiles)', default=None)
    parser.add_argument('--charge', help='Molecular charge (default: 0)', type=int, default=0)
    parser.add_argument('--charge-model', help='Charge model to use (default: cm1a)', default='cm1a')
    parser.add_argument('--opt-iter', help='Number of optimization iterations (default: 0)', type=int, default=0)
    parser.add_argument('--output-dir', help='Output directory (default: ./output)', default='./output')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.smiles:
        # Process single SMILES string
        if not args.name:
            print("Error: --name is required when processing a single SMILES string")
            sys.exit(1)
            
        print(f"Processing single SMILES string: {args.smiles}")
        print(f"Name: {args.name}")
        print(f"Charge: {args.charge}")
        
        try:
            # Validate SMILES
            is_valid, validation_message = validate_smiles(args.smiles)
            if not is_valid:
                raise Exception(f"SMILES validation failed: {validation_message}")
            
            # First try with original SMILES
            try:
                files = submit_to_ligpargen_smiles(
                    smiles=args.smiles,
                    name=args.name,
                    charge=args.charge,
                    charge_model=args.charge_model,
                    optimization_iterations=args.opt_iter,
                    verify_ssl=False,
                    debug=True,
                    output_dir=args.output_dir
                )
            except Exception as e:
                # If original SMILES fails, try with canonical SMILES
                print("Original SMILES failed, trying with canonical SMILES...")
                mol = Chem.MolFromSmiles(args.smiles)
                if mol is None:
                    raise Exception("Could not parse SMILES string")
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                
                files = submit_to_ligpargen_smiles(
                    smiles=canonical_smiles, 
                    name=args.name,
                    charge=args.charge,
                    charge_model=args.charge_model,
                    optimization_iterations=args.opt_iter,
                    verify_ssl=False,
                    debug=True,
                    output_dir=args.output_dir
                )
            
            print("\nSuccessfully processed SMILES string")
            print("Generated files:")
            for filename in files:
                print(f"- {filename}")
                
        except Exception as e:
            print(f"Error processing SMILES string: {str(e)}")
            sys.exit(1)
            
    elif args.excel:
        # Process Excel file
        excel_file = args.excel
        output_dir = args.output_dir
        
        try:
            print("Reading Excel file...")
            # Process each relevant sheet
            sheets_to_process = ['Cations', 'Anions', 'Solvent']
            all_results = []
            failed_species = []  # List to store failed species
            
            for sheet in sheets_to_process:
                print(f"\nProcessing sheet: {sheet}")
                df = pd.read_excel(excel_file, sheet_name=sheet)
                print(f"Found {len(df)} total compounds")
                
                # Filter for compounds with OPLS parameters
                opls_compounds = df[df['Found parameters'].str.contains('OPLS/CM1A', case=False, na=False)]
                print(f"Found {len(opls_compounds)} compounds with OPLS/CM1A parameters")
                
                for i, (idx, row) in enumerate(opls_compounds.iterrows(), 1):
                    print(f"\nProcessing {row['Formula']} from {sheet} ({i}/{len(opls_compounds)})...")
                    print(f"SMILES: {row['SMILES']}")
                    print(f"Charge: {row['Charge']}")
                    try:
                        # Validate SMILES before submission
                        is_valid, validation_message = validate_smiles(row['SMILES'])
                        if not is_valid:
                            raise Exception(f"SMILES validation failed: {validation_message}")
                        
                        print(f"Submitting to LigParGen with SMILES: {row['SMILES']}")
                        files = submit_to_ligpargen_smiles(
                            smiles=row['SMILES'],
                            name=row['Formula'], 
                            charge=row['Charge'],
                            charge_model="cm1a",
                            optimization_iterations=args.opt_iter,
                            verify_ssl=False,
                            debug=True,
                            output_dir=output_dir
                        )
                        
                        all_results.append({
                            'Sheet': sheet,
                            'Formula': row['Formula'],
                            'SMILES': row['SMILES'],
                            'Charge': row['Charge'],
                            'Status': 'Success',
                            'Files': list(files.keys())
                        })
                        print(f"Successfully processed {row['Formula']}")
                        
                    except Exception as e:
                        error_msg = str(e)
                        all_results.append({
                            'Sheet': sheet,
                            'Formula': row['Formula'],
                            'SMILES': row['SMILES'],
                            'Charge': row['Charge'],
                            'Status': f'Failed: {error_msg}',
                            'Files': []
                        })
                        # Add failed species to the list
                        failed_species.append({
                            'Sheet': sheet,
                            'Formula': row['Formula'],
                            'SMILES': row['SMILES'],
                            'Charge': row['Charge'],
                            'Error': error_msg
                        })
                        print(f"Failed to process {row['Formula']}: {error_msg}")
                    
                    # Add a small delay between submissions to avoid overwhelming the server
                    time.sleep(1)
            
            # Create a results DataFrame and save it
            results_df = pd.DataFrame(all_results)
            results_file = os.path.join(output_dir, 'ligpargen_results.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\nProcessing complete. Results saved to {results_file}")
            
            # Save failed species to a separate file
            if failed_species:
                failed_file = os.path.join(output_dir, 'failed_species.txt')
                with open(failed_file, 'w') as f:
                    f.write("Failed Species Report\n")
                    f.write("===================\n\n")
                    for species in failed_species:
                        f.write(f"Sheet: {species['Sheet']}\n")
                        f.write(f"Formula: {species['Formula']}\n")
                        f.write(f"SMILES: {species['SMILES']}\n")
            # Print summary statistics
            print("\nSummary by sheet:")
            for sheet in sheets_to_process:
                sheet_results = results_df[results_df['Sheet'] == sheet]
                successes = len(sheet_results[sheet_results['Status'] == 'Success'])
                total = len(sheet_results)
                print(f"{sheet}: {successes}/{total} successful")
            
        except Exception as e:
            print(f"Error processing Excel file: {str(e)}")
            sys.exit(1)