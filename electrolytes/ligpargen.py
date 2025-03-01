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

# Suppress SSL warnings since we're intentionally using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from rdkit import Chem
    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False
    warnings.warn(
        "RDKit not found. SMILES validation will be disabled. "
        "Install RDKit for better SMILES validation: pip install rdkit"
    )

def validate_smiles(smiles):
    """Validate SMILES string using RDKit if available"""
    if not HAVE_RDKIT:
        return True, "SMILES validation skipped (RDKit not available)"
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES string"
    
    # Additional checks for molecule complexity
    num_atoms = mol.GetNumAtoms()
    if num_atoms > 100:  # This is a reasonable limit for LigParGen
        return False, f"Molecule too large ({num_atoms} atoms). LigParGen works best with smaller molecules."
    
    # Check for unsupported elements - using a list of elements instead of a string
    supported_elements = {'C', 'H', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Si'}  # Common elements supported by LigParGen
    elements = set(atom.GetSymbol() for atom in mol.GetAtoms())
    unsupported = elements - supported_elements
    if unsupported:
        return False, f"Contains unsupported elements: {', '.join(unsupported)}"
    
    return True, "SMILES validation passed"

def create_session(retries=5, backoff_factor=2.0, verify_ssl=False):
    """Create a requests session with retry logic"""
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

def submit_to_ligpargen_smiles(smiles, name, charge=0, charge_model="cm1a", 
                              optimization_iterations=0, output_dir="./output", 
                              verify_ssl=False, debug=True):
    session = create_session(retries=5, backoff_factor=2.0, verify_ssl=verify_ssl)
    
    # Set headers to mimic a browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': 'https://zarbi.chem.yale.edu',
        'Referer': 'https://zarbi.chem.yale.edu/ligpargen/'
    })
    
    # First get the main page to establish session
    try:
        main_page = session.get(
            'https://zarbi.chem.yale.edu/ligpargen/',
            verify=verify_ssl,
            timeout=30  # Increased timeout for initial connection
        )
        main_page.raise_for_status()
        
        # New code
        files = {
            'molpdbfile': ('', '', 'application/octet-stream')  # Empty file field
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

        # Use longer timeout for the main submission
        response = session.post(
            'https://zarbi.chem.yale.edu/cgi-bin/results_lpg.py',
            files=files,  # This makes it use multipart/form-data
            data=data,
            verify=verify_ssl,
            timeout=60  # Increased timeout for submission
        )
        
        if debug:
            save_debug_info(response, output_dir)
            
        if response.status_code == 500:
            print("Server response headers:", dict(response.headers))
            print("Form data submitted:", data)
            raise Exception("Server returned 500 error. Debug info saved.")
            
        # Rest of the code...
    except Exception as e:
        if debug:
            try:
                debug_files = save_debug_info(response, output_dir)
                print(f"Debug files saved to: {debug_files[0]}")
            except:
                pass
        raise Exception(f"Network error: {str(e)}")
    
    # Parse the response
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Check for error messages on the page
    error_messages = [
        "Error processing SMILES string",
        "Invalid SMILES",
        "Exception occurred",
        "Error occurred"
    ]
    
    for error in error_messages:
        if error in response.text:
            error_details = extract_error_message(response.text)
            raise Exception(f"Server reported an error: {error_details}")
    
    # Check if molecule visualization is shown (success indicator)
    if "3Dmol.js" in response.text or "Jmol.js" in response.text or "molecule viewer" in response.text.lower():
        print("Molecule successfully processed. Visualization page reached.")
    elif "Computing of OPLS/CM1A" in response.text or "Computing" in response.text:
        print("Job is being processed...")
    else:
        print("Unexpected response page. Continuing with download attempt...")
    
    # Extract download links
    downloaded_files = {}
    
    # Look for download buttons/forms
    download_forms = soup.find_all('form')
    
    if download_forms:
        print(f"Found {len(download_forms)} forms on the page.")
        
        for form in download_forms:
            # Look for forms that appear to be download forms
            if 'download' in str(form).lower() or 'save' in str(form).lower():
                # Extract the form details
                form_inputs = form.find_all('input', type=['hidden', 'submit'])
                form_data = {input_tag['name']: input_tag.get('value', '') 
                            for input_tag in form_inputs if 'name' in input_tag.attrs}
                
                # Try to identify which format this form is for
                format_name = identify_format(form, form_data)
                
                if format_name:
                    print(f"Processing download form for format: {format_name}")
                    
                    # Get the form submission URL and fix it
                    form_action = form.get('action', '')
                    if not form_action:
                        continue
                        
                    # Fix the URL path
                    if not form_action.startswith('http'):
                        base_url = "https://zarbi.chem.yale.edu"
                        if not form_action.startswith('/'):
                            form_action = f'/cgi-bin/{form_action}'
                        form_action = base_url + form_action
                    
                    print(f"Downloading from: {form_action}")
                    
                    # Submit the form to download the file
                    download_response = session.post(form_action, data=form_data, verify=verify_ssl)
                    
                    if download_response.status_code == 404:
                        print(f"Warning: Download failed for {format_name} - URL not found")
                        continue
                    
                    # Determine filename from Content-Disposition header or make a reasonable guess
                    filename = get_filename_from_response(download_response, format_name, name)
                    
                    # Save the file
                    file_path = os.path.join(output_dir, filename)
                    with open(file_path, 'wb') as f:
                        f.write(download_response.content)
                    
                    downloaded_files[filename] = file_path
                    print(f"Saved {format_name} to {file_path}")
    
    # If we didn't find forms, look for direct download links as a fallback
    if not downloaded_files:
        print("Looking for direct download links...")
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if any(ext in href.lower() for ext in ['.pdb', '.mol', '.xyz', '.itp', '.gro', '.top', '.prm', '.cns', '.str', '.pqr', '.par']):
                file_name = os.path.basename(href)
                
                # Some servers might use relative URLs
                if href.startswith('/') or not href.startswith('http'):
                    base_url = "https://zarbi.chem.yale.edu"
                    download_url = base_url + href if href.startswith('/') else base_url + '/' + href
                else:
                    download_url = href
                
                print(f"Downloading {file_name}...")
                file_response = session.get(download_url, verify=verify_ssl, allow_redirects=True)
                
                # Save the file
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, 'wb') as f:
                    f.write(file_response.content)
                
                downloaded_files[file_name] = file_path
                print(f"Saved to {file_path}")
    
    if not downloaded_files:
        print("No files were found for download. The server response might have changed.")
        print("Saving the HTML response for debugging purposes...")
        
        # Save the HTML for debugging
        debug_file = os.path.join(output_dir, "debug_response.html")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Saved debug HTML to {debug_file}")
    
    return downloaded_files

def extract_error_message(html_text):
    """Extract a more detailed error message from the HTML if possible"""
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
    """Try to identify which format a download form corresponds to"""
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

def get_filename_from_response(response, format_name, name):
    """Get filename from Content-Disposition header or construct a reasonable one"""
    # Try to get filename from Content-Disposition header
    if 'Content-Disposition' in response.headers:
        content_disp = response.headers['Content-Disposition']
        filename_match = re.search(r'filename="?([^";]+)"?', content_disp)
        if filename_match:
            # Replace the default name with our custom name but keep the extension
            orig_filename = filename_match.group(1)
            extension = os.path.splitext(orig_filename)[1]
            return f"{name}{extension}"
    
    # If no Content-Disposition, construct name with format
    format_extensions = {
        'OPLS_AMBER': '.prmtop',
        'OPLS_CHARMM': '.prm',
        'OPLS_GMX': '.top',
        'OpenMM': '.xml',
        'NAMD': '.pdb',
        'DESMOND': '.cms',
        'TINKER': '.prm',
        'BOSS': '.z',
        'MCPRO': '.z',
        'LAMMPS': '.data',
        'CNS': '.top',
        'X-PLOR': '.top',
        'PQR': '.pqr'
    }
    
    extension = format_extensions.get(format_name, '.dat')
    return f"{name}_{format_name.lower()}{extension}"

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate OPLS parameters using LigParGen')
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
            
            # Submit to LigParGen
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
            
            print("\nSuccessfully processed SMILES string")
            print("Generated files:")
            for filename in files:
                print(f"- {filename}")
                
        except Exception as e:
            print(f"Error processing SMILES string: {str(e)}")
            sys.exit(1)
            
    elif args.excel:
        # Process Excel file (existing code)
        excel_file = args.excel
        output_dir = args.output_dir
        
        try:
            print("Reading Excel file...")
            # Process each relevant sheet
            sheets_to_process = ['Cations', 'Anions', 'Solvent']
            all_results = []
            
            for sheet in sheets_to_process:
                print(f"\nProcessing sheet: {sheet}")
                df = pd.read_excel(excel_file, sheet_name=sheet)
                print(f"Found {len(df)} total compounds")
                
                # Filter for compounds with OPLS parameters
                opls_compounds = df[df['Found parameters'].str.contains('OPLS/CM1A', case=False, na=False)]
                print(f"Found {len(opls_compounds)} compounds with OPLS/CM1A parameters")
                
                for idx, row in opls_compounds.iterrows():
                    print(f"\nProcessing {row['Formula']} from {sheet} ({idx+1}/{len(opls_compounds)})...")
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
                            charge=row['Charge'],  # Use charge from the sheet
                            charge_model="cm1a",
                            optimization_iterations=args.opt_iter,  # Use command line argument instead of hardcoded value
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
                        all_results.append({
                            'Sheet': sheet,
                            'Formula': row['Formula'],
                            'SMILES': row['SMILES'],
                            'Charge': row['Charge'],
                            'Status': f'Failed: {str(e)}',
                            'Files': []
                        })
                        print(f"Failed to process {row['Formula']}: {str(e)}")
                    
                    # Add a small delay between submissions to avoid overwhelming the server
                    time.sleep(1)
            
            # Create a results DataFrame and save it
            results_df = pd.DataFrame(all_results)
            results_file = os.path.join(output_dir, 'ligpargen_results.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\nProcessing complete. Results saved to {results_file}")
            
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