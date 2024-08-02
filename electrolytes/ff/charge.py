import glob
import re
import os

def extract_charges_from_lt(file_path):
    """
    Extract charges from the Data Atoms section of an LT file and calculate the total sum.

    Parameters:
    file_path (str): Path to the LT file.

    Returns:
    list: List of charges.
    float: Total sum of charges.
    """
    charges = []
    in_data_atoms_section = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('write("Data Atoms") {'):
                in_data_atoms_section = True
                continue
            if in_data_atoms_section:
                if line == '}':
                    in_data_atoms_section = False
                    continue
                if line:  # Avoid empty lines
                    parts = line.split()
                    if len(parts) >= 4:
                        charge = float(parts[3])  # Assuming the charge is the fourth entry
                        charges.append(charge)

    total_charge = sum(charges)
    return charges, total_charge

def extract_expected_charge_from_filename(filename):
    """
    Extract the expected charge from the LT file name.

    Parameters:
    filename (str): Name of the LT file.

    Returns:
    float: Expected charge extracted from the file name.
    """
    match = re.search(r'([+-]?)(\d*)\.lt$', filename)
    if match:
        sign = match.group(1)
        number = match.group(2)
        
        if sign and not number:  # Case like "H4N+.lt" or "C4H5O-.lt" without a number after the sign
            return 1.0 if sign == '+' else -1.0
        elif sign and number:  # Case like "C4H5O+2.lt" or "C4H5O-2.lt"
            return float(sign + number)
        else:  # Case like "CH7O.lt" without any charge indication
            return 0.0
    return 0.0  # Default to neutral charge if no match

def adjust_charges(charges, total_charge, expected_charge):
    """
    Adjust charges to match the expected charge.

    Parameters:
    charges (list): List of current charges.
    total_charge (float): Current total charge.
    expected_charge (float): Expected total charge.

    Returns:
    list: Adjusted charges.
    """
    difference = expected_charge - total_charge
    if len(charges) > 0:
        adjustment = difference / len(charges)
        adjusted_charges = [charge + adjustment for charge in charges]
        return adjusted_charges
    return charges

def process_all_lt_files(directory_path='.'):
    """
    Process all LT files in the specified directory, extracting and summing charges from each file.

    Parameters:
    directory_path (str): Path to the directory containing LT files. Default is the current directory.

    Returns:
    dict: A dictionary with file names as keys and tuples (charges, total_charge, expected_charge, is_correct, adjusted_charges) as values.
    """
    lt_files = glob.glob(os.path.join(directory_path, "*.lt"))
    results = {}

    for lt_file in lt_files:
        charges, total_charge = extract_charges_from_lt(lt_file)
        expected_charge = extract_expected_charge_from_filename(os.path.basename(lt_file))
        is_correct = (total_charge == expected_charge)
        adjusted_charges = charges
        if not is_correct:
            adjusted_charges = adjust_charges(charges, total_charge, expected_charge)
            write_adjusted_charges_to_lt(lt_file, adjusted_charges)
        results[lt_file] = (charges, total_charge, expected_charge, is_correct, adjusted_charges)

    return results

def write_adjusted_charges_to_lt(file_path, adjusted_charges):
    """
    Write adjusted charges back to the LT file.

    Parameters:
    file_path (str): Path to the LT file.
    adjusted_charges (list): List of adjusted charges.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    in_data_atoms_section = False
    charge_index = 0

    with open(file_path, 'w') as file:
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('write("Data Atoms") {'):
                in_data_atoms_section = True
                file.write(line)
                continue
            if in_data_atoms_section:
                if stripped_line == '}':
                    in_data_atoms_section = False
                    file.write(line)
                    continue
                if stripped_line and charge_index < len(adjusted_charges):  # Avoid empty lines
                    parts = stripped_line.split()
                    if len(parts) >= 4:
                        parts[3] = f"{adjusted_charges[charge_index]:.15f}"  # Adjust the charge value
                        charge_index += 1
                    file.write(" ".join(parts) + "\n")
                else:
                    file.write(line)
            else:
                file.write(line)

# Example usage
directory_path = '.'  # Set to the current directory
results = process_all_lt_files(directory_path)

for file, (charges, total_charge, expected_charge, is_correct, adjusted_charges) in results.items():
    print(f"File: {file}")
    print("Charges:", charges)
    print("Total Charge:", total_charge)
    print("Expected Charge:", expected_charge)
    print("Is Correct:", is_correct)
    print("Adjusted Charges:", adjusted_charges)
    print()
