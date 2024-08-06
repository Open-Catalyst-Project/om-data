import os
import re

# Function to modify *.lt files
def modify_lt_files(directory):
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".lt"):
            filepath = os.path.join(directory, filename)
            # Read the content of the file
            with open(filepath, 'r') as file:
                lines = file.readlines()

            modified_lines = []
            # Check if the file contains a line starting with "dihedral_coeff"
            for line in lines:
                try:
                    if line.split()[0].startswith("dihedral_coeff"):
                        print(line)
                        if 'opls' in line or 'fourier' in line:
                            modified_lines.append(line)
                            continue
                        else:# Modify the line to append 'opls' after '@dihedral:X'
                            modified_line = re.sub(r'(@dihedral:\w+)', r'\1 opls', line)
                            modified_lines.append(modified_line)
                    else:
                        modified_lines.append(line)
                except:
                    continue
            # Write the modified content back to the file
            with open(filepath, 'w') as file:
                file.writelines(modified_lines)

# Provide the directory containing the *.lt files
directory = '.'  # Change this to the directory containing your *.lt files
modify_lt_files(directory)
