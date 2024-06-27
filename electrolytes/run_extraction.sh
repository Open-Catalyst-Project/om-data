#!/bin/bash
#TODO: can we automatically extract the names of all the solute atoms from the PDB file so we don't have to re-run this command for each solute?

# Run these scripts from om-data/electrolytes
# python solvation_shell_extract.py --pdb_file_path 'testfiles/al_clo4_example.pdb' \
#                                   --save_dir 'results' \
#                                   --system_name 'Al_ClO4' \
#                                   --solute_atoms 'C' \
#                                   --min_coord 2 \
#                                   --max_coord 5 \
#                                   --top_n 20

# python solvation_shell_extract.py --input_dir 'testfiles/LiPF6' \
#                                   --save_dir 'results' \
#                                   --system_name 'LiPF6' \
#                                   --solute_atoms 'P00' 'F01' 'F02' 'F03' 'F04' 'F05' 'F06'  \
#                                   --min_coord 2 \
#                                   --max_coord 20 \
#                                   --top_n 20

python solvation_shell_extract.py --input_dir 'testfiles/Al_ClO4' \
                                  --save_dir 'results' \
                                  --system_name 'Al_ClO4' \
                                  --solute_atoms 'CL1' 'O00' 'O02' 'O03' 'O04'  \
                                  --min_coord 2 \
                                  --max_coord 20 \
                                  --top_n 20