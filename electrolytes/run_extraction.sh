#!/bin/bash
#TODO: can we automatically extract the names of all the solute atoms from the PDB file so we don't have to re-run this command for each solute?

$SCHRODINGER/run python3 solvation_shell_extract.py --input_dir 'testfiles/1' \
                                  --save_dir 'results' \
                                  --system_name 'Li_BF4' \
                                  --max_frames 100
                        