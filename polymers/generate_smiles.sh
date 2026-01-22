#!/bin/bash
#SBATCH --job-name=conf_samples
#SBATCH --output=smiles.out
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=16 
#SBATCH --mem=32G  
#SBATCH --time=00:30:00  

source ~/.bashrc

mamba activate loqi-omer

python conformer_eval.py --generate_smiles_list true \
                    --smiles_bank "traditional_charges.csv" \
                    --smiles_dir "$PATH_TO_OM_DATA/omer-files/" \
                    --working_dir "$PATH_TO_WORKING_DIR" \
                    --n_samples 2000
