#!/bin/bash

## job name
#SBATCH --job-name=rna_extract
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00
#SBATCH --array=0-1106

val=$SLURM_ARRAY_TASK_ID
#/private/home/levineds/schrodinger2024-2/run /private/home/levineds/om-data/biomolecules/pdb_pockets/na_extraction.py --start_idx $((val*5)) --end_idx $((val*5+5)) --output_path /checkpoint/levineds/rna/new/ --seed $val --na_type rna
source /checkpoint/levineds/elytes/bin/activate
python /private/home/levineds/om-data/biomolecules/pdb_pockets/na_extraction.py --start_idx $((val*5)) --end_idx $((val*5+5)) --output_path /checkpoint/levineds/rna/addl/ --seed $((val+1)) --na_type rna
