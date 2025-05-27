#!/bin/bash

## job name
#SBATCH --job-name=dna_extract
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel,ocp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00
#SBATCH --array=0-1506

val=$SLURM_ARRAY_TASK_ID
#/private/home/levineds/schrodinger2024-2/run /private/home/levineds/om-data/biomolecules/pdb_pockets/na_extraction.py --start_idx $((val*5)) --end_idx $((val*5+5)) --output_path /checkpoint/levineds/dna/addl --seed $val
source /checkpoint/levineds/elytes/bin/activate
python /private/home/levineds/om-data/biomolecules/pdb_pockets/na_extraction.py --start_idx $((val*5)) --end_idx $((val*5+5)) --output_path /checkpoint/levineds/dna/addl --seed $((val+1))
