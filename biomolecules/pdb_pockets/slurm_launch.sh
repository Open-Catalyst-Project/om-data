#!/bin/bash

## job name
#SBATCH --job-name=pdb_pockets
#SBATCH --output=/private/home/levineds/pdb_pockets/logs/%A_%a.out
#SBATCH --error=/private/home/levineds/pdb_pockets/logs/%A_%a.err

#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=0-595

#list=(51627 69390 70091 77100 77344 78714 96404 107375 108288 109334 12479 13842 17369)
#val=${list[SLURM_ARRAY_TASK_ID]}

val=$SLURM_ARRAY_TASK_ID
#/private/home/levineds/schrodinger2024-2/run biolip_extraction.py --start_idx $((val*50)) --end_idx $((val*50+50)) --output_path /large_experiments/opencatalyst/foundation_models/data/omol/pdb_pockets/
/private/home/levineds/schrodinger2024-2/run biolip_extraction.py --start_idx $val --end_idx $((val+1)) --output_path /large_experiments/opencatalyst/foundation_models/data/omol/pdb_pockets/
