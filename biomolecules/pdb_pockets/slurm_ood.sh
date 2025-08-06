#!/bin/bash

## job name
#SBATCH --job-name=pdb_pockets
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel,ocp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=0-1286

#list=(51627 69390 70091 77100 77344 78714 96404 107375 108288 109334 12479 13842 17369)
#val=${list[SLURM_ARRAY_TASK_ID]}

val=$SLURM_ARRAY_TASK_ID
/private/home/levineds/schrodinger2024-2/run make_ood_pdb_pockets.py --start_idx $val --end_idx $((val+1)) --output_path /checkpoint/levineds/ood_pdb_pockets/
