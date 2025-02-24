#!/bin/bash

## job name
#SBATCH --job-name=prot_core
#SBATCH --output=/checkpoint/levineds/prot_core/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/prot_core/logs/%A_%a.err

#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00
#SBATCH --array=0-4161

val=$SLURM_ARRAY_TASK_ID
/private/home/levineds/schrodinger2024-2/run protein_core_extraction.py --start_idx $((val*50)) --end_idx $((val*50+50)) --output_path /checkpoint/levineds/prot_core/ --n_core_res 20 --seed $val
