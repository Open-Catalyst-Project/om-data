#!/bin/bash

## job name
#SBATCH --job-name=protonation
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00
#SBATCH --array=0-19999%1000

val=$SLURM_ARRAY_TASK_ID
#$SCHRODINGER/run /private/home/levineds/om-data/biomolecules/ani2x/make_tautomers.py --source /checkpoint/levineds/ani2x/xyz_paths_list.txt --output_path /checkpoint/levineds/tautomers/ --n_struct 100000 --n_chunks 1000 --chunk_idx $val
$SCHRODINGER/run /private/home/levineds/om-data/biomolecules/ani2x/make_tautomers.py --source /checkpoint/levineds/heavy_mg_organic/xyz/xyz_paths_list.txt --output_path /checkpoint/levineds/tautomers/heavy_mg/ --n_struct 100000 --n_chunks 20000 --chunk_idx $val
