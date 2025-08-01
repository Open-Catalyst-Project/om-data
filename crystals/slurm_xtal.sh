#!/bin/bash

#SBATCH --job-name=xtal
#SBATCH --output=/checkpoint/levineds/omc_extracts/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/omc_extracts/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=41
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=291,297,298,314,796,829,831,867,894,900,907

idx=$((${SLURM_ARRAY_TASK_ID}+0))

$SCHRODINGER/run extract_clusters.py --output_path /checkpoint/levineds/omc_extracts/ --n_chunks 1000 --chunk_idx $idx --n_workers 41
