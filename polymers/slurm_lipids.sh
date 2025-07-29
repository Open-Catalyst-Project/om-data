#!/bin/bash

#SBATCH --job-name=lipids
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=41
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=0-999

idx=$((${SLURM_ARRAY_TASK_ID}+0))

$SCHRODINGER/run lipids.py --input_dir /checkpoint/levineds/OpenPolymers/lipids --output_path /checkpoint/levineds/OpenPolymers/lipids/clusters --n_chunks 1000 --chunk_idx $idx --n_workers 40
