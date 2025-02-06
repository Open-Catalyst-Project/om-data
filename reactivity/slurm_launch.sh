#!/bin/bash

#SBATCH --job-name=atom_mapping
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=0-999

idx=$((${SLURM_ARRAY_TASK_ID}+0))

echo $SCHRODINGER/run smirks_db_to_complex.py --output_path /checkpoint/levineds/pmechdb/ --n_batch 1000 --batch_idx $idx --db_name pmechdb
$SCHRODINGER/run smirks_db_to_complex.py --output_path /checkpoint/levineds/pmechdb/ --n_batch 1000 --batch_idx $idx --db_name pmechdb
echo done
