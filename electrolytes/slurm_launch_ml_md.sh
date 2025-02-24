#!/bin/bash

#SBATCH --job-name=ml_md_elytes
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=learnaccel,scavenge
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=672-675

idx=$((${SLURM_ARRAY_TASK_ID}+0))

export SCHRODINGER=/private/home/levineds/desmond/mybuild/
source /checkpoint/levineds/elytes/bin/activate
echo python run_desmond.py --job_idx $idx --output_path /checkpoint/levineds/ml_md/elytes --csv ml_elytes.csv --runtime 1
python run_desmond.py --job_idx $idx --output_path /checkpoint/levineds/ml_md/elytes --csv ml_elytes.csv --runtime 1
deactivate
