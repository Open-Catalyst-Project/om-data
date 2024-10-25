#!/bin/bash

#SBATCH --job-name=md_electrolytes
#SBATCH --output=/private/home/levineds/electrolytes/logs/md/%A_%a.out
#SBATCH --error=/private/home/levineds/electrolytes/logs/md/%A_%a.err

#SBATCH --partition=learnaccel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=1-802

idx=$((${SLURM_ARRAY_TASK_ID}+0))

export SCHRODINGER=/private/home/levineds/desmond/mybuild/
source /checkpoint/levineds/elytes/bin/activate
echo python run_desmond.py --job_idx $idx --output_path /checkpoint/levineds/elytes_09_29_2024/
python run_desmond.py --job_idx $idx --output_path /checkpoint/levineds/elytes_09_29_2024/
deactivate
