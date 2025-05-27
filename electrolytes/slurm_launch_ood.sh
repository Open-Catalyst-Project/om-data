#!/bin/bash

#SBATCH --job-name=ood_elytes
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=learnaccel,scavenge,ocp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=1-52

idx=$((${SLURM_ARRAY_TASK_ID}+0))

export SCHRODINGER=/private/home/levineds/desmond/mybuild/
source /checkpoint/levineds/elytes/bin/activate
#echo python run_desmond.py --job_idx $idx --output_path /checkpoint/levineds/ood_elytes/all --csv ood-all-elytes.csv --runtime 250
#python run_desmond.py --job_idx $idx --output_path /checkpoint/levineds/ood_elytes/all --csv ood-all-elytes.csv --runtime 250
#echo python run_desmond.py --job_idx $idx --output_path /checkpoint/levineds/ood_elytes/anion --csv ood-an-elytes.csv --runtime 250
#python run_desmond.py --job_idx $idx --output_path /checkpoint/levineds/ood_elytes/anion --csv ood-an-elytes.csv --runtime 250
#echo python run_desmond.py --job_idx $idx --output_path /checkpoint/levineds/ood_elytes/cation --csv ood-cat-elytes.csv --runtime 250
#python run_desmond.py --job_idx $idx --output_path /checkpoint/levineds/ood_elytes/cation --csv ood-cat-elytes.csv --runtime 250
echo python run_desmond.py --job_idx $idx --output_path /checkpoint/levineds/ood_elytes/solvent --csv ood-sol-elytes.csv --runtime 250
python run_desmond.py --job_idx $idx --output_path /checkpoint/levineds/ood_elytes/solvent --csv ood-sol-elytes.csv --runtime 250
deactivate
