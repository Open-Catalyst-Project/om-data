#!/bin/bash

#SBATCH --job-name=rpmd_elytes
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=learnaccel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=1,2,5,9,10,14,16,20,21,24,45,49

idx=$((${SLURM_ARRAY_TASK_ID}+0))

echo /private/home/levineds/miniconda3/envs/rpmd/bin/python driver_omm.py --row $idx --csv_file rpmd_elytes.csv --rpmd --stepsize 0.0005 --n_beads 32
/private/home/levineds/miniconda3/envs/rpmd/bin/python driver_omm.py --row $idx --csv_file rpmd_elytes.csv --rpmd --stepsize 0.0005 --n_beads 32

# for non-H2O systems
#/private/home/levineds/miniconda3/envs/rpmd/bin/python driver_omm.py --row $idx --csv_file rpmd_elytes.csv --rpmd --stepsize 0.001 --n_beads 32
