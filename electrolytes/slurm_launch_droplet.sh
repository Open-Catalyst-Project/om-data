#!/bin/bash

#SBATCH --job-name=droplet_elytes
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=learnaccel,scavenge,ocp
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=0-499

idx=$((${SLURM_ARRAY_TASK_ID}+0))

echo /private/home/levineds/miniconda3/envs/rpmd/bin/python driver_omm.py --row $idx --csv_file omm-elytes.csv --droplet --stepsize 0.002
/private/home/levineds/miniconda3/envs/rpmd/bin/python driver_omm.py --row $idx --csv_file omm-elytes.csv --droplet --stepsize 0.002
