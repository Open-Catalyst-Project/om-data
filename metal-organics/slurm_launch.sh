#!/bin/bash

## job name
#SBATCH --job-name=architector
#SBATCH --output=/private/home/levineds/architector_generation/logs/%A_%a.out
#SBATCH --error=/private/home/levineds/architector_generation/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel,ocp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=0-18199

export OMP_NUM_THREADS=1
/private/home/levineds/miniconda3/envs/gpsts/bin/python mprun.py /checkpoint/levineds/arch_H/hydride_18200.pkl --n_workers 1 --batch_size 12 --batch_idx $((SLURM_ARRAY_TASK_ID+0)) --outpath /checkpoint/levineds/arch_H/inputs/

