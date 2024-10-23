#!/bin/bash

## job name
#SBATCH --job-name=architector
#SBATCH --output=/private/home/levineds/architector_generation/logs/%A_%a.out
#SBATCH --error=/private/home/levineds/architector_generation/logs/%A_%a.err

#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=41
#SBATCH --mem=12g
#SBATCH --time=72:00:00

#SBATCH --array=0-509

export OMP_NUM_THREADS=1
/private/home/levineds/miniconda3/envs/gpsts/bin/python mprun.py /checkpoint/levineds/arch_Ln/MO_Ln_255k.pkl --n_workers 40 --batch_size 500 --batch_idx $((SLURM_ARRAY_TASK_ID+0)) --outpath /checkpoint/levineds/arch_Ln/inputs/
