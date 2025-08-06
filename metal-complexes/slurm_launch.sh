#!/bin/bash

## job name
#SBATCH --job-name=architector
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=41
#SBATCH --mem=12g
#SBATCH --time=72:00:00

#SBATCH --array=0-157

export OMP_NUM_THREADS=1
#/private/home/levineds/miniconda3/envs/gpsts/bin/python mprun.py /checkpoint/levineds/ml_md/metal_organics/metal_organics_MS_1M.pkl --n_workers 40 --batch_size 250 --batch_idx $((SLURM_ARRAY_TASK_ID+0)) --outpath /checkpoint/levineds/ml_md/metal_organics/inputs/
#/private/home/levineds/miniconda3/envs/gpsts/bin/python mprun.py /checkpoint/levineds/interm_metal_complexes/interm_metal_complexes_200k.pkl --n_workers 40 --batch_size 250 --batch_idx $((SLURM_ARRAY_TASK_ID+0)) --outpath /checkpoint/levineds/interm_metal_complexes/inputs
#/private/home/levineds/miniconda3/envs/gpsts/bin/python mprun.py /checkpoint/levineds/interm_metal_complexes/interm_metal_complexes_300k.pkl --n_workers 40 --batch_size 250 --batch_idx $((SLURM_ARRAY_TASK_ID+0)) --outpath /checkpoint/levineds/interm_metal_complexes/inputs2
#/private/home/levineds/miniconda3/envs/gpsts/bin/python mprun.py /checkpoint/levineds/addl_metal_complexes/new_ox_60k.pkl --n_workers 40 --batch_size 50 --batch_idx $((SLURM_ARRAY_TASK_ID+0)) --outpath /checkpoint/levineds/addl_metal_complexes/inputs
#/private/home/levineds/miniconda3/envs/gpsts/bin/python mprun.py /checkpoint/levineds/heavy_mg_complexes/heavy_mg_50k.pkl --n_workers 40 --batch_size 50 --batch_idx $((SLURM_ARRAY_TASK_ID+0)) --outpath /checkpoint/levineds/heavy_mg_complexes/inputs_tiny
#/private/home/levineds/miniconda3/envs/gpsts/bin/python mprun.py /checkpoint/levineds/heavy_mg_complexes/heavy_mg_200k.pkl --n_workers 40 --batch_size 40 --batch_idx $((SLURM_ARRAY_TASK_ID+0)) --outpath /checkpoint/levineds/heavy_mg_complexes/inputs_nohmglig
/private/home/levineds/miniconda3/envs/gpsts/bin/python mprun.py /checkpoint/levineds/heavy_mg_complexes/heavy_mg_100k.pkl --n_workers 40 --batch_size 40 --batch_idx $((SLURM_ARRAY_TASK_ID+0)) --outpath /checkpoint/levineds/heavy_mg_complexes/inputs_withhmglig
