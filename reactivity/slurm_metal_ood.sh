#!/bin/bash

#SBATCH --job-name=tm_react_pipeline
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=learnaccel,scavenge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=0-1434

idx=$SLURM_ARRAY_TASK_ID

echo /private/home/levineds/miniconda3/envs/mace/bin/python OOD_metal_complex_pipeline.py --input_path /checkpoint/levineds/metal_react/ --output_path /checkpoint/levineds/metal_react/ood/frames/ --start_index $(($idx)) --end_index $(($idx+1))
/private/home/levineds/miniconda3/envs/mace/bin/python OOD_metal_complex_pipeline.py --input_path /checkpoint/levineds/metal_react/ --output_path /checkpoint/levineds/metal_react/ood/frames/ --start_index $(($idx)) --end_index $(($idx+1))
echo done
