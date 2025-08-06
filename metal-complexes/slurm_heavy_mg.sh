#!/bin/bash

## job name
#SBATCH --job-name=cod_extract
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel,ocp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=500m
#SBATCH --time=72:00:00
#SBATCH --array=0-199

val=$SLURM_ARRAY_TASK_ID
source /checkpoint/levineds/elytes/bin/activate
python /private/home/levineds/om-data/metal-organics/heavy_mg_extract.py --output_path /checkpoint/levineds/crystallographic_open_database/heavy_mg_st2/ --cif_csv /checkpoint/levineds/crystallographic_open_database/cod_cifs_without_disorder.csv --cod_path /checkpoint/levineds/crystallographic_open_database/ --num_workers 40 --total_chunks 200 --chunk_idx $val


