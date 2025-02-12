#!/bin/bash

## job name
#SBATCH --job-name=prot_iface
#SBATCH --output=/checkpoint/levineds/prot_interface/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/prot_interface/logs/%A_%a.err

#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00
#SBATCH --array=0-1684

val=$SLURM_ARRAY_TASK_ID
echo /private/home/levineds/schrodinger2024-2/run protein_interface_extraction.py --start_idx $((val*25)) --end_idx $((val*25+25)) --output_path /checkpoint/levineds/prot_interface/ --n_iface_res 48 --seed $val
/private/home/levineds/schrodinger2024-2/run protein_interface_extraction.py --start_idx $((val*25)) --end_idx $((val*25+25)) --output_path /checkpoint/levineds/prot_interface/ --n_iface_res 48 --seed $val
