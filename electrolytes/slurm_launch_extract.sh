#!/bin/bash

#SBATCH --job-name=elec_extract
#SBATCH --output=/private/home/levineds/electrolytes/logs/extract/%A_%a.out
#SBATCH --error=/private/home/levineds/electrolytes/logs/extract/%A_%a.err

#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=1-3757

idx=$((${SLURM_ARRAY_TASK_ID}+0))

echo $SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/elytes_09_29_2024/'$idx --save_dir '/checkpoint/levineds/elytes_09_29_2024/results' --system_name $idx --seed $((${idx}+123)) --top_n 10 --max_shell_size 300 --radii 7
$SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/elytes_09_29_2024/'$idx --save_dir '/checkpoint/levineds/elytes_09_29_2024/results' --system_name $idx --seed $((${idx}+123)) --top_n 10 --max_shell_size 300 --radii 7
