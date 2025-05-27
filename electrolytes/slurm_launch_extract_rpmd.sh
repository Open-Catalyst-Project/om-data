#!/bin/bash

#SBATCH --job-name=elec_extract
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=0-49

idx=$((${SLURM_ARRAY_TASK_ID}+0))

echo $SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/rpmd/'$idx --save_dir '/checkpoint/levineds/rpmd/results' --system_name $idx --seed $((${idx}+913)) --top_n 200 --max_shell_size 130 --radii 3
$SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/rpmd/'$idx --save_dir '/checkpoint/levineds/rpmd/results' --system_name $idx --seed $((${idx}+913)) --top_n 200 --max_shell_size 130 --radii 3
echo done
