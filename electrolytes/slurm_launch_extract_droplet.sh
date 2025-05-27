#!/bin/bash

#SBATCH --job-name=elec_extract
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=15,402,403,467,468,423

idx=$((${SLURM_ARRAY_TASK_ID}+0))

echo $SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/droplet/'$idx --save_dir '/checkpoint/levineds/droplet/results' --system_name $idx --seed $((${idx}+421)) --top_n 20 --max_shell_size 130 --radii 3 --no_pbc
$SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/droplet/'$idx --save_dir '/checkpoint/levineds/droplet/results' --system_name $idx --seed $((${idx}+421)) --top_n 20 --max_shell_size 130 --radii 3 --no_pbc
echo done
