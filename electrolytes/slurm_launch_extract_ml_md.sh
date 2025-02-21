#!/bin/bash

#SBATCH --job-name=elec_extract
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=1-4000

idx=$((${SLURM_ARRAY_TASK_ID}+0))

echo $SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/ml_md/elytes/'$idx --save_dir '/checkpoint/levineds/ml_md/elytes/clusters' --system_name $idx --seed $((${idx}+123)) --top_n 30 --max_shell_size 350 --radii 3 --last_frame_only
$SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/ml_md/elytes/'$idx --save_dir '/checkpoint/levineds/ml_md/elytes/clusters' --system_name $idx --seed $((${idx}+123)) --top_n 30 --max_shell_size 350 --radii 3 --last_frame_only
echo done
