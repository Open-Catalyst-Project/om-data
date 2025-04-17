#!/bin/bash

#SBATCH --job-name=elec_extract
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=1-52

idx=$((${SLURM_ARRAY_TASK_ID}+0))

#echo $SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/ood_elytes/all/'$idx --save_dir '/checkpoint/levineds/ood_elytes/all/results' --system_name $idx --seed $((${idx}+93)) --top_n 20 --max_shell_size 250 --radii 3 --rmsd_sampling
#$SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/ood_elytes/all/'$idx --save_dir '/checkpoint/levineds/ood_elytes/all/results' --system_name $idx --seed $((${idx}+93)) --top_n 20 --max_shell_size 250 --radii 3 --rmsd_sampling
#$SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/ood_elytes/anion/'$idx --save_dir '/checkpoint/levineds/ood_elytes/anion/results' --system_name $idx --seed $((${idx}+4)) --top_n 20 --max_shell_size 250 --radii 3 --rmsd_sampling
#$SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/ood_elytes/cation/'$idx --save_dir '/checkpoint/levineds/ood_elytes/cation/results' --system_name $idx --seed $((${idx}+9)) --top_n 20 --max_shell_size 250 --radii 3 --rmsd_sampling
$SCHRODINGER/run solvation_shell_extract.py --input_dir '/checkpoint/levineds/ood_elytes/solvent/'$idx --save_dir '/checkpoint/levineds/ood_elytes/solvent/results' --system_name $idx --seed $((${idx}+3)) --top_n 20 --max_shell_size 250 --radii 3 --rmsd_sampling
echo done
