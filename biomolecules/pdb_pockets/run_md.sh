#!/bin/bash

#SBATCH --job-name=md_pdb_pockets
#SBATCH --output=/private/home/levineds/pdb_pockets/logs/md/%A_%a.out
#SBATCH --error=/private/home/levineds/pdb_pockets/logs/md/%A_%a.err

#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=0-22606

idx=$((${SLURM_ARRAY_TASK_ID}+1))
paths_list="/private/home/levineds/om-data/biomolecules/pdb_pockets/good_list.txt"
job=$(sed -n "${idx}p" $paths_list)
dirname=$(dirname "$job")
maename=$(basename "$job")
basename=${maename%.mae}

export SCHRODINGER=/private/home/levineds/desmond
mkdir -p ${dirname}/md/${basename}
cd ${dirname}/md/${basename}
$SCHRODINGER/utilities/multisim -JOBNAME $basename -HOST localhost -maxjob 1 -cpu 1 -m /private/home/levineds/om-data/biomolecules/pdb_pockets/desmond_md.msj -c /private/home/levineds/om-data/biomolecules/pdb_pockets/desmond_md.cfg -description 'Molecular Dynamics' $job -mode umbrella -o ${dirname}/md/${basename}/${basename}-out.cms -WAIT 
