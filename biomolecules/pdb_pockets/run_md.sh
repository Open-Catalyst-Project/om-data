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

#SBATCH --array=0-30

jobs_per_array=24
temp="300K"
paths_list="/private/home/levineds/om-data/biomolecules/pdb_pockets/${temp}_paths_list_restart.txt"
export SCHRODINGER=/private/home/levineds/desmond/mybuild
$SCHRODINGER/jsc local-server-start
$SCHRODINGER/jsc download-start

for ((idx=$((${SLURM_ARRAY_TASK_ID}*${jobs_per_array}+1));idx<=$((${SLURM_ARRAY_TASK_ID}*${jobs_per_array}+${jobs_per_array}));idx++)); do
    echo $idx
    job=$(sed -n "${idx}p" $paths_list)
    dirname=$(dirname "$job")
    maename=$(basename "$job")
    basename=${maename%.maegz}

    mkdir -p ${dirname}/md/${temp}/${basename}
    cd ${dirname}/md/${temp}/${basename}
    if [ -f ${basename}-out.cms ]; then
        echo ${basename}-out.cms already present
    else
        echo $SCHRODINGER/utilities/multisim -JOBNAME $basename -HOST localhost -maxjob 1 -cpu 1 -m /private/home/levineds/om-data/biomolecules/pdb_pockets/desmond_md_${temp}.msj -c /private/home/levineds/om-data/biomolecules/pdb_pockets/desmond_md_${temp}.cfg -description 'Molecular Dynamics' $job -mode umbrella -o ${dirname}/md/${temp}/${basename}/${basename}-out.cms -WAIT 
        $SCHRODINGER/utilities/multisim -JOBNAME $basename -HOST localhost -maxjob 1 -cpu 1 -m /private/home/levineds/om-data/biomolecules/pdb_pockets/desmond_md_${temp}.msj -c /private/home/levineds/om-data/biomolecules/pdb_pockets/desmond_md_${temp}.cfg -description 'Molecular Dynamics' $job -mode umbrella -o ${dirname}/md/${temp}/${basename}/${basename}-out.cms -WAIT
    fi 
done
