#!/bin/bash
#SBATCH --job-name=eval_conf
#SBATCH --output=generate_%a.out
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32 
#SBATCH --mem=32G  
#SBATCH --time=05:00:00
#SBATCH --array=0-3

idx=$((${SLURM_ARRAY_TASK_ID}+0))

source ~/.bashrc

mamba activate loqi-omer
export PYTHONPATH="$PATH_TO_LOQI:$PYTHONPATH"

python ./conformer_eval.py --make_conformers true \
                         --working_dir "$PATH_TO_WORKING_DIR" \
                         --loqi_path "$PATH_TO_LOQI" \
                         --n_chunks 4 \
                         --chunk_idx $idx