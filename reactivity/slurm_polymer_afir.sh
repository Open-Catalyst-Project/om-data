#!/bin/bash

#SBATCH --job-name=polymer_afir
#SBATCH --output=/checkpoint/levineds/logs/%A_%a.out
#SBATCH --error=/checkpoint/levineds/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=0,1,2,3,4,5,8,9,12,13,14,17,19,20,21,22,23,24,25,26,27,28,29,32,33,34,35,36,39,40,41,42,43,44,45,47,48,49,50,51,53,56,57,58,59,62,63,64,65,66,68,70,71,72,73,75,76,77,78,79,80,81,82,83,84,85,86,87,88,90,91,92,93,96,98,101,103,105,144,150,182,183,304,305,325,326,355,374,420,445,446,455,512,535,543,544,549,553,562,565,568,569,573,574,575,576,579,583,584,585,597,599,600,602,603,605,606,608,633,634,635,640,641,643,644,665,667,668,671,677,687,694,702,703,707,709,715,716,732,733,735,740,743,750,751,752,757,758,788,789,794,797,799,801,802,807,809,810,818,824,838,849,853,857,858,872,873,889,890,894,896,897,909,911,912,918,922,923,931,947,955,956,959,960,963,970,973,976,981,988,989

idx=$((${SLURM_ARRAY_TASK_ID}+0))

export PYTHONPATH=/private/home/levineds/om-data/
#/private/home/levineds/miniconda3/envs/fairchem/bin/python /private/home/levineds/om-data/reactivity/omer_reactivity_pipeline.py --all_chains_dir /checkpoint/levineds/OpenPolymers/OMers_PDBs_1_v2/pdb/ --csv_dir /private/home/levineds/om-data/omer-files/ --output_path /checkpoint/levineds/OpenPolymers/reactivity/batch1 --n_chunks 1000 --chunk_idx $idx
/private/home/levineds/miniconda3/envs/fairchem/bin/python /private/home/levineds/om-data/reactivity/omer_reactivity_pipeline.py --all_chains_dir /checkpoint/levineds/OpenPolymers/OpenPolymer_PDBs_2/extracted_maes/pdb/ --csv_dir /private/home/levineds/om-data/omer-files/ --output_path /checkpoint/levineds/OpenPolymers/reactivity/batch2 --n_chunks 1000 --chunk_idx $idx
