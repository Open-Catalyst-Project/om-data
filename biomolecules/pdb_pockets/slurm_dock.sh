#!/bin/bash

## job name
#SBATCH --job-name=pdb_pockets
#SBATCH --output=/private/home/levineds/pdb_fragments/logs/%A_%a.out
#SBATCH --error=/private/home/levineds/pdb_fragments/logs/%A_%a.err

#SBATCH --partition=scavenge,learnaccel,ocp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=2001-3000

#/private/home/levineds/schrodinger2024-2/run dock_ligand_into_pocket.py --pdb_path /large_experiments/opencatalyst/foundation_models/data/omol/pdb_pockets --pockets_pkl_path /private/home/levineds/om-data/biomolecules/pdb_pockets/stats.pkl --lig_source geom --lig_file_path /private/home/levineds/om-data/biomolecules/pdb_pockets/rdkit_folder/ --output_path /large_experiments/opencatalyst/foundation_models/data/omol/pdb_fragments --n_to_dock 3000 --random_seed $SLURM_ARRAY_TASK_ID
#/private/home/levineds/schrodinger2024-2/run dock_ligand_into_pocket.py --pdb_path /large_experiments/opencatalyst/foundation_models/data/omol/pdb_pockets --pockets_pkl_path /private/home/levineds/om-data/biomolecules/pdb_pockets/stats.pkl --lig_source chembl --lig_file_path /private/home/levineds/om-data/biomolecules/pdb_pockets/chembl.maegz --output_path /large_experiments/opencatalyst/foundation_models/data/omol/pdb_fragments --n_to_dock 3000 --random_seed $SLURM_ARRAY_TASK_ID
/private/home/levineds/schrodinger2024-2/run dock_ligand_into_pocket.py --pdb_path /large_experiments/opencatalyst/foundation_models/data/omol/pdb_pockets --pockets_pkl_path /private/home/levineds/om-data/biomolecules/pdb_pockets/stats.pkl --lig_source zinc_leads --lig_file_path /private/home/levineds/ZincLeads --output_path /large_experiments/opencatalyst/foundation_models/data/omol/pdb_fragments_batch2 --n_to_dock 3000 --random_seed $((SLURM_ARRAY_TASK_ID*1001))
