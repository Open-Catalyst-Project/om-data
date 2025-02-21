#!/bin/bash

## job name
#SBATCH --job-name=solv_elec
#SBATCH --output=/private/home/levineds/solvate_electrolytes/logs/%A_%a.out
#SBATCH --error=/private/home/levineds/solvate_electrolytes/logs/%A_%a.err

#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=72:00:00

#SBATCH --array=0-71

list=(122-triazole benzothiadizaole cyclic_carbonate glyme hexahydrotantalate lactone naphthoquinone phosphine ptio quinoxaline sultone tetrazine acetate borane cyclic_sulfate guanidinium hydrogen_sulfite linear_carbonate organophosphate phosphonium pyridazinium silane tempo tetrazolium ammonium carbamate cyclic_sulfite_ester hexahydroantimonate imidazolidine maleic_anhydride organosulfate phosphorane pyridine_ester siloxane tetrahydroaluminate thf anthraquinone carboxylate cyclopropenium hexahydroarsenate imidazolium methanoate oxadiazine phthalimide pyridinium sulfamoyl_fluoride tetrahydroborate thiazolium azanide cyclic_aluminate dinitrile hexahydroniobate iosquinolinium methoxyalkylamine oxazolidinium piperazinium pyrrolidinium sulfonylimide tetrahydrogalldate triazolium benzoquinone cyclic_borate glycinate hexahydrophosphate lactam morpholinium phenothiazine piperidinium pyrroline sulphonium tetrahydroindate viologen)

val=${list[SLURM_ARRAY_TASK_ID]}
/private/home/levineds/miniconda3/envs/gpsts/bin/python solvate.py --xyz_dir /large_experiments/opencatalyst/foundation_models/data/omol/electrolytes/functionalization/xyzs/$val --max_core_molecule_size 90 --max_atom_budget 180 --base_dir /large_experiments/opencatalyst/foundation_models/data/omol/electrolytes/functionalization/solvated/$val --structure_idx $SLURM_ARRAY_TASK_ID
