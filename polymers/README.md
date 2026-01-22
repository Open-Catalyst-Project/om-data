# Lipids 


# Conformers 
Includes revised instructions from LoQI repo

## Step 1: Clone the repo
```bash
git clone https://github.com/isayevlab/LoQI.git
```

## Step 2: Make environment
```bash
mamba env create -f loqi-omer_env.yml
mamba activate loqi-omer

pip install git+https://github.com/lanl/Architector.git@Secondary_Solvation_Shell

cd LoQI
pip install -e .

cd $PATH_TO_FAIRCHEM
pip install -e .

pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

module load gcc/11.2.0 # needed for perlmutter 
conda install -c conda-forge xtb-python xtb --freeze-installed
```

## Step 3: Download LoQI files
Download both from [this link](https://drive.google.com/drive/folders/1PvSrep7_qIjTSslzXD3KUYEJ2Qr2lgDD)

```bash
cd LoQI
mkdir LoQI_data
```

Place `loqi.ckpt` and `chembl3d_stereo` in `LoQI_data/`. In `LoQI/scripts/conf/loqi/loqi.yaml`, set the right path:
```python
# .....
data:
  dataset_root: "$PATH_TO_LOQI/LoQI_data/chembl3d_stereo"
# .....

```

## Step 4: Run the program
Edit paths in `generate_smiles.sh` and `conformers.sh` accordingly. Make an output directory, with subdirectories `xyzs/` and `sdfs/`

### First step: Generate SMILES for branched polymers
``` bash
sbatch generate_smiles.sh
```
`all_smiles.csv` contains all the generated SMILES that are branched conformers with at least 3 branching points. `info_all_smiles.csv` contains information regarding the inchikey corresponding to the SMILES as well as the monomer repeat units that make up the sample, whether it is a copolymer, and whether it is charged. Default settings generate 1500 samples, where roughly 50% are copolymers with up to 5 repeat units and 25% have 4 or less charged atoms. 

This distribution is dependent on using `traditional_charges.csv`, which has the first 100 unchraged repeat units of `Traditional_polymers.csv`, as well as all 9 charged units from the list and an additional 53 which have not been seen in the initial dataset. 

### Second step: Run LoQI to generate conformers then keep 10 of the best
``` bash
sbatch conformers.sh
```
In the working directory's `xyzs/` folder, the .xyz files will be generated. Summary information on whether a given SMILES succeeded in conformer generation is recorded in `info_all_confs_{chunk_idx}.csv`, along with the number of conformers saved.





