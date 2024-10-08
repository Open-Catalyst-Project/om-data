# Electrolytes MD Workflow

This README provides a rough guide on how to prepare and run bulk electrolyte simulations. The workflow requires the following to be installed:
1. NumPy
2. Packmol: [https://m3g.github.io/packmol/](https://m3g.github.io/packmol/)
3. Moltemplate: [https://www.moltemplate.org/](https://www.moltemplate.org/)
4. OpenMM: [https://openmm.org/](https://openmm.org/)
5. MDAnalysis: [https://www.mdanalysis.org/](https://www.mdanalysis.org/)
6. RDKit: [https://www.rdkit.org/](https://www.rdkit.org/)
7. PDBFixer: [https://github.com/openmm/pdbfixer](https://github.com/openmm/pdbfixer)
8. (Optional) LAMMPS: [https://www.lammps.org/#gsc.tab=0](https://www.lammps.org/#gsc.tab=0)

## List of files and directories
Only the important ones
- README.md: this file
- `prepsim_omm.sh`: a Bash script that will prepare the initial system configurations in the elytes.csv files for OpenMM simulations
- `prepsim_desmond.sh`: a Bash script that will prepare the initial system configurations in the elytes.csv files for Desmond simulations
- `runsimulations.sh`: a Bash script that will run the simulations one by one. 
- ff: a directory of force field files of all electroyte components. 
- elytes.csv: a CSV file listing all possible electrolyte systems we can simulate.
- litelytes.csv: a CSV file listing electrolyte systems curated from literature. 
- `data2lammps.py`: a Python module to generate LAMMPS DATA and run files. 
- `lammps2omm.py`: a Python module to convert LAMMPS DATA and run files to OpenMM XML and PDB files. 
- `generatesolvent_omm.py`: a Python script to generate solvent configuration and force field files (in OpenMM).
- `generatesystem_omm.py`: a Python script to generate system configuration (salt+solvent) and force field files (in OpenMM).
- `generatesolvent_desmond.py`: a Python script to generate solvent configuration and force field files (in Desmond).
- `generatesystem_desmond.py`: a Python script to generate system configuration (salt+solvent) and force field files (in Desmond).
- `randommixing.py`: a Python script to generate completely random electrolytes and append them to elytes.csv file. 
- `classmixing.py`: a Python script to generate random electrolytes based on their classifications and append them to elytes.csv file. 

## How to run workflow

### Desmond

There is a Bash script that runs the workflow `prepsim_desmond.sh` as follows 
```
./prepsim_desmond.sh
```

If you want to run the workflow per system by yourself, first generate the solvent system
```bash
python generatesolvent_desmond.py 1
```
where `1` can be replaxed by the system number you want to simulate (based on the row index of the `elytes.csv` file). 

Wait until a file `solvent-out.cms` is generated. Next, run MD simulation
```bash
cd 1
$SCHRODINGER/utilities/multisim -o solvent_density.cms -mode umbrella solvent-out.cms -m solvent_multisim.msj -HOST localhost
cd -
```

This outputs a file `solvent_density.cms` that we can use to estimate the solvent density. 

Next, we compute solvent density by running
```bash
$SCHRODINGER/run python3 computedensity.py 1
```

Afterwards, we build the elyte system
```bash
python generatesystem_desmond.py 1
```

And then finally run a test MD simulation
```bash
cd 1
$SCHRODINGER/utilities/multisim -o final_config.cms -mode umbrella elyte-out.cms -m elyte_multisim.msj -HOST localhost
cd -
```


### OpenMM

Run your MD simulations in this directory. Systems are already prepped in the tar file. So first, un-tar the files
```console
tar -xvf electrolytes.tar.gz
```
This should result in 3418 directories labeled in numerical order, each of which containing a set of initial input files to run the MD simulation. The number labeling each directory represents the index in the CSV file `elytes.csv`. To run an MD simulation for one system, we can go to one of the directories (let's say `0`) and do the following: 

```console
cd 0;
cp ../runsystem.py ./
python runsystem.py 0
```

If one wants to run simulations all simulations one-by-one, we can also write the following bash script
```bash
#!/bin/bash 

num_lines=$(wc -l < elytes.csv)
num_lines=$((num_lines-1))

for ((i = 0; i < num_lines; i++)); do
    cd $i
    cp ../runsystem.py ./
    python runsystem.py $i
    cd ..
done
```
which is provided in `runsimulations.sh.` Right now, the simulations are configured to perform an NPT run at 1 bar and whichever temperature relevant for the system for 500 ns

## How it works

The workflow uses Packmol to generate a system configuration and Moltemplate to generate force field files. However, the format generated is only compatible with LAMMPS. Thus, the next step is to convert the LAMMPS files to OpenMM-compatible files. 

The input to the workflow is the `ff` directory, which contains the PDB and LT files of all electrolyte components, and elytes.csv, which specifies the molar/molal concentrations of the salt and ratios for solvent mixtures. 

Because concentrations can be volumetric vs. by-weight, we often need the density of the pure solvent to determine how many moles of salt we need to put in the simulation box. Thus, there is an intermediate step of generating the pure solvent system and running a short simulation to get density data. 
