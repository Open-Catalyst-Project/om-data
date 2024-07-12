# Electrolytes MD Workflow

This README provides rough guide on how to prepare and run bulk electrolyte simulations. The workflow requires the following to be installed:
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
- `preparesimulations.sh`: a Bash script that will prepare the initial system configurations in the elytes.csv files for OpenMM simulations
- `runsimulations.sh`: a Bash script that will run the simulations one by one. 
- ff: a directory of force field files of all electroyte components. 
- elytes.csv: a CSV file listing all possible electrolyte systems we can simulate.
- litelytes.csv: a CSV file listing electrolyte systems curated from literature. 
- `data2lammps.py`: a Python module to generate LAMMPS DATA and run files. 
- `lammps2omm.py`: a Python module to convert LAMMPS DATA and run files to OpenMM XML and PDB files. 
- `generatesolvent.py`: a Python script to generate solvent configuration and force field files (in LAMMPS format).
- `generatesystem.py`: a Python script to generate system configuration (salt+solvent) and force field files (in LAMMPS format).
- `randommixing.py`: a Python script to generate completely random electrolytes and append them to elytes.csv file. 
- `classmixing.py`: a Python script to generate random electrolytes based on their classifications and append them to elytes.csv file. 

In theory, we only need to run preparesimulations.sh, followed by runsimulations.sh

```console
foo@bar:~$ ./preparesimulations.sh
foo@bar:~$ ./runsimulations.sh
```

## How it works

The workflow uses Packmol to generate a system configuration and Moltemplate to generate force field files. However, the format generated is only compatible with LAMMPS. Thus, the next step is to convert the LAMMPS files to OpenMM-compatible files. 

The input to the workflow is the `ff` directory, which contains the PDB and LT files of all electrolyte components, and elytes.csv, which specifies the molar/molal concentrations of the salt and ratios for solvent mixtures. 

Because concentrations can be volumetric vs. by-weight, we often need the density of the pure solvent to determine how many moles of salt we need to put in the simulation box. Thus, there is an intermediate step of generating the pure solvent system and running a short simulation to get density data. 
