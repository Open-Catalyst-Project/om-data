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

Simply run the driver function to generate an electrolyte system given the CSV file available and the row that you want to simulate
```
python driver_omm.py
```

```python 
import os
import runmd_omm

row_number = 1

# Only generate system if not restarting (i.e. checkpoint file doesn't exist)
checkpoint_file = os.path.join(f"{row_number}", "md.chk")
if not os.path.exists(checkpoint_file):
    import system_generator_omm
    system_generator_omm.main("csv", file="rpmd_elytes.csv", density=0.5, row=row_number)

# Run simulation, which can be restarted from a checkpoint file
result = runmd_omm.run_simulation(
    pdb_file=f"{row_number}/system.pdb",
    xml_file=f"{row_number}/system.xml", 
    output_dir=f"{row_number}",
    t_final=500.0,
    n_frames=100,
  #  rpmd=True,
  #  num_replicas=32,
    dt=0.002
)

```
Two modueles, which are `system_generator_omm` and `runmd_omm`, can also be called in the command line. Use `-h` to see what options are available. 
