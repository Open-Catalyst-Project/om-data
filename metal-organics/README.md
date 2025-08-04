# Running Architector scripts for metal-organics generation

We use the Architector package to generate structurally diverse metal-organic coordination complexes and organometallics. The process of generating these complexes is split up in to three phases: generation of Architector inputs, generation of complexes by Architector, and, finally, conversion of the Architector output to an xyz file. The metals, along with their coordination numbers, and oxidation states are taken from Architector. The ligands were selected by the script "Sampling\_ligands\_downselect\_step1". The pickled DataFrames of these two databases are

## Generating Architector Inputs

Sampling from the metal and ligand databases and generating Architector inputs is accomplished with the `architector_sampling.py` script. One must specify and output basename to use for the pickled inputs and a number of samples to generate.

```
python architector_sampling.py --outname MO_1000 --n_samples 1000
```

This script takes about 10 minutes for 100,000 samples. The result is two files: `<outname>.pkl` and `<outname>_uids`. The former is a pickle of a DataFrame which contains the Architector inputs and the latter is a plain-text file which stores the record of which complexes have been generated so that they can be skipped if more sampling is desired.

## Generation of Complexes

Architector complex generation can take from between a few minutes and 10s of hours depending on the input. As a result, it is best to launch these in parallel. To do this with OpenMP, one uses the script `mpriun.py`. One specifies the pickled DataFrame from the previous step, a number of workers, and which batch of the DataFrame to operate on (Assuming that n\_workers has been assigned to each previous batch. One also gives a path to store the output pickled DataFrame (the geometries are stored in text in the DataFrame).

```
python mprun.py MO_1000.pkl --n_workers 20 --batch_idx 2 --outpath outputs
```

This will run lines 40-59 of the DataFrame in `MO_1000.pkl` on 20 cores, with 1 core per job. A SLURM script is included which distributes the jobs over multiple nodes at once.

While most complexes finish in under 1 hour, a handfull will take much longer.

## Conversion to XYZ file

Finally, the outputs can be converted into xyz files with the `analyze_outputs.py` script. This needs to be pointed at the directory created in the previous step:

```
python analyze_outputs.py outputs
```

This will create a subdirectory named "xyzs" which will contain an xyz file for each input. The files will be named in the format `a_b_c_d.xyz`, where `a` is the line number of the sample in the pickle generated in step 1, `b` is a counter of the conformers of the given line (i.e. will range from 0 to `n` for `n` conformers), `c` is the charge of the complex, and `d` is the spin multiplicity of the complex.

# Running Crystallographic Open Database (COD) for metal-complex generation

We extract additional structures from COD to use as evaluations (and, in the case of boron and heavy-main group clusters, training data. Note that such complexes are excluded from evaluation datasets).

## Extract complexes from COD

Run the `cod_extract.py` script at scale with the `slurm_cod.sh` SLURM sbatch script. Select reasonable examples with `cod_select.py`. The results of these are checked for common issues (incorrectly labeled disordered groups for example) with `cod_flag.py`
