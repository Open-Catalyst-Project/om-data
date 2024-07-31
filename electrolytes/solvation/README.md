# Non-MD-based generation of noncovalently bound "solvation" structures

Elsewhere (see `md_snapshots`), we extract solvation structures and molecular clusters from classical molecular dynamics simulations using empirical (non-machine-learned) force-fields. Because these MD simulations require parameters for each species (molecule or ion), we had to significantly restrict the number of solvents, salts, and additives that we used for MD.

Here, we supplement the solvation structures and clusters from MD with noncovalently bound clusters and solvation-like environments obtained using the [`architector` package](https://doi.org/10.1038/s41467-023-38169-2). The script `solvate.py` takes as input a collection of `*.xyz` files (generated, perhaps, from the functional group substitution script in `om-data/electrolytes/functionalization/substitutions.py` or the metal-organic complexes generated in `om-data/metal_organics`), and for each of those structures, generates different kinds of noncovalently bound structures:
1. "Dimers", where the input structure is in a cluster with one other quasi-randomly-chosen molecule or ion
2. "Full solvation shells", where the input structure is surrounded by *n* instances of a single solvent (e.g. water or acetonitrile)
3. "Random solvation shells", where the input structure is surrounded by a random collection of molecules and ions

In all cases, the molecules that can be chosen to surround the input molecule are chosen from relatively modest, predefined lists that are similar to those used for our MD simulations. The user can determine how many in-distribution dimer (1) structures and random solvation shells (3) they want generated; because there are relatively few solvents to choose from, we choose only one solvent per input structure for the full solvation shells (2). In addition, there are some molecules and ions that we have labeled as "out-of-distribution". For each input structure, we generate one out-of-distribution structure inclusing those species using Method 1, 2, or 3 (chosen randomly).

## Running `solvate.py`

`solvate.py` contains predefined sets of templates and substituents and requires only a directory containing `*.xyz` files. To run:

```
python solvate.py\
--xyz_dir <directory with *.xyz files>\
--seed <random seed>\
--base_dir <output directory>\
--num_dimers <number of dimer structures to generate per input structure>\
--num_random_shells <number of random solvation shells to generate per input structure>\
--max_core_molecule_size <maximum number of atoms in the input structure>\
--max_atom_budget <total maximum number of atoms, including the input structure and the solvating molecules/ions>
```

Most parameters have defaults (run `solvate.py -h` for descriptions of these parameters).

## Outputs

After running `solvate.py`, the outputs will be found in the directory specified by the `--base_dir` argument. Within that directory will be one sub-directory per input structure, containing `*.xyz` files of the generated solvation structures with the name format `<input_prefix>_solv<index>_<charge>_<spin>.xyz`. There will also be a sub-directory called `ood`, containing the out-of-distribution outputs in the same format.

## Runtime

Because `solvate.py` requires the use of `architector`, it is quite slow. Testing on 3 input structures of 35-50 atoms with `--max_atom_budget 70` took roughly 5 minutes. Possibly this script should be run in parallel, with multiple, smaller directories of `*.xyz` files.