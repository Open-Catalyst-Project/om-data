# Electrolyte-like functional group substitution

This module contains a standalone script, `substitutions.py`, that will generate new electrolyte-like molecules by adding functional groups from a pre-defined list to a core template, where the template reflects structures used in the electrolyte literature. Functional group addition is basically random, but:
- In general, selection is biased such that small functional groups are preferred over large ones
- Molecules with uncommon bonding patterns are (sometimes, randomly) removed from the pool of generated molecules
- Certain types of molecules (namely, highly energetic molecules) are excluded

The included Jupyter notebook, `functional_group_substitution.ipynb`, is mainly for visualization of the templates and functional groups used in `substitutions.py`.

## Running `substitutions.py`

`substitutions.py` contains predefined sets of templates and substituents and requires no additional setup from the user. To run:

```
python substitutions.py\
--seed <random seed>\
--dump_path <output directory>\
--attempts_per_template <number of structures per template>\
--attempts_per_template_ood <number of structures to attempt with OOD substituents>\
--max_heavy_atoms <maximum number of heavy atoms>
```

Note that most parameters have defaults (run `substitutions.py -h` for descriptions of these parameters). Further note that, if you want to limit the total number of atoms, rather than the number of heavy (non-H) atoms, you can instead use the flag `--max_atoms <maximum number of atoms>`. Note that you cannot use use both `--max_atoms` and `--max_heavy_atoms`.

## Outputs

After running `substitutions.py`, the outputs will be found in the directory specified by the `--dump_path` argument. Within that directory will be:

- A file `initial_library_smiles.json`, containing an (unfiltered) library of generated in-distribution molecules
- A directory `xyz`, which will contain sub-directories with `*.xyz` files for each template. Each `*.xyz` file will have a name like `<template>_mol<index>_<charge>_<spin>.xyz`.
- A directory `figures`, containing figures describing the generated dataset of molecules
- A directory `ood`, which will contain similar information to that described above, but for out-of-distribution molecules (the templates and substituents that are out-of-distribution are described in `substitutions.py`)