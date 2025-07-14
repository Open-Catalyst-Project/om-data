
### Details for running polymer chain dissociation reactions:
* `bond_to_break` selected at random from following bonds that are (1) within 5.0 Angstrom of c.o.m. and (2) selected from the top 10 bonds with highest surrounding atom density:
    * 1 H-containing bond
    * 1 ring bond, if rings present
    * 3 non-ring bonds, 4 if no rings present
* 0.33:0.33:0.33 split for (adding H) : (removing H) : (keeping clean chain)
* 0.1:0.1:0.8 split for (adding 1e-) : (removing 1e-) : (keeping clean chain)
* AFIR max force : 10, force_increment: 0.75
* uses omol `filter_unique_structures` algorithm, yielding 15-30 per chain
* `unique_structures` trimmed to `max_atoms = 250`. Cutoff randomly selected between 4.0–6.0 Angstroms. Iteratively reduced by 0.2 Angstroms to reach `max_atoms` 250 if necessary. If fails to trim below 250, skips the structure. 
* naming of output file:
    ```bash
    $WORKING_DIR/$H_CHOICE/{$POLYMER_CLASS}_{$PDB_FILE_NAME}/afir_struct_0_charge_1_uhf_0_natoms_216_bondbreak_brCpCpCpC_to_brH_modsmarts_4_cutoff_5.71.xyz
    ```
* `$H_CHOICE/` : { `add_H/`, `remove_H/`, `none/`}
* `$POLYMER_CLASS` : { "Traditional", "Optical", "Electrolyte", "Fluoro"}
* `$PDB_FILE_NAME`: name from MD pipeline that contains monomer indices
* `bondbreak` : `[C](C)(C)(C)–[H]` -> `brCpCpCpC_to_brH`, where `[C]` is bonded to `[H]`
* `modsmarts` : index corresponds to `smarts_dict` in `omer_reactivity_pipeline.py` to indicate how the original chain was modified
* `cutoff` : cutoff used to trim the structures

### Execution:
```bsh
python omer_reactivity_pipeline.py --all_chains_dir CHAIN_DIR_PATH --csv_dir CSV_DIR --output_path OUTPUT_DIR
```
* `CHAIN_DIR_PATH` should have monomer indices and the polymer class in the path. If optical homopolymer, also needs `atomA` or `atomB` in path
* `CSV_DIR` is the dir where the .csv corresponding to the monomer indices lives
* `OUTPUT_DIR` can be whatever you like. Recommended to be separate from where the original chain pdb's live



