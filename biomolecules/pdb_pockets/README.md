# PDB-Ligand pockets

We wish to extract ligands and their associated pockets from the PDB. We use a number of tools for this, partly because the PDB data is very dirty and often not readily applied to atomistic simulation without some curation. Pockets are taken from the BioLiP2 database. The procedure is as follows:

### Prepare the BioLiP2 database

1) Download the database
2) Remove peptide, DNA, and RNA ligands
3) Remove duplicate pockets
    a) Pockets with the same ligand binding to exactly the same residues (including numbering) are uniquified. We don't require the PDB IDs to also match as closely related proteins or structures may have conserved binding sites which are still duplicates of each other.
    b) Pockets with the same ligand in the same PDB structure, often on different copies of the same chain are uniquifid. Sometimes there are residues that are right at the edge of inclusion, and so the pockets may be "different" by inclusion of an additional, marginal residue. We opt for only the smallest of such binding sites since these indicate the key interactions.
4) Label a pocket as drug-like if it has any binding affinity data

### Extracting pockets

1) Download the PDB files
2) Add protons, fix missing sidechains, optimize hydrogen bonding networks, adjust metal bonding, and various other cleanups with the Schrodinger Protein Preparation Wizard (free for academics). We also enumerate protonation/tautomeric states for the ligands at this stage with the Schrodinger Epik package (not free for academics)
3) Extract the pocket as described in the prepared BioLiP2 database
    a) We take all residues labeled as part of the pocket (labeled as chain "A")
    b) If there are waters present in the pocket (defined as within 2.5 A of the ligand), include them labeled as chain "c" (for "coordinated")
    c) If the ligand is a simple ion (i.e. a single atom), also include any non-water ligand with 2.5 A
    d) The ligand is extracted as chain "l" (for "ligand")
4) Cap all residues with NMA or ACE
    a) if two residues are separated by exactly one residue in the original protein, then these capping group will overlap. Instead of a cap, the one missing residue is also included (but mutated to GLY to not increase atom counts unduly) so that the two residues are connected into a single chain again
    b) ACE and NMA capping groups are set to the same phi/psi angles as the adjacent residues in the original protein (that is, the CA of these groups matches the CA of the original structure)
4) Some pockets in practice span the boundary between two receptor chains. The BioLiP 2 database considers the ligand interacting with each chain to be separate pockets, even though that uses the same physical ligand twice. Since we want an accurate picture of the physical pocket, we combine multiple pockets which share the same physical ligand into one pocket.
5) Some ligands are covalently bound to other ligands, we cap each side of the disrupted covalent bond with an H.
6) The extracted pockets are subjected to further cleanup procedures to ensure that the pockets will both work in MD and DFT calculations. Approximate numbers of structures that required each correction are given in parenthesis (note that some structures need multiple corrections):
    a) Correction ion charges (~34000): ensure that:
        i) Li, Na, K, Rb, Cs, and Ag ions have +1 charge,
        ii)  Mg, Ca, Sr, Zn, Cd, Hg have +2 charge,
        iii) Al, Eu have +3 charge,
        iv) if Fe is present in state other than +2 or +3, make it +2,
        v) if Cu is present in a state other than +1 or +2, make it +2,
        vi) if Sn is present in a state other than +4, make it +4 but reduce that charge for any Sn-C or Sn-S bonds.
    b) Lewis structure charge correction (~12000): Analyze the Lewis structure of the system and see if there are any places where the specified charges are inconsistent with that Lewis structure. For example, O=N([O-])([O-]) to O=[N+]([O-])([O-])
    c) Allow open-shell species (~9000): Not all pockets in the PDB are closed-shell, often because they contain metal ions with odd numbers of electrons
    d) Lewis structure hydrogen correction (~7000): Correct protonation states that are incompatible with the Lewis structure, e.g. C[O] to CO
    e) Disrupted disulfide (~6000): The extraction procedure can possibly take one half of a disulfide bond, we cap the kept half with an H
    f) Guandinium charge correction (~3000): Guanadinium is present in the side-chain of arginine. This group is often present as a +1 charge, but because of Lewis structure labeling inconsistencies, it can be accidentally labeled as +2 in some of our workflow. We correct it back to +1.
    g) Problems with BeF3 (~2000): Schrodinger tools do not like BeF3, we therefore have to correctly restore proper bonding and charge assignments
    h) Unusually long bonds (~1000): sometimes the PDB files including very long bonds (H-X > 1.4 A, X-X > 1.85 A, where X is any first row element)  which disrupt Lewis structure assignment and analysis, these are removed
    i) Remove HF (~1000): Schrodinger tools likes to turn F- into HF, this is unphysical and are removed
    j) Deprotonate carbonyl ligands (~300): there are metal complexes with carbonyl ligands in some pockets, Schrodinger tools add an H to the carbonyl C. This is removed
    k) Assign name to unknown residues (~200): Sometimes the residue identity isn't known in a PDB and are often filled in with ALA or GLY residues. Since we wish to accurately describe the pocket extracted (as opposed to any specific protein sequence), we add the correct residue name for what is present. This is important as one of our splits is on non-canonical amino acids and we don't want that contaminated with what are effectively standard amino acids.
    l) Pentavalent carbons (~150): In a few cases, we have carbon making more than 4 bonds including a double bond, that double bond is reduced to a single bond
    m) Meld ACE and NMA grops (~100): In a few cases, the ACE and NMA capping groups become so close that the are overlapping, in this case the ACE and NMA groups are removed and replaced with a GLY coordinated to both capped groups
7) Porphyrins are important groups present in many ligands. Epik seems to have a hard time realizing that they are deprotonated when a metal is present in them. We automatically create an additional protonation state for a list of 39 ligand ids that contain porphyrins where the H's are properly removed as needed. TODO: We will do the same for phosphates and sulfates 
8) Statistics are collected on each cleaned up pocket to assess which will proceed for MD calculation. We also remove metal-metal zero-order bonds at this stage (as well as certain metal-ligand in order to promote structural diversity as the Desmond MD engine freezes zero-order bonded bonds). Among the statistics collected are whether the pocket with the specified charge and spin is physically valid and whether the Lewis structure seems to be valid. At this stage, there are 542746 pockets.
9) We select which systems will be subject to MD.
    1) Systems with 350 or fewer atoms, physical charge and spin states, which pass the Lewis structure checks (415374 structures)
    2) Systems with 350 or fewer atoms, physical charge and spin states, containing Be (751 structures, Schrodinger flags all systems with BeF3 as bad Lewis strucrtures)
10) MD calculations are performed per the procedure described below
11) Frames from the MD trajectories are extracted:
    a) Frames that are within 1 A RMSD of the starting structure are discarded (this happens not infrequently for smaller systems with just ions)
    b) Frames were the ligand center of mass has moved more than 5A from the starting COM are discarded
    c) The remaining frames are discretized into equally spaced frames of at most 10 frames.
    d) The first and last frame (i.e. the two most dissimilar in time) are passed to DFT


## Running the workflow

There are several steps to running the workflow. All steps must be run with the Schrodinger Python executable. Since some of these scripts use the `multiprocessing` package to speed-up calculations, which is normally incompatible with Schrodinger tools due to Qt, one must also `export SCHRODINGER_ALLOW_UNSAFE_MULTIPROCESSING=1`. Each of these scripts takes an argument `--output_path` which points to the directory where pockets will be generated and manipulated. Here is the sequence of steps to run:

1) `slurm_launch.sh` (i.e. `biolip_extraction.py` via SLURM)
1b) `slurm_launch.sh` again with `--no_fill_sidechain` set (only needed for PDB pockets that failed with "Atoms are missing" error in the previous step)
2) `combine_pockets.py`
3) `cap_covalently_attached_ligands.py`
4) `cleanup.py`
5) `cleanup_heme.py`
6) `some_stats.py`
7) `run_md.py`
8) `run_md.sh` (i.e. Desmond via SLURM) with both the `temp='300K'` and `temp='400K'`
9) `filter_md.py`
10) `mae2xyz_pockets.py`



## Outputs

At the end of the this workflow, there will be a directory called `md/frames/<date>_inputs` in the `--output_path` directory used above which contains .xyz files of protein-ligand pockets with filenames that end in `_n_m.xyz` where `n` is the charge, and `m` is the multiplicity of the pocket.

`some_stats.py` creates a pickled DataFrame, which is located at `https://dl.fbaipublicfiles.com/opencatalystproject/data/large_files/pdb_pockets_stats.pkl`
