List of questions as of 5/30:

1. Do we want to remove the other solutes (including counterion on the same solute) when doing the solvation analysis?
2. How to handle multiple solutes or solvents with the same atoms present? Can't just select by atom names
    One Idea: Pass in a list of lists of solutes and solvents - how to automatically get this from the PDB or LAMMPS file?
3. How do we want to deal with coordination numbers? I.e. pick a defined min and max as we're doing currently, or pick the k most common coordination numbers? Or pick a set number from each type of coordination number? Over-sample the rare ones?
4. For the case of multiple solvents, not enough to split up shells by coordination number - need to also split up by number of each solvent.
5. What to do when it's only an ionic liquid (no solvent)


Notes from chatting with Evan:

Maybe put ionic liquids on hold

Focus on the case where we have one or more solutes and one or more solvents

Compute average solvation radius over all atoms in the solute (or randomly sample a subset for larger solute molecules)

Anything that isn't the solute under consideration needs to be defined as a separate solvent (with some cutoff to ensure that we don't have silly comparison e.g between two cations.)

Save rdfs and rdf data/troughs, etc so that if needed, we can do a post-analysis with tabulaed radii

For solvent-solvent interactions, try two approaches:
1. Same analysis code as with solutes, but just make the solvent a solute (may not work if there aren't clear RDFs for the solvent)
2. Pick random solvent molecules, pick a random radius with bounds determined by a solvent-dependent heuristic, sample shells, sort by composition, and pick most diverse



Meeting on 6/7:

Main problem with the current analysis pipeline is that there are different analysis approaches for low concentration, high concentration, ionic liquids, etc. Also unclear how to define radii for multi-atom solutes. This is going to be too slow. 

Maybe the simplest approach is to choose 2-3 heuristic radii (not system specific) around each solute and solvent and then bin them by composition/graph hash. 

How do we determine these radii? Go through a few sample pdb files with different cases (multi solute, multi solvent, ionic liquid, etc.) and then do some heuristic solvation analysis to find the radii.

Question: when we expand a shell around a solute, if we run into another solute, do we also expand around that? Probably not needed because this will be captured in the next larger radius. 

Schrodinger has easy ways to do this. Sanjeev/Daniel sit down and work with Schrodinger to see if it solves all these problems.

LAMMPS files from Muhammad need to have a mapping from atom to residue names and solute/solvent. Muhammad and Evan work on this together


Meeting of 6/11 with Daniel:
MGCF at Berkeley
Maestro for visualizing stuff

Schrodinger easily accounts for different atom orderings (e.g permutations, etc.) and PBCs









