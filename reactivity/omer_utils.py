
import os
import re
import random
import numpy as np
import pandas as pd

from ase import Atom
from ase.data import covalent_radii

from rdkit.Chem import EditableMol, BondType, MolFromSmarts, Conformer
from rdkit.Geometry import Point3D
from rdkit.Chem import Atom as RdAtom
from io_chain import Chain, remove_bond_order, process_repeat_unit

def get_chain_path_info(pdb_path, csv_dir):
    basename = os.path.basename(pdb_path)
    smiles_lists = {}
    
    # Load all necessary SMILES lists once
    if "copolymer" in pdb_path and "Solvent" in pdb_path:
        smiles_lists['Traditional'] = pd.read_csv(os.path.join(csv_dir, 'copolymer_plus_solvent/Traditional_smiles.txt'), header=None)[0]
        smiles_lists['Fluoro'] = pd.read_csv(os.path.join(csv_dir, 'copolymer_plus_solvent/Fluoro_smiles.txt'), header=None)[0]
        smiles_lists['Electrolyte'] = pd.read_csv(os.path.join(csv_dir, 'copolymer_plus_solvent/Electro_smiles.txt'), header=None)[0]
        smiles_lists['Optical'] = pd.read_csv(os.path.join(csv_dir, 'copolymer_plus_solvent/Optical_smiles.txt'), header=None)[0]
        smiles_lists['Peptoid'] = pd.read_csv(os.path.join(csv_dir, 'peptoids.csv'), header=None)[0]
    else:
        smiles_lists['Traditional'] = pd.read_csv(os.path.join(csv_dir, 'Traditional_polymers.csv'), header=None)[0]
        smiles_lists['Fluoro'] = pd.read_csv(os.path.join(csv_dir, 'Fluoropolymers.csv'), header=None)[0]
        smiles_lists['Electrolyte'] = pd.read_csv(os.path.join(csv_dir, 'Electrolytes.csv'), header=None)[0]
        smiles_lists['A'] = pd.read_csv(os.path.join(csv_dir, 'A_optical_copolymers.csv'), header=None)[0]
        smiles_lists['B'] = pd.read_csv(os.path.join(csv_dir, 'B_optical_copolymers.csv'), header=None)[0]
        smiles_lists['Chaos'] = pd.read_csv(os.path.join(csv_dir, 'CHAOS_smiles.txt'), header=None)[0]
        smiles_lists['Peptoid'] = pd.read_csv(os.path.join(csv_dir, 'peptoids.csv'), header=None)[0]
    
    extra_smiles = []
    if 'Solvent' in basename:
        # no_other_numbers_Solvent_0000_MD_monomer000_no_other_numbers.pdb
        try:
            solv_idx = int(re.search(r'Solvent_(\d+)_MD', basename).group(1)) - 1 # 0-based indexing, assumes one solvent species
        except Exception:
            solv_idx = int(re.search(r'Solvent_(\d+)_c', basename).group(1)) - 1 # handle peptoid discrepancy, will depend on pre-processing naming
        smiles_lists['extra'] = pd.read_csv(os.path.join(csv_dir, 'solvents.csv'))['smiles'].tolist()
        extra_smiles.append(smiles_lists['extra'][solv_idx])
        basename = re.sub(r'Solvent_\d+_', '', basename) # remove solvent number from name

    # Determine polymer class
    if 'Traditional' in pdb_path:
        polymer_class = 'Traditional'
    elif 'Fluoro' in pdb_path:
        polymer_class = 'Fluoro'
    elif 'Electrolytes' in pdb_path:
        polymer_class = 'Electrolyte'
    elif 'Optical' in pdb_path:
        polymer_class = 'Optical'
    elif 'CHAOS' in pdb_path and 'Peptoid' not in pdb_path:
        polymer_class = 'Chaos'
    else:
        polymer_class = 'Peptoid'
        # raise ValueError(f"Cannot determine polymer class for {pdb_path}")

    # needs copolymer fix
    if "copolymer" in basename:
        pattern = re.search(r'copolymer_([^_]+(?:_[^_]+)*?)_(Hterm|plus)', basename)[0]
    else:
        pattern = re.search(r'monomer(\d+)_(Hterm|plus)', basename)[0]
    pattern = re.findall(r'([AB]?)(\d+)', pattern)
    
    repeat_smiles = []
    for prefix, number_str in pattern:
        idx = int(number_str) - 1  # 0-based indexing
        if prefix == 'A' or "atom_A" in pdb_path:
            repeat_smiles.append(smiles_lists['A'][idx])
        elif prefix == 'B' or "atom_B" in pdb_path:
            repeat_smiles.append(smiles_lists['B'][idx])
        else:
            # No prefix: use the current polymer class
            repeat_smiles.append(smiles_lists[polymer_class][idx])

    return repeat_smiles, extra_smiles, polymer_class

def trim_structure(chain, structure, bonds_breaking, cutoff, min_atoms=100):
    reacted_chain = Chain(structure, chain.repeat_units, extra_smiles=chain.extra_units)
    react_mol = reacted_chain.rdkit_mol

    initial_bonds = get_bonds(chain.rdkit_mol)
    reacted_bonds = get_bonds(react_mol)

    formed_bonds = [idx for tup in initial_bonds - reacted_bonds for idx in tup]
    broken_bonds = [idx for tup in reacted_bonds - initial_bonds for idx in tup]
    reacting_bonds = broken_bonds + formed_bonds

    # if no bonds broken yet, add bonds_breaking
    if any(bond not in reacting_bonds for bond in bonds_breaking):
        reacting_bonds += list(bonds_breaking)

    stop_indices = [] # all indicies that should not be deleted
    for idx in reacting_bonds:
        if idx not in stop_indices: stop_indices.append(idx)

        shielded = get_shielded_zone(reacted_chain.ase_atoms, idx, cutoff=cutoff)
        for s_idx in shielded:
            if s_idx not in stop_indices: stop_indices.append(s_idx)

    # get pattern for substruct matching chain ends and flatten
    all_smiles = list(replace_stars(repeat_unit) for repeat_unit in chain.repeat_units)
    all_smiles = [s for pair in all_smiles for s in pair]
    all_mols = list(process_repeat_unit(smiles) for smiles in all_smiles)

    # get atom mapping for rdkit and ase
    new_atoms = reacted_chain.ase_atoms

    clean_mol = remove_bond_order(react_mol)
    # iterate through chain ends, starting with those farthest from bonds_breaking
    reacted_ends = sort_by_bond_distance(reacted_chain.ase_atoms, bonds_breaking, reacted_chain.ends)[::-1]
    for i in range(len(reacted_ends)):
        max_capped = False
        stop_positions = [new_atoms[idx].position.copy() for idx in stop_indices]
        end_positions = [new_atoms[idx].position.copy() if idx is not None else None
                            for idx in reacted_ends]
        
        current_idx = reacted_ends[i]
        if current_idx is None:
            # print("STOP: Chain end previously deleted")
            continue
        
        while not max_capped and len(new_atoms) > min_atoms:
            new_atoms = new_atoms.copy()
            matches = list(clean_mol.GetSubstructMatches(mol) for mol in all_mols)       
            match = None
            match_mol = None

            for mol, mol_matches in zip(all_mols, matches):
                for m in mol_matches:
                    if current_idx in m: # take first match found
                        match = m
                        match_mol = mol
                        break
                if match is not None:
                    break

            if match is None:
                # print("ERROR: No match")
                max_capped = True
                break
            
            if any(idx in stop_indices for idx in match):
                # print("STOP: Hit spheres of influence")
                max_capped = True
                break
            
            # Get star atom from pattern
            query_star_idx = [i for i, atom in enumerate(match_mol.GetAtoms()) if atom.GetAtomicNum() == 0][0]
            idx_to_keep = match[query_star_idx]
            rdkit_atom_to_keep = clean_mol.GetAtomWithIdx(idx_to_keep)

            pos_to_keep = new_atoms[idx_to_keep].position

            if rdkit_atom_to_keep.GetAtomicNum() == 1: # only one repeat unit
                # print("STOP: whole chain will be deleted")
                new_atoms = delete_from_ase(new_atoms, match)
                clean_mol = delete_from_rdkit(clean_mol, match)

                stop_indices, reacted_ends, _ = reset_indices(new_atoms, stop_positions, end_positions, [0,0,0])
                
                max_capped = True
                break

            # Get position of neighbor attached to star atom
            heavy_neighbors = [ neighbor for neighbor in rdkit_atom_to_keep.GetNeighbors()
                if neighbor.GetAtomicNum() > 1  and neighbor.GetIdx() in match
            ]
            idx_for_pos = heavy_neighbors[0].GetIdx()
            old_pos = new_atoms[idx_for_pos].position

            direction = old_pos - pos_to_keep
            direction /= np.linalg.norm(direction)

            # add solvent matches for removal
            extra_to_delete = []
            for extra in chain.extra_rdkit_mol:
                extra = remove_bond_order(extra)
                extra_matches = clean_mol.GetSubstructMatches(extra)
                for extra_match in extra_matches:
                    if any(idx in stop_indices for idx in extra_match):
                        continue
                    extra_to_delete += extra_match

            # Do not delete the atom to keep
            remove_from_match = list(match) + extra_to_delete
            remove_from_match.remove(idx_to_keep)
            tuple(remove_from_match)
            
            # Add H then delete others ASE 
            old_Z = new_atoms[idx_for_pos].number
            new_bond_length = covalent_radii[old_Z] + covalent_radii[1]
            new_pos = pos_to_keep + direction * new_bond_length

            new_atoms += Atom('H', position=new_pos) 

            new_atoms = delete_from_ase(new_atoms, remove_from_match)

            # Add H then delete others RDkit 
            clean_mol = add_to_rdkit(clean_mol, idx_to_keep, new_pos)
            clean_mol = delete_from_rdkit(clean_mol, remove_from_match)

            stop_indices, reacted_ends, idx_to_keep = reset_indices(new_atoms, stop_positions, end_positions, pos_to_keep)
            current_idx = idx_to_keep

    return new_atoms

def trim_structures(chain, unique_structures, bonds_breaking, max_atoms=250, delta_cutoff=0.2):
    trimmed_structures = []
    cutoff = random.uniform(4.0, 6.0)

    last_structure = unique_structures[-1]
    last_structure.arrays['residuenames'] = np.copy(chain.ase_atoms.arrays['residuenames'])

    last_trimmed = trim_structure(chain, last_structure, bonds_breaking, cutoff)
    while len(last_trimmed) > max_atoms:
        new_cutoff = cutoff - delta_cutoff
        if new_cutoff < 0:
            break
        cutoff = new_cutoff
        last_trimmed = trim_structure(chain, last_structure, bonds_breaking, cutoff)
    last_trimmed.info['trim_cutoff'] = cutoff

    trimmed_pos = last_trimmed.get_positions().copy()
    original_pos = last_structure.get_positions().copy()

    original_set = set(map(tuple, original_pos))
    trimmed_set = set(map(tuple, trimmed_pos))

    deleted_pos = original_set - trimmed_set
    added_pos = trimmed_set - original_set

    deleted_indices = [i for i, pos in enumerate(original_pos) if tuple(pos) in deleted_pos]
    added_indices = [i for i, pos in enumerate(trimmed_pos) if tuple(pos) in added_pos]

    added_H_info = [] # bring back H caps
    for i in added_indices:
        if last_trimmed[i].symbol != 'H':
            continue  

        pos_H = last_trimmed[i].position.copy()
        dists = np.linalg.norm(trimmed_pos - pos_H, axis=1)
        nearest_idx = np.argmin([
            d if last_trimmed[j].symbol != 'H' else np.inf
            for j, d in enumerate(dists)])
        
        direction = pos_H - last_trimmed[nearest_idx].position.copy()
        old_Z = last_trimmed[nearest_idx].number
        new_bond_length = covalent_radii[old_Z] + covalent_radii[1]

        direction /= np.linalg.norm(direction)
        added_H_info.append((nearest_idx, direction * new_bond_length))

    for i in range(len(unique_structures) - 1):
        structure = unique_structures[i]
        new_atoms = delete_from_ase(structure, deleted_indices)
        for idx, offset in added_H_info:
            pos = new_atoms[idx].position.copy() + offset
            new_atoms.append(Atom('H', position=pos))

        new_atoms.info['trim_cutoff'] = cutoff
        trimmed_structures.append(new_atoms)
        assert len(new_atoms) == len(last_trimmed)
    
    trimmed_structures.append(last_trimmed)
    return trimmed_structures

def get_bonds(rdkit_mol):
    bonds = set()
    for bond in rdkit_mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonds.add(tuple(sorted((i, j))))
    return bonds

def get_shielded_zone(atoms, center_idx, cutoff=5.0):
    center_pos = atoms[center_idx].position
    all_positions = atoms.get_positions()
    distances = np.linalg.norm(all_positions - center_pos, axis=1)
    neighbor_indices = [i for i, d in enumerate(distances) if d <= cutoff and i != center_idx]
    return neighbor_indices

def sort_by_bond_distance(atoms, bond, indices):
    i, j = bond
    pos_i = atoms[i].position
    pos_j = atoms[j].position

    def dist_to_bond(idx):
        pos = atoms[idx].position
        return min(np.linalg.norm(pos - pos_i), np.linalg.norm(pos - pos_j))
    
    sorted_indices = sorted(indices, key=dist_to_bond)
    return sorted_indices

def replace_stars(smiles):
    smiles = smiles.replace('[*]', '*')
    star_indices = [i for i, c in enumerate(smiles) if c == '*']
    if len(star_indices) != 2:
        raise ValueError("SMILES must contain exactly two '*' atoms.")
    
    first_star_smiles = smiles[:star_indices[1]] + '[H]' + smiles[star_indices[1]+1:]
    second_star_smiles = smiles[:star_indices[0]] + '[H]' + smiles[star_indices[0]+1:]
    
    return first_star_smiles, second_star_smiles

def reset_maps(new_atoms, clean_mol):
    rdkit_to_ase = {}  # key: rdkit_idx, value: ase_idx
    ase_to_rdkit = {}  # key: ase_idx, value: rdkit_idx

    rdkit_conf = clean_mol.GetConformer()
    ase_positions = np.array([atom.position for atom in new_atoms])

    for rdkit_idx in range(clean_mol.GetNumAtoms()):
        pos = rdkit_conf.GetAtomPosition(rdkit_idx)
        pos = np.array([pos.x, pos.y, pos.z])
        
        # Find closest ASE atom (within small distance tolerance)
        distances = np.linalg.norm(ase_positions - pos, axis=1)
        ase_idx = int(np.argmin(distances))
        
        if distances[ase_idx] < 0.1:
            rdkit_to_ase[rdkit_idx] = ase_idx
            ase_to_rdkit[ase_idx] = rdkit_idx

    return rdkit_to_ase, ase_to_rdkit

def reset_indices(new_atoms, stop_positions, end_positions, pos_to_keep):
    ase_positions = np.array([atom.position for atom in new_atoms])
    remap_stop_indices = []
    for pos in stop_positions:
        distances = np.linalg.norm(ase_positions - pos, axis=1)
        new_idx = int(np.argmin(distances))
        if distances[new_idx] < 0.1:  # adjust tolerance if needed
            remap_stop_indices.append(new_idx)
    
    remap_reacted_indices = []
    for pos in end_positions:
        if pos is None:
            remap_reacted_indices.append(None)
            continue
        distances = np.linalg.norm(ase_positions - pos, axis=1)
        new_idx = int(np.argmin(distances))
        if distances[new_idx] < 0.1:  # adjust tolerance if needed
            remap_reacted_indices.append(new_idx)
        else:
            remap_reacted_indices.append(None)
    remap_to_keep = None
    distances = np.linalg.norm(ase_positions - pos_to_keep, axis=1)
    new_idx = int(np.argmin(distances))
    if distances[new_idx] < 0.1:  # adjust tolerance if needed
        remap_to_keep = new_idx

    return remap_stop_indices, remap_reacted_indices, remap_to_keep

def delete_from_ase(atoms, remove_from_match):
    keep_mask = []
    for i in range(len(atoms)):
        if i not in remove_from_match:
            keep_mask.append(i)
    atoms = atoms[keep_mask]
    return atoms

def delete_from_rdkit(clean_mol, remove_from_match):
    emol = EditableMol(clean_mol)
    for idx in sorted(remove_from_match, reverse=True):
        emol.RemoveAtom(idx)
    clean_mol = emol.GetMol()
    return clean_mol

def add_to_rdkit(clean_mol, new_atom_to_keep, position):
    emol = EditableMol(clean_mol)
    new_h = RdAtom('H')
    new_h.SetNoImplicit(True)

    new_rdkit_idx = emol.AddAtom(new_h)
    emol.AddBond(new_atom_to_keep, new_rdkit_idx, BondType.SINGLE)

    mol = emol.GetMol()
    conf = Conformer(mol.GetNumAtoms())

    orig_conf = clean_mol.GetConformer()
    for i in range(clean_mol.GetNumAtoms()):
        pos = orig_conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, pos)

    conf.SetAtomPosition(new_rdkit_idx, Point3D(*position))
        
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    return mol

def get_bond_smarts(mol, idx1, idx2):
    atom1 = mol.GetAtomWithIdx(idx1)
    atom2 = mol.GetAtomWithIdx(idx2)

    # Get element symbols
    sym1 = atom1.GetSymbol()
    sym2 = atom2.GetSymbol()

    # Get neighbors excluding the bonded atom
    nbs1 = [nbr.GetSymbol() for nbr in atom1.GetNeighbors() if nbr.GetIdx() != idx2]
    nbs2 = [nbr.GetSymbol() for nbr in atom2.GetNeighbors() if nbr.GetIdx() != idx1]

    # Format: "[A](N1)(N2)[B](N3)(N4)" etc.
    frag1 = f"[{sym1}]" + ''.join(f"({n})" for n in nbs1)
    frag2 = f"[{sym2}]" + ''.join(f"({n})" for n in nbs2)

    return f"{frag1}-{frag2}"

def does_not_overlap(pos, atoms, threshold=0.5):
    positions = atoms.get_positions()
    if len(positions) == 0:
        return True
    dists = np.linalg.norm(positions - pos, axis=1)
    return np.all(dists > threshold)

def reset_idx(new_atoms, old_atoms, idx_list):
    old_positions = np.array([atom.position for atom in old_atoms])
    new_positions = np.array([atom.position for atom in new_atoms])
    new_idx_list = tuple(int(np.argmin(np.linalg.norm(new_positions - old_positions[idx], axis=1))) for idx in idx_list)
    return new_idx_list

def add_h_to_chain(chain, bonds_reacting, attempts=5):
    new_atoms = chain.ase_atoms.copy()

    # first atom listed in reactants needs to be the one that's protonated in products
    smarts_dict = {"reactants": ["[#7D3]","[#7][#1]",
                                 "[#8D1]", "[#8][#1]", "[#6;R0]=[#6;R0]", # avoid hydrogenation of aromatic rings
                                 "[#7]#[#6]", "[#7D2]=[#6]"],
                   "products": ["[#7D3;+1][#1]","[#7;+1]([#1])[#1]", 
                                "[#8D1;+1][#1]", "[#8;+1]([#1])[#1]", "[#6;+1]([#1])[#6]",
                                "[#7;+1]([#1])#[#6]", "[#7D2;+1]([#1])=[#6]"]}
    reacting_region = []
    for idx in bonds_reacting:
        reacting_region += get_shielded_zone(new_atoms, idx)

    all_valid_matches = []
    all_smarts = []
    for i in range(len(smarts_dict["reactants"])):
        reactant_smarts = smarts_dict["reactants"][i]

        pattern = MolFromSmarts(reactant_smarts)
        chain_mol = chain.rdkit_mol
        matches = chain_mol.GetSubstructMatches(pattern)
    
        valid_match = [match[0] for match in matches if match[0] in reacting_region]
        if valid_match:
            all_valid_matches.append(valid_match)
            all_smarts.append(reactant_smarts + "." + smarts_dict["products"][i])
 
    if not all_valid_matches:
        return chain 
    
    for attempt in range(attempts):
        i = random.randrange(len(all_valid_matches))
        valid_match = all_valid_matches[i]
        match_smarts = all_smarts[i]
        match = random.choice(valid_match)

        conf = chain_mol.GetConformer()
        new_pos = get_idealized_H_position(match, chain_mol, conf)

        if does_not_overlap(new_pos, new_atoms):
            new_atoms.append(Atom('H', new_pos))
            new_atoms.info["charge"] = +1
            new_atoms.info["mod_smarts"] = match_smarts 
            new_chain = Chain(new_atoms, chain.repeat_units, extra_smiles=chain.extra_units)

            return new_chain 
        else:
            continue
    return chain

def remove_h_from_chain(chain, bonds_reacting):
    new_atoms = chain.ase_atoms.copy()
    # avoid removing H from C in ring to avoid aromatic problems
    smarts_dict = {"reactants": ["[#1][#6;R0][!#1]","[#1][#8]", "[#1][#7]([!#1])[!#1]",
                                  "[#1][#7]([#1])[!#1]", "[#1][#16]","[#1][#6]#[#6]",
                                  "[#1][#6][#6](=[#8])[#6]"],
                   "products": ["[#6;+1][!#1]", "[#8;-1]","[#7;-1]([!#1])[!#1]",
                                "[#7;-1]([#1])[!#1]", "[#16;-1]", "[#6;-1]#[#6]",
                                "[#6]=[#6]([#8;-1])[#6]"],
                    "charge": [+1, -1, -1, -1, -1, -1, -1]
                   }
    
    reacting_region = []
    for idx in bonds_reacting:
        reacting_region += get_shielded_zone(new_atoms, idx)

    # react all the smarts that apply
    all_valid_matches = []
    all_smarts = []
    all_charges = []
    for i in range(len(smarts_dict["reactants"])):
        reactant_smarts = smarts_dict["reactants"][i]
        charge = smarts_dict["charge"][i]

        pattern = MolFromSmarts(reactant_smarts)
        chain_mol = chain.rdkit_mol
        matches = chain_mol.GetSubstructMatches(pattern)
    
        valid_match = [match[0:2] for match in matches if match[0] in reacting_region]
        if valid_match:
            all_valid_matches.append(valid_match)
            all_smarts.append(reactant_smarts + "." + smarts_dict["products"][i])
            all_charges.append(charge)
    
    if not all_valid_matches:
        return chain, bonds_reacting
    
    carbocat_patt = smarts_dict["reactants"][0] + "." + smarts_dict["products"][0]
    has_first = carbocat_patt in all_smarts
    probabilities = []
    for pat in all_smarts:
        if pat == carbocat_patt:
            probabilities.append(1/14) 
        else:
            num_other = len(all_smarts) - (1 if has_first else 0)
            prob = (13/14) / num_other if num_other > 0 else 0
            probabilities.append(prob)
    chosen_smarts = random.choices(all_smarts, weights=probabilities, k=1)[0]
    i = all_smarts.index(chosen_smarts)
    valid_match = all_valid_matches[i]
    match = random.choice(valid_match)
    highlight_idx = match[1]
    match = match[0]

    mask = np.ones(len(new_atoms), dtype=bool)
    mask[list([match])] = False
    new_atoms =  new_atoms[mask]
    new_atoms.info["charge"] = all_charges[i]
    new_atoms.info["mod_smarts"] = chosen_smarts

    index_map = {}
    new_idx = 0
    for old_idx in range(len(chain.ase_atoms)):
        if old_idx == match:
            index_map[old_idx] = None
        else:
            index_map[old_idx] = new_idx
            new_idx += 1

    new_chain = Chain(new_atoms, chain.repeat_units, extra_smiles=chain.extra_units)
    new_bonds_reacting = [index_map[idx] for idx in bonds_reacting if index_map[idx] is not None]
    highlight_idx = [index_map[idx] for idx in [highlight_idx] if index_map[idx] is not None]
    
    return new_chain, new_bonds_reacting 

def get_idealized_H_position(n_idx, chain_mol, conf, bond_length=1.01):
    atom = chain_mol.GetAtomWithIdx(n_idx)
    neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
    
    heavy_neighbor_pos = [np.array(conf.GetAtomPosition(n)) 
                          for n in neighbors if chain_mol.GetAtomWithIdx(n).GetAtomicNum() >= 1]
    atom_pos = np.array(conf.GetAtomPosition(n_idx))

    if len(heavy_neighbor_pos) == 3:
        vec1 = heavy_neighbor_pos[0] - atom_pos
        vec2 = heavy_neighbor_pos[1] - atom_pos
        vec3 = heavy_neighbor_pos[2] - atom_pos

        normal1 = np.cross(vec1, vec2)
        normal2 = np.cross(vec2, vec3)
        normal3 = np.cross(vec3, vec1)
        normal = (normal1 + normal2 + normal3) / 3.0

        direction = normal / np.linalg.norm(normal)

        trial_pos = atom_pos + bond_length * direction
        dists = [np.linalg.norm(trial_pos - pos) for pos in heavy_neighbor_pos]
        if min(dists) < bond_length:  # too close to one neighbor
            direction = -direction

        return atom_pos + bond_length * direction

    elif len(heavy_neighbor_pos) == 2:
        vec1, vec2 = heavy_neighbor_pos
        direction = - (vec1 + vec2 - 2 * atom_pos)
        unit_dir = direction / np.linalg.norm(direction)
        return atom_pos + bond_length * unit_dir

    elif len(heavy_neighbor_pos) == 1:
        direction = - (heavy_neighbor_pos[0] - atom_pos)
        unit_dir = direction / np.linalg.norm(direction)
        return atom_pos + bond_length * unit_dir

    else:
        all_pos = [np.array(conf.GetAtomPosition(n)) for n in neighbors]
        direction = - (np.mean(all_pos, axis=0) - atom_pos)
        unit_dir = direction / np.linalg.norm(direction)
        return atom_pos + bond_length * unit_dir

def surround_chain_with_extra(chain, bond=None, remove=False, max_atoms=250):
    from ase.geometry import find_mic
    ase_atoms = chain.ase_atoms.copy()
    positions, cell = ase_atoms.get_positions(), ase_atoms.get_cell()
    chain_list, extra_list = chain.get_idx_lists()

    ref_pos = positions[chain_list]
    ref_com = np.mean(ref_pos, axis=0)
    for idx_group in extra_list:
        mol_pos = positions[list(idx_group)]
        mol_com = np.mean(mol_pos, axis=0)

        dr = mol_com - ref_com
        dr_wrapped = find_mic(dr[np.newaxis, :], cell)[0]

        dist_raw = np.linalg.norm(dr)
        if dist_raw > 5.0:
            translation = np.squeeze(dr_wrapped) - dr
            for idx in idx_group:
                positions[idx] += translation

    ase_atoms.set_positions(positions)

    if remove:
        flat_extra_list = list(match for matches in extra_list for match in matches)
        flat_extra_list = sorted(flat_extra_list, reverse=True)
        near_chain_list = []
        for idx in chain_list:
            near_chain_list += get_shielded_zone(ase_atoms, idx, cutoff=8.0)
        sorted_near_chain = sort_by_bond_distance(ase_atoms, bond, near_chain_list)

        keep_set = set(chain_list)
        for idx in sorted_near_chain:
            if len(keep_set) >= max_atoms:
                break
            matching_groups = [group for group in extra_list if idx in group]

            for group in matching_groups:
                keep_set.update(group) 

        kept_indices = sorted(keep_set)
        ase_atoms = ase_atoms[kept_indices]
    
    return Chain(ase_atoms, chain.repeat_units, extra_smiles=chain.extra_units)

def is_bonded(ase_atoms, bonded_atoms_list, scale=1.2):
    i, j = bonded_atoms_list
    d = ase_atoms.get_distance(i, j, mic=True)
    cutoff = scale * (covalent_radii[ase_atoms[i].number] + covalent_radii[ase_atoms[j].number])
    return bool(d < cutoff)
