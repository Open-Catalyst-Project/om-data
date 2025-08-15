import os
import random
import numpy as np

import argparse
import traceback

from architector.io_molecule import Molecule
from omdata.reactivity_utils import filter_unique_structures, run_afir
from ase.io import write, read

from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase import neighborlist
from ase.data import covalent_radii

allowed_pairs = {frozenset(pair) for pair in [
    ('O', 'H'), ('O', 'O'), ('C', 'O'), ('C', 'C'), ('C', 'H'), ('N', 'H')
]}

def find_adsorbate_pairs(atoms, ads_tag=2, max_dist=5.0):
    ads_indices = np.where(atoms.get_tags() == ads_tag)[0]
    symbols = atoms.get_chemical_symbols()
    
    # get neighbor lists
    cutoffs = [covalent_radii[atoms[i].number] for i in range(len(atoms))]
    nl = neighborlist.NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    # id all atoms in adsorbates (includes neighbors)
    mol_id = {idx: None for idx in ads_indices}
    visited = set()
    current_id = 0
    for idx in ads_indices:
        if mol_id[idx] is not None:
            continue
        stack = [idx]
        while stack:
            i = stack.pop()
            if i in visited or i not in ads_indices:
                continue
            visited.add(i)
            mol_id[i] = current_id
            neighs, _ = nl.get_neighbors(i)
            for n in neighs:
                if n in ads_indices and mol_id[n] is None:
                    stack.append(n)
        current_id += 1

    valid_pairs = []
    symbols_list = []
    terminal_bonds = []
    for i in ads_indices:
        for j in ads_indices:
            if i >= j: continue
            # is an allowed pair
            if frozenset([symbols[i], symbols[j]]) not in allowed_pairs: continue
            # not in same adsorbate fragment
            if mol_id[i] == mol_id[j]: continue
            # within distance cutoff
            dist_ij = np.linalg.norm(atoms.positions[j] - atoms.positions[i])
            if dist_ij > max_dist: continue

            unit_ij = (atoms.positions[j] - atoms.positions[i]) / dist_ij
            atom_between = False
            for k in range(len(atoms)):
                if k in (i, j):
                    continue
                vec_ik = atoms.positions[k] - atoms.positions[i]
                proj_len = np.dot(vec_ik, unit_ij)
                if 0 < proj_len < dist_ij:
                    perp_dist = np.linalg.norm(vec_ik - proj_len * unit_ij)
                    if perp_dist < 1.0:
                        atom_between = True
                        break
                        
            if not atom_between:
                valid_pairs.append((int(i), int(j)))
                symbols_list.append(symbols[i] + "–" + symbols[j])

                # check if a terminal atom is involved
                term_info = []
                for atom_idx in (i, j):
                    neighs, _ = nl.get_neighbors(atom_idx)
                    neighs_ads = [n for n in neighs if atoms.get_tags()[n] == ads_tag]
                    if len(neighs) == 1:  # terminal atom
                        term_info.append((int(atom_idx), int(neighs[0])))
                    elif len(neighs_ads) == 1 and symbols[atom_idx]=="H": # catch terminal Hs
                         term_info.append((int(atom_idx), int(neighs_ads[0])))
                terminal_bonds.append(tuple(term_info))
    
    return valid_pairs, symbols_list, terminal_bonds
                
def ocat_react_pipeline(traj, output_path, cache_dir, return_ase=False):
    atoms = read(traj)
    pairs, symbols, term_bonds = find_adsorbate_pairs(atoms)
    if not pairs: 
        pairs, symbols, term_bonds = find_adsorbate_pairs(atoms, max_dist=10.0)

    if not pairs: raise ValueError("Failed to extract adsorbate pairs")

    i = random.choice(range(len(pairs)))
    bond_to_form = pairs[i]
    react_symbols = symbols[i]
    bonds_to_break = random.choice([(), term_bonds[i]])
    if bonds_to_break: react_symbols += "_terminalbondbreak_true"
    
    name = os.path.splitext(os.path.basename(traj))[0]
    os.makedirs(os.path.join(output_path, name), exist_ok=True)
    logfile = os.path.join(output_path, name, "logfile.txt")

    mol = Molecule(atoms)
    mol.ase_atoms.info['charge'] = 0
    mol.ase_atoms.info['uhf'] = 1
    mol.charge = 0
    mol.uhf = 1

    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda",
                                                    inference_settings="turbo",
                                                    cache_dir=cache_dir,)
    UMA = FAIRChemCalculator(predictor, task_name="oc20")
    
    save_trajectory, _ = run_afir(mol, None, UMA, logfile,
                                    bonds_forming=[bond_to_form], 
                                    bonds_breaking=bonds_to_break,
                                    is_crystal=True, skip_first=True)
    
    unique_structures = filter_unique_structures(save_trajectory)    
    with open(logfile, 'a') as file1:
        file1.write(f"Found {len(unique_structures)} unique structures\n")

    for j, unique_structure in enumerate(unique_structures):
        print(os.path.join(output_path, name, f"afir_struct_{j}_bondform_{react_symbols}.traj"))
        write(os.path.join(output_path, name, f"afir_struc_{j}_bondform_{react_symbols}.traj"), unique_structure, format="traj")
    
    if return_ase:
        return unique_structures, save_trajectory

def main(traj_dir, output_path, cache_dir, n_chunks=1, chunk_idx=0):
    random.seed(17) # fine for now
    atom_files = []
    for subdir, _, files in os.walk(traj_dir):
        for filename in files:
            if filename.endswith(".traj"):
                path = os.path.join(subdir, filename)
                atom_files.append(path)
   
    random.shuffle(atom_files)
    chunks_to_process = np.array_split(atom_files, n_chunks)
    chunk = chunks_to_process[chunk_idx] 

    logfile = os.path.join(output_path, "logfile.txt")
    with open(logfile, 'a') as file1:
        file1.write(f"length of chunk: {len(chunk)}\n")
        file1.write(f"chunk: {chunk}\n")
    for traj in chunk:
        try:
            ocat_react_pipeline(traj, output_path, cache_dir)
        except Exception as e:
            with open(logfile, 'a') as file1:
                file1.write(f"######## ERROR processing {traj}:\n")
                file1.write(traceback.format_exc() + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", default=".")
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--cache_dir", default=".")
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.traj_dir, args.output_path, args.cache_dir, args.n_chunks, args.chunk_idx)
