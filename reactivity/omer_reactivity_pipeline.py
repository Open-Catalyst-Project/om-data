import sys
import os
import random
import numpy as np
import json
import argparse
import signal

from omdata.reactivity_utils import filter_unique_structures, run_afir
from ase.io import write
from omer_utils import get_chain_path_info, trim_structures, get_bond_smarts, add_h_to_chain, remove_h_from_chain
from io_chain import Chain, get_bonds_to_break

import torch
from torch.serialization import add_safe_globals
add_safe_globals([slice])

from fairchem.core import pretrained_mlip, FAIRChemCalculator

smarts_dict = { "none": 0,
                "[#7D3].[#7D3;+1][#1]": 1,
                "[#7H2].[#7H2;+1][#1]": 2,
                "[#7H1].[#7H1;+1][#1]": 3,
                "[#8D1].[#8D1;+1][#1]": 4,
                "[#8H1].[#8H1;+1][#1]": 5,
                "[#6;R0]=[#6;R0].[#6;+1]([#1])[#6]": 6,
                "[#7]#[#6].[#7;+1]([#1])#[#6]": 7, 
                "[#7D2]=[#6].[#7D2;+1]([#1])=[#6]": 8,
                "[#1][#6;R0][!#1].[#6;+1][!#1]":9,
                "[#1][#8].[#8;-1]":10,
                "[#1][#7]([!#1])[!#1].[#7;-1]([!#1])[!#1]":11,
                "[#1][#7]([#1])[!#1].[#7;-1]([#1])[!#1]": 12, 
                "[#1][#16][#16;-1]": 13, 
                "[#1][#6]#[#6].[#6;-1]#[#6]": 14,
                "[#1][#6][#6](=[#8])[#6].[#6]=[#6]([#8;-1])[#6]": 15
}

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

def get_splits_for_protonation(pdb_files, csv_dir, logfile):
    all_add_remove, all_add, all_remove, all_none = [], [], [], []
    too_far, time_out = 0, 0
    for pdb_path in pdb_files:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(3)
        try: 
            # get all valid H mutations of chain
            parent_chain, add_chain, remove_chain, bonds, updated_bonds = process_one_pdb(pdb_path, csv_dir)
            if add_chain != parent_chain:
                if remove_chain != parent_chain:
                    all_add_remove.append({"parent_chain": parent_chain, "add_chain": add_chain, "remove_chain": remove_chain,
                                            "a_bond_to_break": bonds, "r_bond_to_break": updated_bonds, "path": pdb_path})
                else:
                    all_add.append({"parent_chain": parent_chain, "add_chain": add_chain, "remove_chain": None,
                                            "a_bond_to_break": bonds, "r_bond_to_break": None, "path": pdb_path})
            elif remove_chain != parent_chain:
                all_remove.append({"parent_chain": parent_chain, "add_chain": None, "remove_chain": remove_chain,
                                            "a_bond_to_break": bonds, "r_bond_to_break": updated_bonds, "path": pdb_path})
            else:
                all_none.append({"parent_chain": parent_chain, "add_chain": None, "remove_chain": None,
                                            "a_bond_to_break": bonds, "r_bond_to_break": None, "path": pdb_path})
        except (TimeoutException, IndexError) as e:
            if type(e) == IndexError: # No bonds near center of mass
                too_far += 1
            else: # Error processing pdb to Chain
                time_out += 1
            continue
        finally:
            signal.alarm(0)

    with open(logfile, 'a') as file:
        file.write(f"Both: {len(all_add_remove)}, Added: {len(all_add)}, Removed: {len(all_remove)}, None: {len(all_none)}\n")
        file.write(f"Total failed: {too_far + time_out}, No bond breaking: {too_far}, Timeout: {time_out}\n")

    total = len(all_add_remove) + len(all_add) + len(all_remove) + len(all_none)
    add_goal = round(1/3 * total)
    remove_goal = round(1/3 * total)
    none_goal = total - add_goal - remove_goal

    # shuffle all lists before splitting between H mutation schemes
    random.shuffle(all_add_remove), random.shuffle(all_add), random.shuffle(all_remove), random.shuffle(all_none)

    add_list, remove_list, none_list = [], [], []
    used_remove, used_both = set(), set()

    for none_dict in all_none: # fill with chaings where nothing can happen
        if none_goal <= 0:
            break
        none_list.append({"chain": none_dict["parent_chain"], "bond_to_break": none_dict["a_bond_to_break"], "path": none_dict["path"]})
        none_goal -= 1
    for add_dict in all_add:   # fill with chaings where +H is only option
        if add_goal <= 0:
            break
        add_list.append({"chain": add_dict["add_chain"], "bond_to_break": add_dict["a_bond_to_break"], "path": add_dict["path"]})
        add_goal -= 1
    for i, both_dict in enumerate(all_add_remove):
        if add_goal > 0:       # fill with chains where +H is possible
            add_list.append({"chain": both_dict["add_chain"], "bond_to_break": both_dict["a_bond_to_break"], "path": both_dict["path"]})
            add_goal -= 1
            used_both.add(i)
        elif remove_goal > 0:  # fill with chaings where -H is possible
            if both_dict["remove_chain"].ase_atoms.info["mod_smarts"] == "[#1][#6;R0][!#1].[#6;+1][!#1]":
                continue       # fill with everything but carbocations
            remove_list.append({"chain": both_dict["remove_chain"],"bond_to_break": both_dict["r_bond_to_break"], "path": both_dict["path"]})
            remove_goal -= 1
            used_both.add(i)

    for i, remove_dict in enumerate(all_remove): # fill where -H is possible, except carbocations
        if remove_dict["remove_chain"].ase_atoms.info["mod_smarts"] == "[#1][#6;R0][!#1].[#6;+1][!#1]":
            continue
        if remove_goal <= 0:
            break
        remove_list.append({"chain": remove_dict["remove_chain"],"bond_to_break": remove_dict["r_bond_to_break"], "path": remove_dict["path"]})
        remove_goal -= 1
        used_remove.add(i)
    if remove_goal > 0:
        for i, both_dict in enumerate(all_add_remove): # allow carbocations where +/- H possible
            if i in used_both:
                continue
            else:
                remove_list.append({"chain": remove_dict["remove_chain"],"bond_to_break": remove_dict["r_bond_to_break"], "path": remove_dict["path"]})
                remove_goal -= 1
                used_remove.add(i)
            if remove_goal <=0:
                break
    for i, remove_dict in enumerate(all_remove): # fill remaining as needed to none
        if none_goal <= 0:
            break
        if i in used_remove:
            continue
        none_list.append({ "chain": remove_dict["parent_chain"],"bond_to_break": remove_dict["a_bond_to_break"], "path": remove_dict["path"]})
        used_remove.add(i)
        none_goal -= 1

    for i, both_dict in enumerate(all_add_remove): # fill remaining as needed to none
        if none_goal <= 0:
            break
        if i in used_both:
            continue
        none_list.append({"chain": both_dict["parent_chain"], "bond_to_break": both_dict["a_bond_to_break"], "path": both_dict["path"]})
        used_both.add(i)
        none_goal -= 1
    
    with open(logfile, 'a') as file:
        file.write(f"Remaining: add_goal={add_goal}, remove_goal={remove_goal}, none_goal={none_goal}\n")
        file.write(f"Final counts: add={len(add_list)}, remove={len(remove_list)}, none={len(none_list)}\n")

    return add_list, remove_list, none_list

def process_one_pdb(pdb_path, csv_dir):
    repeat_smiles, _ = get_chain_path_info(pdb_path, csv_dir)
    chain = Chain(pdb_path, repeat_smiles)
    all_bonds_to_break = get_bonds_to_break(chain, max_H_bonds=1, max_other_bonds=4)
    bond_to_break = random.choice(all_bonds_to_break)

    try:
        new_h_chain = add_h_to_chain(chain, bond_to_break)
    except Exception:
        new_h_chain = chain

    try:
        new_r_chain, new_bonds = remove_h_from_chain(chain, bond_to_break)
    except Exception:
        new_r_chain, new_bonds = chain, None

    return chain, new_h_chain, new_r_chain, bond_to_break, new_bonds

def omer_react_pipeline(chain_dict, output_path, csv_dir):
    chain_path = chain_dict['path']
    chain = chain_dict['chain']
    bond_to_break = chain_dict['bond_to_break']
    _, polymer_class = get_chain_path_info(chain_path, csv_dir)

    name = polymer_class + "_" + chain_path.split("/")[-1][:-4]

    os.makedirs(os.path.join(output_path, name), exist_ok=True)
    logfile = os.path.join(output_path, name, "logfile.txt")

    add_electron = random.choices([-1, 0, 1], weights=[0.1, 0.8, 0.1], k=1)[0]
    charge = chain.ase_atoms.info.get("charge", 0) + add_electron
    uhf = chain.ase_atoms.info.get("spin", 1) - 1
    uhf += 1 if add_electron != 0 and uhf % 2 == 0 else -1 if add_electron != 0 else 0
    uhf = max(0, uhf)
    chain.ase_atoms.info["spin"] = uhf + 1
    chain.ase_atoms.info["charge"] = charge

    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda", inference_settings="turbo")
    UMA = FAIRChemCalculator(predictor, task_name="omol")
    save_trajectory, _ = run_afir(chain, None, UMA, logfile,
                                    bonds_breaking=[bond_to_break], maxforce=10.0, force_step=0.75, is_polymer=True)
    
    unique_structures = filter_unique_structures(save_trajectory)    
    with open(logfile, 'a') as file1:
        file1.write(f"Found {len(unique_structures)} unique structures\n")

    trimmed_structures = trim_structures(chain, unique_structures, bond_to_break)
    
    for i, atoms in enumerate(trimmed_structures):
        react_symbols = get_bond_smarts(chain.rdkit_mol, bond_to_break[0], bond_to_break[1]).replace("[", "br").replace("]", "").replace("(", "p").replace(")", "").replace("-","_to_")
        n_atoms = len(atoms)
        if "mod_smarts" in atoms.info:
            string = atoms.info["mod_smarts"]
            mod_num = smarts_dict[string]
        else:
            mod_num = 0
        cutoff = np.round(atoms.info['trim_cutoff'], 2)
        comment = json.dumps(atoms.info)

        print(os.path.join(output_path, name, f"afir_struct_{i}_charge_{charge}_uhf_{uhf}_natoms_{n_atoms}_bondbreak_{react_symbols}_modsmarts_{mod_num}_cutoff_{cutoff}_{charge}_{uhf+1}.xyz"))
        write(os.path.join(output_path, name, f"afir_struc_{i}_charge_{charge}_uhf_{uhf}_natoms{n_atoms}_bondbreak_{react_symbols}_modsmarts_{mod_num}_cuttoff_{cutoff}_{charge}_{uhf+1}.xyz"), atoms, format="xyz", comment=comment)

def main(args):
    pdb_files = []
    for subdir, _, files in os.walk(args.all_chains_dir):
        for filename in files:
            if filename.endswith(".pdb"):
                pdb_path = os.path.join(subdir, filename)
                pdb_files.append(pdb_path)
   
    chunks_to_process = np.array_split(pdb_files, args.n_chunks)
    chunk = chunks_to_process[args.chunk_idx] 
    print('length of chunk', len(chunk))
    print(chunk)
    add_list, remove_list, none_list = get_splits_for_protonation(chunk, args.csv_dir, "logfile.txt")

    os.makedirs(os.path.join(args.output_path, 'none/'), exist_ok=True)
    none_path = os.path.join(args.output_path, 'none/')
    for none_chain_dict in none_list:
        omer_react_pipeline(none_chain_dict, none_path, args.csv_dir)
    
    os.makedirs(os.path.join(args.output_path, 'remove_H/'), exist_ok=True)
    remove_path = os.path.join(args.output_path, 'remove_H/')
    for remove_chain_dict in remove_list:
        omer_react_pipeline(remove_chain_dict, remove_path, args.csv_dir)

    os.makedirs(os.path.join(args.output_path, 'add_H/'), exist_ok=True)
    add_path = os.path.join(args.output_path, 'add_H/')
    for add_chain_dict in add_list:
        omer_react_pipeline(add_chain_dict, add_path, args.csv_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_chains_dir", default=".")
    parser.add_argument("--csv_dir", default=".")
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
