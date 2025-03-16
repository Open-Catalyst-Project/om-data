import argparse
import glob
import os
import random
from typing import List, Tuple

from collections import defaultdict
import sys
from schrodinger.structure import Residue, Structure
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.structutils import build

import biolip_extraction as blp_ext
import protein_core_extraction as prot_core

MAX_ATOMS = 350


def has_protein(st, at_list):
    prot_atoms = evaluate_asl(st, 'protein')
    return bool(set(prot_atoms).intersection(at_list))

def has_nucleic_acid(st, at_list):
    na_atoms = evaluate_asl(st, 'dna or rna')
    return bool(set(na_atoms).intersection(at_list))

def append_O3prime(st, system):
    P_atoms = evaluate_asl(st, f'atom.num {",".join([str(i) for i in system[1]])} and at.ele P')
    for at in P_atoms:
        for b_at in st.atom[at].bonded_atoms:
            if b_at.index not in system[1]:
                system[1].append(b_at.index)
                system[1].sort()

def stringify(at_list):
    return ",".join(str(i) for i in at_list)

def main(output_path, start_pdb=0, end_pdb=1):
    """
    For each DNA entry in biolip, extract from around a residue (a new random one each time)
        1) within 3A and different chain (S--) fillres (within 2.5 res.num 5015) and not (chain B and dna) or res.num 5015
        2) within 3A and same chain not protein (E) fillres within 2.5 res.num 4003) and not protein
        3) within 3A and protein (S-) nvm fillres (within 2.5 res.num 5005) and not dna or res.num 5005
        4) including res+1 or -1, within 3A and different chain and not protein (==) fillres (within 2.5 res.num 4019,4020) and not (protein or chain A and dna)
        5) within 3A and not protein (E-) ??? fillres (within 2.5 res.num 5007) and not protein
    """
    biolip_df = blp_ext.get_biolip_db("macromol")
    biolip_df = biolip_df[biolip_df['ligand_id'] == 'dna']
    
    grouped_biolip = biolip_df.groupby("pdb_id")
    pdb_list = list(grouped_biolip.groups.keys())
    done_list = {
        tuple(os.path.basename(f).split("_")[:2])
        for f in glob.glob(os.path.join(output_path, "*.mae"))
    }
    for pdb_count in range(start_pdb, min(end_pdb, len(pdb_list))):
        pdb_id = pdb_list[pdb_count]
        st = blp_ext.download_cif(pdb_id)
        try:
            st = prot_core.prepwizard_core(st, pdb_id)
        except RuntimeError:
            continue
        rows = grouped_biolip.get_group(pdb_id)
        print(f"preparing {pdb_id} (entry: {pdb_count})", flush=True)
        chains = set()
        for idx, row in rows.iterrows():
            chains.add((row['ligand_chain'], (row['ligand_residue_number'], row['ligand_residue_end'] + 1)))
            sys.stdout.flush()
        for chain in chains:
            res_list = list(range(*chain[1]))
            random.shuffle(res_list)
            system_types = defaultdict(list)
            for seed_res in res_list:
                at_list = evaluate_asl(st, f'chain {chain[0]} and res.num {seed_res}')
                if not at_list:
                    continue
                if len(system_types['S--']) < 3:
                    addl_atoms = evaluate_asl(st, f'fillres (within 2.5 at.num {stringify(at_list)}) and not (chain {chain[0]} and (dna or rna))')
                    system_types['S--'].append((seed_res, at_list+addl_atoms))
                elif len(system_types['S-']) < 2:
                    addl_atoms = evaluate_asl(st, f'fillres (within 2.5 at.num {stringify(at_list)}) and not (dna or rna)')
                    system_types['S-'].append((seed_res, at_list+addl_atoms))
                elif not system_types['E']:
                    addl_atoms = evaluate_asl(st, f'fillres (within 2.5 at.num {stringify(at_list)}) and not protein and chain {chain[0]}')
                    system_types['E'].append((seed_res, at_list+addl_atoms))
                elif not system_types['--']:
                    addl_atoms = evaluate_asl(st, f'fillres (within 2.5 at.num {stringify(at_list)}) and not (chain {chain[0]} and (dna or rna)) and (not protein)')
                    system_types['--'].append((seed_res, at_list+addl_atoms))
                elif not system_types['=='] and not system_types['E-']:
                    if random.random() < 0.5:
                        addl_atoms = evaluate_asl(st, f'fillres (within 2.5 at.num {stringify(at_list)}) and not protein')
                        system_types['E-'].append((seed_res, at_list+addl_atoms))
                        break
                    else:
                        if seed_res == chain[1][1] - 1:
                            seed_res -= 1
                        at_list = evaluate_asl(st, f'chain {chain[0]} and res.num {seed_res},{seed_res+1}')
                        addl_atoms = evaluate_asl(st, f'fillres (within 2.5 at.num {stringify(at_list)}) and not protein')
                        system_types['=='].append((seed_res, at_list+addl_atoms))
                        break
            for sys_class, system_list in system_types.items():
                for system in system_list:
                    st_copy = st.copy()
                    append_O3prime(st_copy, system)
                    protein_atoms = evaluate_asl(st_copy, f'atom.num {stringify(system[1])} and protein')
                    if protein_atoms:
                        res_list = list({st_copy.atom[at].getResidue() for at in protein_atoms})
                        gap_res = prot_core.get_single_gap_residues(st_copy, res_list)
                        try:
                            st_copy = blp_ext.make_gaps_gly(st_copy, None, gap_res)
                        except:
                            continue

                        for res in gap_res:
                            system[1].extend(res.getAtomIndices())

                    # Try to label ligands
                    for at in evaluate_asl(st_copy, "ligand"):
                        st_copy.atom[at].chain = "l"
                    na_st = st_copy.extract(system[1])

                    build.add_hydrogens(na_st, atom_list=evaluate_asl(na_st, 'atom.ptype " O3\'"'))
                    try:
                        blp_ext.cap_termini(st_copy, na_st, remove_lig_caps=True)
                    except Exception as e:
                        print("Error: Cannot cap termini")
                        print(e)
                        continue
                    na_st = build.reorder_protein_atoms_by_sequence(na_st)
                    fname = f'{pdb_id}_{chain[0]}{system[0]}_{sys_class}_{na_st.formal_charge}_1.mae'
                    na_st.write(os.path.join(output_path, fname))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--seed", type=int, default=4621)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    main(args.output_path, args.start_idx, args.end_idx)
