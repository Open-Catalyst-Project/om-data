import argparse
import json
import os
import random
from collections import Counter

import pandas as pd
from schrodinger.adapter import to_structure
from schrodinger.structutils.analyze import rotatable_bonds_iterator
from schrodinger.application.jaguar.packages.rotation_barriers_modules.pes_fitting import constrained_dihedral_ff_minimization
from schrodinger.structutils import minimize

SCHRO = "/private/home/levineds/schrodinger2024-2"
random.seed(432)

def load_zinc():
    with open("zinc_2783k.json", "r") as fh:
        data = json.loads(fh.read())
    zinc_df = pd.DataFrame(data)
    return zinc_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--pkl_path", default=".")
    return parser.parse_args()


def get_ligand(smi):
    st = to_structure(smi)
    st.generate3dConformation(require_stereo=False)
    return st

def scan_random_torsion(st, rotatable_bonds):
    bond = random.choice(rotatable_bonds)
    at1 = min(b_at.index for b_at in st.atom[bond[0]].bonded_atoms if b_at.index not in bond)
    at4 = min(b_at.index for b_at in st.atom[bond[1]].bonded_atoms if b_at.index not in bond)
    tors = [at1, bond[0], bond[1], at4]
    st_list = []
    minimizer = minimize.Minimizer(struct=st.copy())
    # Pre-minimize without constraint
    minimizer.minimize()

    for ang in range(0, 360, 10):
        st_copy, _ = constrained_dihedral_ff_minimization(st, tors, ang, minimizer)
        st_copy.title = f'{"_".join(str(i-1) for i in tors)}={ang}'
        st_list.append(st_copy)
    return st_list

def main(args):
    zinc_df = load_zinc()
    counter = Counter()
    for idx, zinc_entry in zinc_df.iterrows():
        ligand_fname = zinc_entry["zincid"]
        st = get_ligand(zinc_entry["SMILES"])
        rotatable_bonds = list(rotatable_bonds_iterator(st))
        n_rot = len(rotatable_bonds)
        print(n_rot)
        st_list = scan_random_torsion(st, rotatable_bonds)
        for tors_st, ang in zip(st_list, range(0, 360, 10)):
            tors_st.write(os.path.join(args.output_path, f'{ligand_fname}_{ang}_{st.formal_charge}_1.xyz'))
        counter[n_rot] += 1
        if sum(counter.values()) >= 100:
            print(counter)
            break

if __name__ == "__main__":
    args = parse_args()
    main(args)
