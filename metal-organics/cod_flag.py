import glob
import sys
from schrodinger.structure import StructureReader
from schrodinger.structutils.analyze import evaluate_asl
from tqdm import tqdm
import os
import multiprocessing as mp

max_charge ={'Li':1, 'Be':2, 'Na':1, 'Mg':2, 'Al':3, 'K':1, 'Ca':2, 'Sc':3, 'Ti':4, 'V':5, 'Cr':6, 'Mn':7, 'Fe':4, 'Co':4, 'Ni':3, 'Cu':2, 'Zn':2, 'Ga':3, 'Rb':1, 'Sr':2, 'Y':3, 'Zr':4, 'Nb':5, 'Mo':6, 'Tc':7, 'Ru':4, 'Rh':3, 'Pd':4, 'Ag':2, 'Cd':2, 'In':3, 'Sn':4, 'Cs':1, 'Ba':2, 'La':3, 'Ce':4, 'Pr':3, 'Nd':3, 'Pm':3, 'Sm':3, 'Eu':3, 'Gd':3, 'Tb':3, 'Dy':3, 'Ho':3, 'Er':3, 'Tm':3, 'Yb':3, 'Lu':3, 'Hf':4, 'Ta':5, 'W':6, 'Re':7, 'Os':8, 'Ir':5, 'Pt':4, 'Au':3, 'Hg':2, 'Tl':3, 'Pb':4, 'Bi':5}

min_charge ={'Li':1, 'Be':2, 'Na':1, 'Mg':2, 'Al':3, 'K':1, 'Ca':2, 'Sc':3, 'Ti':3, 'V':2, 'Cr':1, 'Mn':1, 'Fe':2, 'Co':1, 'Ni':2, 'Cu':1, 'Zn':2, 'Ga':3, 'Rb':1, 'Sr':2, 'Y':3, 'Zr':3, 'Nb':5, 'Mo':1, 'Tc':1, 'Ru':1, 'Rh':1, 'Pd':2, 'Ag':1, 'Cd':2, 'In':1, 'Sn':2, 'Cs':1, 'Ba':2, 'La':3, 'Ce':3, 'Pr':3, 'Nd':2, 'Pm':3, 'Sm':2, 'Eu':2, 'Gd':3, 'Tb':3, 'Dy':3, 'Ho':3, 'Er':3, 'Tm':3, 'Yb':2, 'Lu':3, 'Hf':4, 'Ta':3, 'W':1, 'Re':1, 'Os':1, 'Ir':1, 'Pt':2, 'Au':1, 'Hg':1, 'Tl':1, 'Pb':2, 'Bi':3}
def has_actinide(st):
    return any(at.atomic_number > 83 for at in st.atom)
def parallel(fname):
    *rest,spin  = os.path.basename(fname).split('_')
    spin, ext = os.path.splitext(spin)
    try:
        st = StructureReader.read(fname)
    except:
        raise
        return
    metals = evaluate_asl(st, 'metals')
    for at in st.atom:
#        if at.element in {'C', 'O', 'N', 'F', 'Cl', 'H'} and at.formal_charge > 1:
#            print(fname, at.element, at.formal_charge)
#            return
        if at.element == 'C' and len([b_at for b_at in at.bonded_atoms if b_at.index not in metals]) > 4:
            print(fname, at.element, at.formal_charge, flush=True)
            return
        elif at.element == 'H' and len([b_at for b_at in at.bonded_atoms if b_at.index not in metals]) > 1 and at.formal_charge >0:
            print(fname, at.element, at.formal_charge, flush=True)
            return
        elif at.element == 'B' and len([b_at for b_at in at.bonded_atoms if b_at.index not in metals]) > 4:
            print(fname, at.element, at.formal_charge, flush=True)
            return
    for metal in metals:
        elt = st.atom[metal].element
        charge = st.atom[metal].formal_charge
        if charge == 0:
            return
        if charge < min_charge[elt] or charge > max_charge[elt]:
            print(fname, elt, charge, flush=True)

with open('cod_complexes.txt', 'r') as fh:
    lst = [f.strip() for f in fh.readlines()]
with mp.Pool(60) as pool:
    list(tqdm(pool.imap(parallel, lst), total=len(lst)))
