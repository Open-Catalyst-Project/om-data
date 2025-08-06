"""
Take COD complexes with total atoms less than 150, charge between +2 and -2
"""
import os
import glob
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.structure import StructureReader
from tqdm import tqdm
import multiprocessing as mp
import mendeleev

radius_dict = {i: mendeleev.element(i).covalent_radius_pyykko for i in range(1,84)}

def parallel_work(fname):
    if os.path.basename(fname).split('_')[0] in HCMG:
        return
    try:
        st = StructureReader.read(fname)
    except:
        print(fname)
        return
    if len(evaluate_asl(st, 'metals')) == 1 and abs(st.formal_charge) < 3 and st.atom_total <= 250 and not has_actinide(st) and not has_collisions(st):
        return fname


def has_actinide(st):
    return any(at.atomic_number > 83 for at in st.atom)

def get_expected_length(at1, at2):
    total = 0
    for at in (at1, at2):
        radius = radius_dict[at.atomic_number]
        total += radius
    return total / 100.0
    
def has_collisions(st):
    return any(bond.length < 0.55 * get_expected_length(*bond.atom) for bond in st.bond)

output_path = '/checkpoint/levineds/crystallographic_open_database/addl_st/'
with open('high_charge_main_group.txt', 'r') as fh:
    HCMG = {f.strip() for f in fh.readlines()}

fname_list = glob.glob(os.path.join(output_path,'*.mae'))
#kill_set = {os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(output_path,'*molecule_1*.mae'))}
#fname_list = [f for f in fname_list if os.path.basename(f).split('_')[0] not in kill_set]
with mp.Pool(60) as pool:
    good_points = set(tqdm(pool.imap(parallel_work, fname_list), total=len(fname_list)))
good_points -= {None}
with open('cod_complexes.txt', 'w') as fh:
    fh.writelines([f+'\n' for f in good_points])

