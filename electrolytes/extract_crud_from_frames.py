import os
import random
import sys
from tqdm import tqdm
import multiprocessing as mp
from schrodinger.structure import StructureReader
from schrodinger.structutils.analyze import evaluate_asl


st_list = list(StructureReader(os.path.join(sys.argv[1], 'frames.maegz')))
os.makedirs(os.path.join(sys.argv[1], 'extracts'), exist_ok=True)
basename = os.path.basename(os.path.dirname(sys.argv[1]))
def remove_proper_subsets(frozen_sets):
    """
    Removes frozensets that are proper subsets of another frozenset in the set.

    Args:
        frozen_sets (set of frozensets): The input set of frozensets.

    Returns:
        set of frozensets: The filtered set with proper subsets removed.
    """
    # Sort the frozensets by their size in descending order
    sorted_sets = sorted(frozen_sets, key=len, reverse=True)
    
    result = set()
    for s in sorted_sets:
        if not any(s.issubset(other) and s != other for other in result):
            result.add(s)
    
    return result

def sulfur_to_se(st):
    S_idx = evaluate_asl(st, f'smarts. S=P and at.ele S')
    if S_idx:
        st_copy = st.copy()
        for at in S_idx:
            st_copy.atom[at].element = 'Se'
        return st_copy

def parallel_work(data):
    st_idx, st = data
    extracts = set()
    core_mols = random.sample(range(1, st.mol_total+1), 100)
    for mol_idx in core_mols:
        at_idxs = evaluate_asl(st, f'fillres (within 3 mol.num {mol_idx})')
        if len(at_idxs) == 1:
            continue
        extracts.add(frozenset(at_idxs))
    extracts = remove_proper_subsets(extracts)
    for idx, at_idxs in enumerate(extracts):
        cluster = st.extract(at_idxs)
        #O2_count = evaluate_asl(cluster, f'smarts. O=O')
        #spin = 3 if ((len(O2_count)//2) % 2 == 1) else 1
        spin = 1
        charge = cluster.formal_charge
        cluster.write(os.path.join(sys.argv[1], 'extracts', f'{basename}_{st_idx}_{idx}_{charge}_{spin}.mae'))
        cluster_sub = sulfur_to_se(cluster)
        if cluster_sub is not None:
            cluster_sub.write(os.path.join(sys.argv[1], 'extracts', f'{basename}_{st_idx}_{idx}b_{charge}_{spin}.mae'))

with mp.Pool(60) as pool:
    list(tqdm(pool.imap(parallel_work, enumerate(st_list)), total=len(st_list)))
