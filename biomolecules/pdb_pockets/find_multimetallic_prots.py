import os
import re
import glob
import multiprocessing as mp
from tqdm import tqdm
from schrodinger.structure import StructureReader
from schrodinger.structutils.analyze import evaluate_asl
from cleanup import TM_LIST

TMS = f'atom.atomicnum {",".join([str(i) for i in TM_LIST])}'

def parallel_work(fname):
    st = StructureReader.read(fname)
    tms = evaluate_asl(st, TMS)
    if len(tms) > 1:
        return fname
#    if st.atom_total < 130:
#        return fname

def main():
    flist = glob.glob('/checkpoint/levineds/pdb_tm_ood/*.mae')
    with mp.Pool(60) as pool:
        coords = set(tqdm(pool.imap(parallel_work, flist), total=len(flist)))
    coords -= {None}
    with open('pdb_clusters.txt', 'w') as fh:
        fh.writelines([f+'\n' for f in coords])

if __name__ == '__main__':
    main()
