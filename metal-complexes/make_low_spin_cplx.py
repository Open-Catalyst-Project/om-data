import glob
import os
from ase.io import read
from tqdm import tqdm
import multiprocessing as mp

TM_LIST = {*range(39, 49), *range(72, 81)}
if 1:
    TM_LIST.update(range(21, 31))
flist = glob.glob('/large_experiments/opencatalyst/foundation_models/data/omol/metal_organics/inputs_06142024/xyzs/*.xyz')
#flist = glob.glob('/large_experiments/opencatalyst/foundation_models/data/omol/metal_organics/outputs/xyzs/*.xyz')
#flist = glob.glob('/checkpoint/levineds/arch_H/inputs/xyzs/*.xyz')
flist = [f for f in flist if not (f.endswith('_1.xyz') or f.endswith('_2.xyz') or f.endswith('_3.xyz') or f.endswith('_4.xyz'))]

def parallel(f):
    atoms = read(f)
    if any(at in TM_LIST for at in atoms.get_atomic_numbers()) and len(atoms.get_atomic_numbers()) <= 130:
        print(f, flush=True)
with mp.Pool(60) as pool:
    list(tqdm(pool.imap(parallel, flist), total=len(flist)))
with open('ls_23.txt', 'w') as fh:
    fh.writelines([k+ '\n' for k in keep])
