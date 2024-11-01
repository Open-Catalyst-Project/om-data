import argparse
import csv
import glob
import os
import multiprocessing as mp
from collections import Counter
from functools import partial
from tqdm import tqdm

from schrodinger.application.jaguar.autots_bonding import clean_st
from schrodinger.application.jaguar.packages.shared import read_cartesians

ION_SPIN = {
    "Ag+2": 1,
    "Co+2": 3,
    "Cr+2": 4,
    "Cr+3": 3,
    "Cu+2": 1,
    "Fe+2": 4,
    "Fe+3": 5,
    "Mn+2": 5,
    "Ni+2": 2,
    "Pd+2": 2,
    "Pt+2": 2,
    "Ti+": 3,
    "OV+2": 1,
    "V+2": 3,
    "V+3": 2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


def count_elements(fname):
    with open(fname, "r") as fh:
        data = fh.readlines()[2:]
    elts = Counter(line.strip().split(" ")[0] for line in data)
    return elts

def rename_tempo(fname):
    carts = read_cartesians(fname)[0]
    st = carts.getStructure()
    st = clean_st(st)
    bonds_to_del = []
    for bond in st.bond:
        if bond.order == 0:
            bonds_to_del.append(bond.atom)
    for bond in bonds_to_del:
        st.deleteBond(*bond)
    cnt = Counter(
        frozenset(Counter(at.atomic_number for at in mol.atom).items())
        for mol in st.molecule
    )
    n_tempo = cnt[frozenset(((1, 18), (6, 9), (7, 1), (8, 1)))]
    new_name = fname.replace("_1.xyz", f"_{n_tempo+1}.xyz")
    os.rename(fname, new_name)

def rename_job(spec, path):
    job_idx, system = spec
    ion_contents = {k.split("+")[0].replace('O',''): v for k, v in ION_SPIN.items() if k in system[5:]}
    for fname in glob.glob(f"{os.path.join(path, str(job_idx))}*"):
        elt_counts = count_elements(fname)
        curr_spin = int(os.path.splitext(os.path.basename(fname))[0].split("_")[-1]) - 1
        n_unpaired_spins = curr_spin
        for elt, count in elt_counts.items():
            n_unpaired_spins += ion_contents.get(elt, 0) * count
        if curr_spin != n_unpaired_spins:
            new_fname = fname.replace(
                f"_{curr_spin + 1}.xyz", f"_{n_unpaired_spins + 1}.xyz"
            )
            print(fname, new_fname)
            os.rename(fname, new_fname)

def main():
    args = parse_args()
    pool = mp.Pool(60)

    # Do TEMPO jobs
    for i in [659, 767, 3681, 3682, 3683, 3684]:
        lst = glob.glob(f"{os.path.join(args.output_path, str(i))}_*xyz")
        list(tqdm(pool.imap(rename_tempo, lst), total=len(lst)))

    # Do metal jobs
    with open("elytes.csv", "r") as f:
        systems = list(csv.reader(f))[1:]
    rename_fxn = partial(rename_job, path=args.output_path)
    vo2_list = [515,655,656,657,658,667,668,669,670,715,716,717,718,815,816,915,916,917,918,939,940,941,942,1033,1034,1035,1036,1215,1216,1217,1218,1265,1266,1267,1268,1389,1390,1391,1392,1409,1410,1411,1412,1421,1422,1423,1424,1449,1450,1451,1452,1485,1486,1487,1488,1523,1524,1525,1526,1561,1562,1563,1564,1609,1610,1611,1612,1709,1710,1711,1712,1725,1726,1727,1728,1741,1742,1743,1744,1841,1842,1843,1844,1967,1968,1969,1970,2191,2192,2193,2194,2459,2460,2461,2462,2563,2564,2565,2566,2759,2760,2761,2762,2771,2772,2773,2774,2803,2804,3137,3138,3139,3140,3377,3378,3379,3380,3397,3398,3399,3400,3409,3410,3411,3412,3429,3430,3431,3432,3517,3518,3519,3520,3553,3554,3555,3556,3649,3650,3651,3652,3681,3682,3683,3684,3709,3710,3711,3712]
    lst = [val for val in enumerate(systems, start=1) if val[0] in vo2_list] 
    list(tqdm(pool.imap(rename_fxn, lst), total=len(lst)))


if __name__ == "__main__":
    main()
