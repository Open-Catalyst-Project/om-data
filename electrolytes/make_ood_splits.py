import csv
import glob
import multiprocessing as mp
import os
from functools import partial
from typing import Set

from schrodinger.structure import Structure, StructureReader
from tqdm import tqdm

from solvation_shell_utils import get_species_from_res

#with open('elytes.csv', 'r') as fh:
#   cls_elytes = list(csv.reader(fh))
#
#with open('ml_elytes.csv', 'r') as fh:
#   ml_elytes = list(csv.reader(fh))
#
#lines_with_ood = [line for line in cls_elytes + ml_elytes if 'C4H6F3O2-' in line]
#print(cls_elytes.index(lines_with_ood[0]))
#print(len(cls_elytes + ml_elytes) - 2 )

DATA_DIR = "/checkpoint/levineds/elytes_09_29_2024/results2"


def system_has_fragments(st: Structure, frags: Set[str]):
    """
    Check if the given system has a given molecular fragment

    :param st: Structure to inspec
    :param frag: stoichiometry string
    """
    for mol in st.residue:
        if get_species_from_res(mol) in frags:
            frag_present = True
            break
    else:
        frag_present = False
    return frag_present


def parallel_work(fname, ood_frags):
    job_num, species, radius, *rest, charge, spin = os.path.splitext(
        os.path.basename(fname)
    )[0].split("_")

    mae_name = os.path.join(
        DATA_DIR,
        job_num,
        species,
        f"radius_{radius}",
        "_".join(rest + [charge, spin]) + ".mae",
    )
    if os.path.exists(mae_name):
        st = StructureReader.read(mae_name)
    else:
        glob_pat = glob.glob(mae_name.replace(f'_{spin}.mae', '_*.mae'))
        if glob_pat:
            st = StructureReader.read(glob_pat[0])
        else:
            print(f'problem with {fname}')
            return None
    if system_has_fragments(st, ood_frags):
        return fname
    else:
        return ""


def main():
    with open("archives_md_elytes_3A.txt", "r") as fh:
        data = [
            os.path.dirname(os.path.dirname(f)) for f in fh.readlines() if "step0" in f
        ]

    ood_frags = {"C4F3H6O2-1"}
    pool = mp.Pool(60)
    fxn = partial(parallel_work, ood_frags=ood_frags)
    ood_list = set(tqdm(pool.imap(fxn, data), total=len(data)))
    with open("ood_systems.txt", "w") as fh:
        fh.write(str(ood_list))


if __name__ == "__main__":
    main()
