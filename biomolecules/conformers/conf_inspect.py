import argparse
import contextlib
import glob
import json
import multiprocessing as mp
import os
import tempfile
from collections import defaultdict

import numpy as np
from ase import Atoms
from schrodinger.application.jaguar.autots_bonding import clean_st, copy_bonding
from schrodinger.application.jaguar.file_logger import FileLogger
from schrodinger.application.jaguar.packages.csrch import eliminate_duplicate_conformers
from schrodinger.application.matsci.aseutils import get_structure
from schrodinger.structure import StructureWriter
from tqdm import tqdm
import re


@contextlib.contextmanager
def chdir(destination):
    # Store the current working directory
    current_dir = os.getcwd()

    try:
        # Change to the new directory
        os.chdir(destination)
        yield
    finally:
        # Change back to the original directory
        os.chdir(current_dir)


def build_st(data):
    atomic_numbers = data["atom_numbers"]
    # coords are saved as [1, N, 3]
    atomic_positions = np.array(data["coords [A]"]).reshape(-1, 3)
    total_energy = data["total_energy [Eh]"]
    atoms = Atoms(atomic_numbers, atomic_positions)
    st = get_structure(atoms)
    st.property["r_j_Gas_Phase_Energy"] = total_energy
    return st


def group_by_system(conformer_jsons):
    grouped_confs = defaultdict(list)
    for json_name in conformer_jsons:
        identifier = os.path.basename(re.search(r"/.*?state\d", json_name).group(0))
        grouped_confs[identifier].append(json_name)
    return grouped_confs


def get_conformers(conformer_list):
    st_list = []
    for json_file in conformer_list:
        with open(json_file, "r") as fh:
            data = json.load(fh)
        st = build_st(data)
        st.title = json_file
        st_list.append(st)
    st_list[0] = clean_st(st_list[0])
    for st in st_list[1:]:
        copy_bonding(st_list[0], st)
    return st_list


def parallel_work(conformer_list):
    st_list = get_conformers(conformer_list)
    print("initial", len(st_list))

    with tempfile.TemporaryDirectory() as temp_dir:
        with chdir(temp_dir), FileLogger("csrch", False):
            final_list = eliminate_duplicate_conformers(st_list)
    print("final", len(final_list))
    # Keep at most the 100 lowest energy ones
    final_list.sort(key=lambda x: x.property['r_j_Gas_Phase_Energy'])
    return [st.title for st in final_list[:100]]


def main():
    #conformer_jsons = glob.glob(
    #    "/large_experiments/opencatalyst/foundation_models/data/omol/evals/processed/ligand_strain_25041*/*/step*.json"
    #)
    #conformer_jsons = [f for f in conformer_jsons if not f.endswith("step0.json")]
    #with open('conf_list','w') as fh:
    #    fh.write(str(conformer_jsons))

    with open(
        "/large_experiments/opencatalyst/foundation_models/data/omol/evals/ligand_strain/ligand_strain_conformers_to_keep_max100", 
'r') as fh:
        conformer_jsons = [f.strip() for f in fh.readlines()]
    grouped_confs = group_by_system(conformer_jsons)
    for key,group in grouped_confs.items():
        if len(group) < 40:
            continue
        st_list = get_conformers(group)
        with StructureWriter(key + '.mae') as writer:
            writer.extend(st_list)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    main()
