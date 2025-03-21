"""
Script for sampling from extracted clusters
"""

import argparse
from functools import partial
import glob
import os
import random
import shutil
import multiprocessing as mp
from tqdm import tqdm
from schrodinger.structure import StructureReader
from collections import defaultdict

TOTAL_PER_SYSTEM = 10000
RADIUS_LIST = [3]
TM_LIST = {*range(21, 31), *range(39, 49), *range(72, 81)}
MAX_TMS = 3
# radius 5 seed 4353
# radius 7 seed 353
# radius 3 ml-md seed 628
# radius 5 ml-md seed 9173, TOTAL 25
# radius 3 rpmd seed 631, TOTAL 10000
random.seed(631)

def filter_heavy_atoms(species_glob):
    species_list = []
    grouped_dict = defaultdict(list)
    for species in species_glob:
        if 'group' not in species:
            grouped_dict[species].append(species)
        else:
            group = os.path.basename(species).split('_')[1]
            grouped_dict[group].append(species)
    for group_list in grouped_dict.values():
        st = StructureReader.read(group_list[0])
        if sum(1 for at in st.atom if at.atomic_number in TM_LIST) < MAX_TMS:
            species_list.extend(group_list)
    return species_list
        

def sample_clusters(res_dir, favor_ions):
    def sample_to_dict(species_sample):
        return {f: f'{res_dir}_{os.path.basename(species)}_{radius}_{os.path.basename(f)}' for f in species_sample}

    systems_to_keep = {}
    species_list = glob.glob(os.path.join(res_dir, '*'))
    if not species_list:
        print(f'No species for {res_dir}')
        return
    # Put the ions first
    if favor_ions:
        n_non_ions = sum(1 for x in species_list if os.path.basename(x).count('+') == 0 and os.path.basename(x).count('-') == 0)
        n_ions = len(species_list) - n_non_ions
        n_needed = TOTAL_PER_SYSTEM
    else:
        n_to_take = round(TOTAL_PER_SYSTEM / len(species_list))
    for idx, species in enumerate(species_list):
        species_sample = {}
        if not favor_ions or idx > n_ions:
            n_needed = n_to_take
        elif idx == n_ions: # first non-ion
            n_to_take = round(n_needed / n_non_ions)
            n_needed = n_to_take
        for radius in RADIUS_LIST:
            species_glob = glob.glob(os.path.join(species, f'radius_{radius}', '*'))
            filtered_species = filter_heavy_atoms(species_glob)
            if n_needed < len(filtered_species):
                species_sample.update(sample_to_dict(random.sample(filtered_species, n_needed)))
                break
            else:
                species_sample.update(sample_to_dict(filtered_species))
                n_needed -= len(filtered_species)
        systems_to_keep.update(species_sample)

    save_samples(os.path.dirname(res_dir), systems_to_keep)


def save_samples(path, systems_to_keep):
    save_dir = 'sampled_electrolytes'
    os.makedirs(os.path.join(path, save_dir), exist_ok=True)
    for fname, name in systems_to_keep.items():
        st = StructureReader.read(fname)
        st.write(os.path.join(os.path.dirname(name), save_dir, os.path.basename(name).replace('.mae', '.xyz')))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--favor_ions", action='store_true')
    return parser.parse_args()


def main(output_path, favor_ions):
    dir_list = [f for f in glob.glob(os.path.join(output_path, '*')) if os.path.basename(f).isdigit()]
    sample_fxn = partial(sample_clusters, favor_ions=favor_ions)
    with mp.Pool(60) as pool:
        list(tqdm(pool.imap(sample_fxn, dir_list), total=len(dir_list)))

if __name__ == "__main__":
    args = parse_args()
    main(args.output_path, args.favor_ions)
