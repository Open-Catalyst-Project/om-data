"""
Script for sampling from extracted clusters
"""

import argparse
import glob
import os
import random
import shutil
import multiprocessing as mp
from tqdm import tqdm
from schrodinger.structure import StructureReader
from collections import defaultdict

TOTAL_PER_SYSTEM = 50
RADIUS_LIST = [7]
MAX_ATOMS = 2
# radius 5 seed 4353
# radius 7 seed 353
random.seed(353)

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
        if sum(1 for at in st.atom if at.atomic_number > 20) < MAX_ATOMS:
            species_list.extend(group_list)
    return species_list
        

def sample_clusters(res_dir):
    def sample_to_dict(species_sample):
        return {f: f'{res_dir}_{os.path.basename(species)}_{radius}_{os.path.basename(f)}' for f in species_sample}

    systems_to_keep = {}
    species_list = glob.glob(os.path.join(res_dir, '*'))
    if not species_list:
        print(f'No species for {res_dir}')
        return
    n_to_take = round(TOTAL_PER_SYSTEM / len(species_list))
    for species in species_list:
        species_sample = {}
        n_needed = n_to_take
        for radius in RADIUS_LIST:
            species_glob = glob.glob(os.path.join(species, f'radius_{radius}', '*'))
            species_list = filter_heavy_atoms(species_glob)
            if n_needed < len(species_list):
                species_sample.update(sample_to_dict(random.sample(species_list, n_needed)))
                break
            else:
                species_sample.update(sample_to_dict(species_list))
                n_needed -= len(species_list)
        systems_to_keep.update(species_sample)

    save_samples(os.path.dirname(res_dir), systems_to_keep)


def save_samples(path, systems_to_keep):
    save_dir = 'sampled_electrolytes2'
    os.makedirs(os.path.join(path, save_dir), exist_ok=True)
    for fname, name in systems_to_keep.items():
        st = StructureReader.read(fname)
        st.write(os.path.join(os.path.dirname(name), save_dir, os.path.basename(name).replace('.mae', '.xyz')))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


def main():
    args = parse_args()
    pool = mp.Pool(60)
    dir_list = [f for f in glob.glob(os.path.join(args.output_path, '*')) if os.path.basename(f).isdigit()]
    list(tqdm(pool.imap(sample_clusters, dir_list), total=len(dir_list)))

if __name__ == "__main__":
    main()
