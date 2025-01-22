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

TOTAL_PER_SYSTEM = 50
RADIUS_LIST = [7]
MAX_METALS = 2
# random.seed(4353) # for %a
random.seed(1269) # fo 7A

def filter_heavy_atoms(species_glob):
    species_list = []
    for species in species_glob:
        st = StructureReader.read(species)
        st.title = species
        if sum(1 for at in st.atom if at.atomic_number > 20) < MAX_METALS:
            species_list.append(st)
    return species_list
        

def sample_clusters(res_dir):
    def sample_to_dict(species_sample):
        return {st: f'{res_dir}_{os.path.basename(species)}_{radius}_{os.path.basename(st.title)}' for st in species_sample}

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
    save_dir = 'sampled_electrolytes2/7A'
    os.makedirs(os.path.join(path, save_dir), exist_ok=True)
    for st, name in systems_to_keep.items():
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
