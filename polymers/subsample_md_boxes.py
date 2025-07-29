import argparse
import glob
import os
import random
import multiprocessing as mp
from collections import Counter
from itertools import cycle

from schrodinger.application.matsci import clusterstruct
from schrodinger.structure import StructureReader
from schrodinger.structutils import build
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.structutils.ringspear import check_for_spears

from tqdm import tqdm

CHECK_FOR_MISSING_H = False
MAX_TRIES = 100

def get_cut_ends(st, at_set):
    """
    Get the indices of the atoms that have dangling bonds.

    These are the atoms in at_list where their bonded atoms in the full structure are not all in at_list.
    """
    cut_ends = []
    for at in at_set:
        bonded_atoms = set(b_at.index for b_at in st.atom[at].bonded_atoms)
        if not bonded_atoms.issubset(at_set):
            cut_ends.append(at)
    return cut_ends


def get_tail_init(st, chain_name):
    """
    Get the indices of the terminal residues in the given chain.
    """
    main_chain = st.chain[chain_name]
    termini = [
        res for res in main_chain.residue if res.pdbres.startswith("T")
    ]  # This might be TU0 always, but just in case...
    end = random.choice(termini)  # Randomly choose one of the termini
    return end


def get_middle_init(st, chain_name):
    """
    Get the indices of the terminal residues in the given chain.
    """
    main_chain = st.chain[chain_name]
    res_list = list(main_chain.residue)
    # randomly draw from the middle fifth of residues
    mid_index = random.choice(range(len(res_list)//5*2, len(res_list)//5*3))
    mid_residue = res_list[mid_index]  # Get the middle residue
    return mid_residue


def get_cylinder(st, chain_name, radius, budget, tail_init):
    """
    Get the end residue and the atoms within radius r of it (fillres), check if this is within budget, if not, terminate, add hydrogens to the residue boundaries
    if it is add another residue to the main chain and include the atoms within radius r of it (fillres), repeat until budget is reached.
    """
    if tail_init:
        init_res = get_tail_init(st, chain_name)
    else:
        init_res = get_middle_init(st, chain_name)
    main_chain_atoms = [at.index for at in init_res.atom]
    selected_atoms = set(main_chain_atoms)
    while len(selected_atoms) < budget:
        str_mc_atoms = ",".join(map(str, main_chain_atoms))
        neighbors = evaluate_asl(
            st,
            f"fillres (( within {radius} at.num {str_mc_atoms}) and not (fillres withinbonds 1 at.num {str_mc_atoms}))",
        )
        cand_atoms = selected_atoms.union(neighbors + main_chain_atoms)
        # Cut ends are capped with H's (and probably only one)
        cut_ends = get_cut_ends(st, cand_atoms)
        if len(cand_atoms) + len(cut_ends) > budget:
            break
        selected_atoms.update(neighbors + main_chain_atoms)
        main_chain_atoms = evaluate_asl(
            st, f"fillres withinbonds 1 at.num {str_mc_atoms}"
        )
    return selected_atoms, init_res

def remove_lonesome_hydrogens(cluster):
    """
    Remove any isolated hydrogen atoms

    It is possible to end up with isolated H atoms since they are
    common termination groups and we might pull in the termination,
    but nothing else.
    """
    lone_H = evaluate_asl(cluster, "at.ele H and at.att 0")
    if lone_H:
        cluster.deleteAtoms(lone_H)
    

def get_sphere(st, chain_name, budget, tail_init):
    """
    Get the start residue atoms and expand a sphere until the budget is reached.
    Since we don't know a good guess for the radiues to do a binary search, we will just slowly increase the radius until we reach the budget.
    """
    if tail_init:
        init_res = get_tail_init(st, chain_name)
    else:
        init_res = get_middle_init(st, chain_name)
    main_chain_atoms = [at.index for at in init_res.atom]
    selected_atoms = set(main_chain_atoms)
    str_mc_atoms = ",".join(map(str, main_chain_atoms))

    min_radius = 1.0
    cur_radius = min_radius
    max_radius = None
    while len(selected_atoms) < budget:
        sphere = evaluate_asl(st, f"fillres within {cur_radius} at.num {str_mc_atoms}")
        cut_ends = get_cut_ends(st, sphere)
        if len(sphere) + len(cut_ends) > budget:
            max_radius = cur_radius
            break
        else:
            selected_atoms.update(sphere)
            min_radius = cur_radius
            cur_radius += 3.0
    # refine the sphere to the budget by doing a binary search to 0.1A such that we get at most the budget
    if max_radius is not None:
        best_atoms = set()
        while max_radius - min_radius > 0.1:
            mid = (min_radius + max_radius) / 2
            sphere = set(
                evaluate_asl(st, f"fillres within {mid} at.num {str_mc_atoms}")
            )
            cut_ends = get_cut_ends(st, sphere)
            if len(sphere) + len(cut_ends) > budget:
                max_radius = mid
            else:
                best_atoms = sphere
                min_radius = mid
        selected_atoms = best_atoms
    return selected_atoms, init_res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".", type=str)
    return parser.parse_args()

def extract_cluster(st, extracted_ats):
    cut_ends = get_cut_ends(st, extracted_ats)
    cluster, at_map = build.extract_structure(
        st, extracted_ats, copy_props=True, renumber_map=True
    )
    clusterstruct.contract_structure2(cluster)
    mapped_cut_ends = [at_map[at] for at in cut_ends]
    build.add_hydrogens(cluster, atom_list=mapped_cut_ends)
    remove_lonesome_hydrogens(cluster)
    if check_for_spears(cluster, distorted=True, return_bool=True):
        return
    if CHECK_FOR_MISSING_H:
        h_check = cluster.copy()
        build.add_hydrogens(h_check)
        if h_check.atom_total != cluster.atom_total:
            print("I felt the need to add more hydrogens!")
    return cluster

def main_loop(fname):
    st = StructureReader.read(fname)
    total_clusters = 10
    for mol in st.molecule:
        for at in mol.atom:
            at.chain = str(mol.number - 1)
    n_clusters = 0
    tries = 0
    chain_iter = iter(cycle(st.chain))
    used_seeds = set()
    while n_clusters < total_clusters and tries < MAX_TRIES:
        tries += 1
        n_total_atoms = random.choice([150, 200, 300])
        chain = next(chain_iter)
        end_init = random.random() < 0.5 # If true, sample from end of polymer chain. Else, sample from middle.
        cyl_topo = random.random() < 0.5 # If true, sample extract a cylinder around the chain, else sample a sphere around the chain.
        if cyl_topo:
            extracted_ats, init_res = get_cylinder(
                st, chain.name, 2.5, n_total_atoms, end_init
            )
        else:
            extracted_ats, init_res = get_sphere(st, chain.name, n_total_atoms, end_init)
        if len(extracted_ats) < 70:
            continue
        if (chain.name, init_res.resnum) in used_seeds:
            continue
        cluster = extract_cluster(st, extracted_ats)
        if cluster is None:
            continue
        name_parts = os.path.splitext(fname)[0].split("/")
        cluster_name = "_".join(name_parts + [chain.name,str(init_res.resnum), "0", "1"]) + ".mae"
        cluster.write(cluster_name)
        used_seeds.add((chain.name, init_res.resnum))
        n_clusters += 1

def main():
    args = parse_args()
    file_list = glob.glob("**/*.pdb", recursive=True)
    with mp.Pool(60) as pool:
        list(tqdm(pool.imap(main_loop, file_list), total=len(file_list)))

if __name__ == '__main__':
    random.seed(42)
    main()
