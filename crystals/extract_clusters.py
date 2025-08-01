import argparse
import glob
import multiprocessing as mp
import os
import random
from functools import partial

import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.ML.Cluster import Butina
from schrodinger.adapter import to_rdkit
from schrodinger.application.jaguar.utils import mmjag_update_lewis
from schrodinger.application.matsci import clusterstruct
from schrodinger.application.matsci.aseutils import get_structure
from schrodinger.application.matsci.nano.xtal import (Crystal,
                                                      SuperCellBuilder,
                                                      connect_atoms)
from schrodinger.structure import Structure
from schrodinger.structutils import build
from schrodinger.structutils.analyze import evaluate_asl
from tqdm import tqdm

df = pd.read_csv('/large_experiments/ondr_fair/vaheg/osc_data/scripts_stats/short_data_omc_spg_from_genarris.csv')
TRAIN_CODES = df[df['split'] == 'train']['csd_refcode'].unique()
VAL_CODES = df[df['split'] == 'val']['csd_refcode'].unique()
TEST_CODES = df[df['split'] == 'test']['csd_refcode'].unique()

def get_frame_indices(traj, n_atoms):
    """
    Selects at most n_atoms from the trajectory, evenly spaced and including the first and last frames.
    """
    n_frames = len(traj)
    if n_frames <= n_atoms:
        return list(range(n_frames))
    stride = (n_frames-1) / (n_atoms - 1)  # -2 to account for first and last frames
    indices = [round(i * stride) for i in range(n_atoms)]
    return indices

def select_clusters(st, budget):
    max_radius = 20.0
    min_radius = 2.0
    best_atoms = set()
    mol_num = random.choice(range(1, st.mol_total + 1))
    at_num = random.choice(range(1, st.atom_total + 1))
    while max_radius - min_radius > 0.02:
        mid = (min_radius + max_radius) / 2
        sphere = set(
            #evaluate_asl(st, f"fillmol within {mid} mol.num {mol_num}")
            evaluate_asl(st, f"fillmol within {mid} at.num {at_num}")
        )
        if len(sphere) > budget:
            max_radius = mid
        else:
            best_atoms = sphere
            min_radius = mid
    if not best_atoms and len(sphere) < budget + 50:
        best_atoms = sphere
    if not best_atoms:
        print("This system is just too big")
        return None
    selected_ats = best_atoms
    cluster, at_map = build.extract_structure(st, selected_ats, copy_props=True, renumber_map=True)
    #core = [at_map[at.index] for at in st.molecule[mol_num].atom]
    #clusterstruct.contract_structure(cluster, contract_on_atoms=core)
    clusterstruct.contract_structure(cluster, contract_on_atoms=[at_map[at_num]])
    if cluster.mol_total < 3:
        print("Need at least a trimer")
        return None
    return cluster

def get_molecules(st, get_conformers=True):
    mol_st = st.molecule[random.choice(range(1, st.mol_total + 1))].extractStructure()
    if get_conformers:
        try:
            rdkit_confs = generate_rdkit_conformers(mol_st, 8)
        except:
            rdkit_confs = []
    else:
        rdkit_confs = []
    final_mol_list = [mol_st] + rdkit_confs
    for mol in final_mol_list:
        box_diag = np.linalg.norm(np.max(mol.getXYZ(), axis=0) - np.min(mol.getXYZ(), axis=0))
        mol.pbc = [box_diag*5]*3
    return final_mol_list

def generate_rdkit_conformers(mol_st: Structure, nconfs: int) -> list[Structure]:
    mmjag_update_lewis(mol_st)
    mol = to_rdkit(mol_st)
    mol.RemoveAllConformers()
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=nconfs, useRandomCoords=True)
    # Cluster conformers
    dists = []
    for i in range(len(cids)):
        for j in range(i):
            dists.append(rdMolAlign.GetBestRMS(mol,mol,i,j))
    clusts = Butina.ClusterData(dists, len(cids), 1.5, isDistData=True, reordering=True)
    centroids = [x[0] for x in clusts]

    # Optimize conformers
    #AllChem.MMFFOptimizeMoleculeConfs(mol)

    conf_list = []
    for i, conf_id in enumerate(centroids):
        conf = mol.GetConformer(conf_id)
        conf_st = mol_st.copy()
        conf_st.setXYZ(conf.GetPositions()+10.0)
        conf_st.title = f'RDKit: {i}'
        conf_list.append(conf_st)
    return conf_list

def get_split(csd_code):
    if csd_code in TRAIN_CODES:
        return 'train'
    elif csd_code in VAL_CODES:
        return 'val'
    elif csd_code in TEST_CODES:
        return 'test'
    else:
        return None

def parallel_work(traj_file, output_path, do_molecules, max_atoms):
    traj = Trajectory(traj_file)
    selected_indices = get_frame_indices(traj, 10)
    basename = os.path.splitext(os.path.basename(traj_file))[0]
    csd_code, expected_Z, *_ = basename.split('_')
    split = get_split(csd_code)
    if split is None:
        print(f"Skipping {csd_code} as it is not in any split.")
        return
    for frame_idx in selected_indices:
        atoms = traj[frame_idx]
        st = get_structure(atoms)
        connect_atoms(st)
        clusterstruct.contract_structure(st)
        st = SuperCellBuilder(st, (4,4,4)).run()
        #st.write(f'{basename}_{frame_idx}.mae')
        #xtal = Crystal(st, bonding='on', ncella=3, ncellb=3, ncellc=3, translate_centroids=True)
        #xtal.orchestrate()
        #st = xtal.crystal_super_cell
        if st.mol_total != int(expected_Z)*64:
            print(f"Skipping {csd_code} frame {frame_idx} as it has {st.mol_total} molecules, expected {int(expected_Z)*64}.")
            return

        # Clusters
        cluster = select_clusters(st, max_atoms)
        if cluster is None:
            return
        dir_name = os.path.join(output_path, 'clusters', split) 
        os.makedirs(dir_name, exist_ok=True)
        cluster.write(os.path.join(dir_name, f'{basename}_{frame_idx}_0_1.mae'))

        # Molecules
        if do_molecules:
            dir_name = os.path.join(output_path, 'molecules', split) 
            os.makedirs(dir_name, exist_ok=True)
            if frame_idx == 0:
                mols = get_molecules(st, get_conformers=True)
                for mol_idx, mol in enumerate(mols):
                    mol.write(os.path.join(output_path, 'molecules', split, f'{basename}_{frame_idx}_{mol_idx}.cif'))
            elif frame_idx == selected_indices[-1]:
                mols = get_molecules(st, get_conformers=False)
                # Always call the last frame 9, regardless of the number of conformers
                mols[0].write(os.path.join(output_path, 'molecules', split, f'{basename}_{frame_idx}_9.cif'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--max_atoms", type=int, default=280)
    parser.add_argument("--skip_molecules", action='store_false', dest='do_molecules')
    return parser.parse_args()


def main(output_path, n_chunks, chunk_idx, n_workers, do_molecules, max_atoms):
    traj_list = glob.glob('/large_experiments/ondr_fair/vaheg/osc_data/traj/finished_relaxations/batch*/sampled_traj_new_clean/*.traj')
    chunks_to_process = np.array_split(traj_list, n_chunks)
    chunk = chunks_to_process[chunk_idx]
    fxn = partial(parallel_work, output_path=output_path, do_molecules=do_molecules, max_atoms=max_atoms)
    with mp.Pool(n_workers) as pool:
        list(tqdm(pool.imap(fxn, chunk), total=len(chunk)))
    #for traj_file in glob.glob('/large_experiments/ondr_fair/vaheg/osc_data/traj/finished_relaxations/batch1/sampled_traj_new_clean/ABAFEQ_2_gener_4da4f9f3c6da5d7.traj'):

if __name__ == '__main__':
    args= parse_args()
    main(args.output_path, args.n_chunks, args.chunk_idx, args.n_workers, args.do_molecules, args.max_atoms)
