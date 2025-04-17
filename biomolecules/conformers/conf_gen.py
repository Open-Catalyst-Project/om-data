import argparse
import glob
import random
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from schrodinger.comparison import are_conformers
from schrodinger.structure import (Structure, StructureReader, StructureWriter,
                                   create_new_structure)
from schrodinger.structutils import analyze, rmsd
from schrodinger.adapter import to_rdkit, to_structure
from schrodinger.application.jaguar.autots_bonding import clean_st
from schrodinger.application.jaguar.file_logger import FileLogger
from schrodinger.application.jaguar.packages.collate_conformers import collate
from schrodinger.application.jaguar.packages.csrch import run_csrch, eliminate_duplicate_conformers
from schrodinger.application.jaguar.packages.shared import read_cartesians, get_smiles
from schrodinger.application.matsci import clusterstruct
from schrodinger.comparison.atom_mapper import ConnectivityAtomMapper

N_CONF = 100
SCHRO = "/private/home/levineds/schrodinger2024-2/"

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

def generate_rdkit_conformers(mol: Chem.Mol, mol_st: Structure, nconfs: int) -> List[Structure]:
    mol.RemoveAllConformers()
    ncpu = 1
    confs = AllChem.EmbedMultipleConfs(
        mol, numConfs=nconfs, numThreads=ncpu, useRandomCoords=True
    )
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    conf_list = []
    for i, conf in enumerate(mol.GetConformers()):
        conf_st = mol_st.copy()
        conf_st.setXYZ(conf.GetPositions())
        conf_st.title = f'RDKit: {i}'
        conf_list.append(conf_st)
    return conf_list

def reopt_conformers_with_xtb(st_list):
    with tempfile.TemporaryDirectory() as temp_dir:
        with chdir(temp_dir):
            xyz_file = 'struc.xyz'
            with StructureWriter(xyz_file) as writer:
                writer.extend(st_list)
            remove_xyz_newlines(xyz_file)
            with open('crest.out', 'w') as fh:
                subprocess.run([f'{SCHRO}/run', 'crest', '--mdopt', xyz_file, '--gfn2', '--opt', 'normal'], stdout=fh)
            conf_file = 'crest_ensemble.xyz'
            if not os.path.exists(conf_file):
                return []
            carts = read_cartesians(conf_file)

    opt_st_list = []
    for old_st, new_cart in zip(st_list, carts):
        new_st = clean_st(new_cart.getStructure())
        new_st.property['r_j_Gas_Phase_Energy'] = float(new_st.title)
        opt_st_list.append(new_st)
    return opt_st_list

def remove_xyz_newlines(xyz_file):
    with open(xyz_file, 'r') as fh:
        data = fh.readlines()
    cleaned_data = []
    line_idx = 0
    while line_idx < len(data):
        n_atoms = int(data[line_idx])
        cleaned_data.extend(data[line_idx:line_idx+n_atoms+2])
        line_idx += n_atoms + 3
    with open(xyz_file, 'w') as fh:
        fh.writelines(cleaned_data)


def generate_crest_conformers(st: Structure, nconfs: int, method='gfn2') -> List[Structure]:
    with tempfile.TemporaryDirectory() as temp_dir:
        with chdir(temp_dir):
            xyz_file = 'struc.xyz'
            #st.generate3dConformation(require_stereo=False) # Remove the original 3D conformation
            st.write(xyz_file)
            with open('crest.out', 'w') as fh:
                subprocess.run(['crest', xyz_file, f'--{method}'], stdout=fh)
            conf_file = 'crest_conformers.xyz'
            if not os.path.exists(conf_file):
                return []
            carts = read_cartesians('crest_conformers.xyz')

    st_list = []
    for idx, cart in enumerate(carts[:nconfs]):
        st = clean_st(cart.getStructure())
        st.title = f'CREST: {idx}'
        st_list.append(st)
    st_list = reopt_conformers_with_xtb(st_list)
    return st_list

def generate_sdgr_conformers(st: Structure, nconfs: int) -> List[Structure]:
    st.generate3dConformation(require_stereo=False) # Remove the original 3D conformation
    with tempfile.TemporaryDirectory() as temp_dir:
        with chdir(temp_dir), FileLogger('csrch', False):
            print(os.getcwd())
            sdgr_confs = run_csrch(st, 'sdgr_csrch', max_conf=nconfs, erg_window=500)#in KJ/mol
    for idx, st in enumerate(sdgr_confs):
        st.title = f'SDGR: {idx}'
    return sdgr_confs

def renumber_molecules_to_match(ref_st, st_list):
    """
    Ensure that topologically equivalent sites are equivalently numbered
    """
    mapper = ConnectivityAtomMapper(use_chirality=False)
    atlist = range(1, ref_st.atom_total + 1)
    renumbered_mols = []
    for st in st_list:
        _, r_st = mapper.reorder_structures(ref_st, atlist, st, atlist)
        renumbered_mols.append(r_st)
    return renumbered_mols

def main(output_path, job_idx):
    flist = sorted(glob.glob(os.path.join(output_path, '*_ligand_0.mae')))
    mae_name = flist[job_idx]
    print(mae_name, flush=True)
#    xyz_file  = '/private/home/levineds/7g7l_ABS03_state0_1_ligand_0_conf_0_0_1.xyz'
#    mol_st = clean_st(read_cartesians(xyz_file)[0].getStructure())
    mol_st = StructureReader.read(mae_name)
    ref_st = mol_st.copy()
    #n_rotatable = analyze.get_num_rotatable_bonds(ref_st) # excludes trivial rotors like methyl groups
    #print(n_rotatable)
    #output['n_rotatable'] = n_rotatable
    rdmol = to_rdkit(mol_st)

    # RDKit
    try:
        rdkit_confs = generate_rdkit_conformers(rdmol, mol_st, N_CONF)
    except:
        rdkit_confs = []
    print('rdkit', len(rdkit_confs))

    # SDGR (MacroModel)
    try:
        sdgr_confs = generate_sdgr_conformers(mol_st, N_CONF)
    except:
        sdgr_confs = []
    print('sdgr', len(sdgr_confs))

    # RDKit + xTB
    try:
        xtb_confs = reopt_conformers_with_xtb(rdkit_confs + sdgr_confs)
    except:
        xtb_confs = []
    print('xtb', len(xtb_confs))

    with tempfile.TemporaryDirectory() as temp_dir:
        with chdir(temp_dir), FileLogger('csrch', False):
            final_list = eliminate_duplicate_conformers(xtb_confs)
    print('final', len(final_list))

    #with StructureWriter(f'dump.mae') as writer:
    #    writer.extend(final_list)
    dirname = os.path.join(output_path, 'confs')
    os.makedirs(dirname, exist_ok=True)
    for idx, st in enumerate(final_list[:N_CONF]):
        st.write(os.path.join(dirname, os.path.basename(mae_name).replace('_ligand_0.mae', f'_ligand_0_conf_{idx}.mae')))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--job_idx", type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.output_path, args.job_idx)
