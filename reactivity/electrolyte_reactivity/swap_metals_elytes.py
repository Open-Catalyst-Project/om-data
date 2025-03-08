import argparse
import multiprocessing as mp
import os
import csv
import math
import tarfile
from typing import List, Tuple
# from urllib.request import urlretrieve
# from architector.io_molecule import convert_io_molecule
# from architector.io_align_mol import align_rmsd
# from omdata.reactivity_utils import find_min_distance
from tqdm import tqdm
# from gpsts.geodesic import construct_geodesic_path
from pymatgen.core.structure import Molecule
import pandas as pd
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
import ase
from more_itertools import collapse
from rdkit import Chem
from rdkit.Chem import AllChem
from schrodinger.adapter import to_structure
from schrodinger.application.jaguar.autots_input import AutoTSInput
from schrodinger.application.jaguar.autots_rmsd import \
    reform_barely_broken_bonds
from schrodinger.application.jaguar.file_logger import FileLogger
from schrodinger.application.jaguar.packages.autots_modules.active_bonds import \
    mark_active_bonds
from schrodinger.application.jaguar.packages.autots_modules.autots_stereochemistry import \
    ChiralityMismatchError
from schrodinger.application.jaguar.packages.autots_modules.complex_formation import (
    _add_atom_transfer_dummies, _remove_atom_transfer_dummies,
    minimize_path_distance, reaction_center, translate_close)
from schrodinger.application.jaguar.packages.autots_modules.renumber import \
    build_reaction_complex
from schrodinger.application.jaguar.packages.reaction_mapping import \
    build_reaction_complex as get_renumbered_complex
from schrodinger.application.jaguar.packages.reaction_mapping import (
    flatten_st_list, get_net_matter)
from schrodinger.infra import fast3d
from schrodinger.structure import Structure, StructureWriter, StructureReader
from schrodinger.structutils.transform import get_centroid, translate_structure
from schrodinger.application.matsci.aseutils import get_structure
from schrodinger.application.jaguar.autots_bonding import clean_st
from schrodinger.application.jaguar.packages.shared import read_cartesians
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.structutils import rmsd
from ase.data import vdw_alvarez, atomic_numbers
from schrodinger.comparison.atom_mapper import ConnectivityAtomMapper

def renumber_molecules_to_match(mol_list):
    """
    Ensure that topologically equivalent sites are equivalently numbered
    """
    mapper = ConnectivityAtomMapper(use_chirality=False)
    atlist = range(1, mol_list[0].atom_total + 1)
    renumbered_mols = [mol_list[0]]
    for mol in mol_list[1:]:
        _, r_mol = mapper.reorder_structures(mol_list[0], atlist, mol, atlist)
        renumbered_mols.append(r_mol)
    return renumbered_mols

def rmsd_wrapper(st1: Structure, st2: Structure) -> float:
    """
    Wrapper around Schrodinger's RMSD calculation function.
    """
    assert (
        st1.atom_total == st2.atom_total
    ), "Structures must have the same number of atoms for RMSD calculation"
    if st1 == st2:
        return 0.0
    at_list = list(range(1, st1.atom_total + 1))
    return rmsd.superimpose(st1, at_list, st2.copy(), at_list, use_symmetry=True)


def delete_all_m_lig_bonds(st):
    for metal in evaluate_asl(st, 'metals'):
        for bond in list(st.atom[metal].bond):
            st.deleteBond(*bond.atom)

def delete_m_lig_bonds(st, metal_ind):
    for bond in list(st.atom[metal_ind].bond):
        st.deleteBond(*bond.atom)
            

def dilate_distance(st: Structure, scale_factor: float, metal_index: int) -> None:
    """
    Dilate the distances between molecules by a scale factor.

    :param st: The structure to dilate
    :param scale_factor: The factor to change the molecule separation by
    :param metal_index: The index of the metal atom to use as the center of dilation
    """
    metal_centroid = get_centroid(st, atom_list=[metal_index])
    for mol in st.molecule:
        mol_centroid = get_centroid(st, atom_list=mol.getAtomList())
        vec = mol_centroid - metal_centroid
        scaled_vec = vec * scale_factor
        translate_vec = scaled_vec - vec
        translate_structure(st, *translate_vec[:3], atom_index_list=mol.getAtomList())

# Consistent with mendeleev.element(element).ec.ionize(oxidation_state).unpaired_electrons()
plus1_swaps = {"Li":0,"Na":0,"K":0,"Cs":0,"Cu":0,"Ag":0,"Rb":0,"Tl":0,"Hg":1}
plus2_swaps = {"Ca":0,"Mg":0,"Zn":0,"Be":0,"Cu":1,"Ni":2,"Pt":2,"Co":3,"Pd":2,"Ag":1,"Mn":5,"Hg":0,"Cd":0,"Yb":0,"Sn":0,"Pb":0,"Eu":7,"Sm":6,"Cr":4,"Fe":4,"V":3,"Ba":0,"Sr":0}
plus3_swaps = {'La': 0, 'Ce': 1, 'Pr': 2, 'Nd': 3, 'Pm': 4, 'Sm': 5, 'Eu': 6, 'Gd': 7, 'Tb': 6, 'Dy': 5, 'Ho': 4, 'Er': 3, 'Tm': 2, 'Yb': 1, 'Lu': 0, 'Al': 0, 'Ga': 0, 'In': 0, 'Tl': 0, 'Bi': 0, 'Sc': 0, 'Cr': 3, 'Fe': 5, 'Co': 4, 'Y': 0, 'Ru': 5, 'Rh': 4, 'Ir': 4, 'Au': 2}

special_cases_orig = [5367, 3447, 10165, 8330, 5835, 2998, 2365, 1502, 961]
special_cases_mapped = [5218, 10329, 9480, 9333, 11184]
special_cases_both = [5708, 10745, 13336]

def main(args):
    output_dir = args.output_path
    xyz_dir = args.input_path

    os.makedirs(output_dir, exist_ok=True)

    num_reactions = 0

    reaction_histogram = {}

    for i in range(14306):
        print(f"Processing reaction {i}")
        if i in special_cases_both:
            to_consider = ["mapped", "orig"]
        elif i in special_cases_mapped:
            to_consider = ["mapped"]
        elif i in special_cases_orig:
            to_consider = ["orig"]
        else:
            to_consider = []
            if os.path.exists(f"{xyz_dir}/{i}_reactant.xyz"):
                lines = open(f"{xyz_dir}/{i}_reactant.xyz").readlines()
                charge = lines[1].split()[0]
                spin_multiplicity = lines[1].split()[1][0]
                mapped_reactant_st=clean_st(read_cartesians(os.path.join(xyz_dir, f"{i}_reactant.xyz"))[0].getStructure())
                mapped_product_st=clean_st(read_cartesians(os.path.join(xyz_dir, f"{i}_product.xyz"))[0].getStructure())
                mapped_rmsd = rmsd_wrapper(mapped_reactant_st, mapped_product_st)
                if mapped_rmsd > 0.3:
                    to_consider.append("mapped")
            if os.path.exists(f"{xyz_dir}/{i}_orig_reactant.xyz") and os.path.exists(f"{xyz_dir}/{i}_reactant.xyz"):
                run_orig = True
                unmapped_reactant_st=clean_st(read_cartesians(os.path.join(xyz_dir, f"{i}_orig_reactant.xyz"))[0].getStructure())
                unmapped_product_st=clean_st(read_cartesians(os.path.join(xyz_dir, f"{i}_orig_product.xyz"))[0].getStructure())
                for ii, atom in enumerate(unmapped_reactant_st.atom):
                    if atom.element != unmapped_product_st.atom[ii+1].element:
                        run_orig = False
                        break
                if run_orig:
                    unmapped_rmsd = rmsd_wrapper(unmapped_reactant_st, unmapped_product_st)
                    if unmapped_rmsd < 0.3:
                        run_orig = False
                if run_orig and "mapped" in to_consider:
                    try:
                        renumbered_Rs = renumber_molecules_to_match([mapped_reactant_st, unmapped_reactant_st])
                        renumbered_Ps = renumber_molecules_to_match([mapped_product_st, unmapped_product_st])
                        reactant_rmsd = rmsd_wrapper(renumbered_Rs[0], renumbered_Rs[1])
                        product_rmsd = rmsd_wrapper(renumbered_Ps[0], renumbered_Ps[1])
                    except Exception as e:
                        try:
                            delete_all_m_lig_bonds(unmapped_reactant_st)
                            delete_all_m_lig_bonds(unmapped_product_st)
                            delete_all_m_lig_bonds(mapped_reactant_st)
                            delete_all_m_lig_bonds(mapped_product_st)
                            renumbered_Rs = renumber_molecules_to_match([unmapped_reactant_st, mapped_reactant_st])
                            renumbered_Ps = renumber_molecules_to_match([unmapped_product_st, mapped_product_st])
                            reactant_rmsd = rmsd_wrapper(renumbered_Rs[0], renumbered_Rs[1])
                            product_rmsd = rmsd_wrapper(renumbered_Ps[0], renumbered_Ps[1])
                        except Exception as e:
                            print(e)
                            continue
                    if reactant_rmsd < 0.4 or product_rmsd < 0.4 or reactant_rmsd+product_rmsd < 1.4:
                        run_orig = False
                if run_orig:
                    to_consider.append("orig")

        for option in to_consider:
            if option == "mapped":
                orig_reactant_st=clean_st(read_cartesians(os.path.join(xyz_dir, f"{i}_reactant.xyz"))[0].getStructure())
                orig_product_st=clean_st(read_cartesians(os.path.join(xyz_dir, f"{i}_product.xyz"))[0].getStructure())
            elif option == "orig":
                orig_reactant_st=clean_st(read_cartesians(os.path.join(xyz_dir, f"{i}_orig_reactant.xyz"))[0].getStructure())
                orig_product_st=clean_st(read_cartesians(os.path.join(xyz_dir, f"{i}_orig_product.xyz"))[0].getStructure())
            else:
                continue
                
            metal_dict = {}

            for metal_ind in evaluate_asl(orig_reactant_st, 'metals'):
                metal_dict[metal_ind] = orig_reactant_st.atom[metal_ind].element

            if metal_dict == {}:
                for st, name in zip((orig_reactant_st, orig_product_st), ("R", "P")):
                    st.title = f"{charge} {spin_multiplicity}"
                    with StructureWriter(f"{output_dir}/{i}_{option}_{name}_0.xyz", format="xyz") as writer:
                        writer.extend([st])
                num_reactions += 1
                if orig_reactant_st.atom_total not in reaction_histogram:
                    reaction_histogram[orig_reactant_st.atom_total] = 1
                else:
                    reaction_histogram[orig_reactant_st.atom_total] += 1
                

            for j, metal_ind in enumerate(metal_dict.keys()):
                metal = metal_dict[metal_ind]
                if metal in plus1_swaps:
                    swaps_dict = {0: plus1_swaps}
                elif metal in plus2_swaps:
                    swaps_dict = {0: plus2_swaps, 1: plus3_swaps}
                else:
                    print(f"Metal {metal} is not in plus1 or plus2 swaps! Exiting...")
                    exit()
                for charge_change in swaps_dict.keys():
                    swaps = swaps_dict[charge_change]
                    for el in swaps.keys():
                        swapped_reactant_st = orig_reactant_st.copy()   
                        swapped_product_st = orig_product_st.copy()
                        swapped_reactant_st.atom[metal_ind].element = el
                        swapped_product_st.atom[metal_ind].element = el
                        ratio = vdw_alvarez.vdw_radii[atomic_numbers[el]] / vdw_alvarez.vdw_radii[atomic_numbers[metal]]
                        for st, name in zip((swapped_reactant_st, swapped_product_st), ("R", "P")):
                            delete_m_lig_bonds(st, metal_ind)
                            dilate_distance(st, ratio, metal_ind)
                            st.title = f"{int(charge)+int(charge_change)} {int(spin_multiplicity) + int(swaps[el])}"
                            filename = f"{output_dir}/{i}_{option}_{name}"
                            for k, key in enumerate(metal_dict.keys()):
                                if k == j:
                                    assert st.atom[key].element == el
                                    filename = f"{filename}_{el}{k}"
                                else:
                                    assert st.atom[key].element == metal_dict[key]
                                    filename = f"{filename}_{st.atom[key].element}{k}"
                            filename = f"{filename}_{charge_change}.xyz"
                            with StructureWriter(filename, format="xyz") as writer:
                                writer.extend([st])
                        if swapped_reactant_st.atom_total not in reaction_histogram:
                            reaction_histogram[swapped_reactant_st.atom_total] = 1
                        else:
                            reaction_histogram[swapped_reactant_st.atom_total] += 1
                        num_reactions += 1

    print(f"Output {num_reactions} reactions")
    print(reaction_histogram)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=".")
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
