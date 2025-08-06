import argparse
import csv
import itertools
import glob
import math
import os
import random
import sys
import numpy as np
from collections import defaultdict

import mendeleev
from functools import partial,reduce
from tqdm import tqdm
import multiprocessing as mp
from schrodinger.application.jaguar.packages.shared import \
    uniquify_with_comparison
from schrodinger.application.jaguar.utils import LewisModes, mmjag_update_lewis, mmjag_reset_connectivity
from schrodinger.application.jaguar.autots_bonding import clean_st
from schrodinger.application.matsci import clusterstruct
#from schrodinger.application.jaguar.utils import group_with_comparison
from schrodinger.application.jaguar.utils import group_items
from schrodinger.application.matsci.nano.xtal import (connect_atoms, get_cell,
                                                      is_infinite)
from schrodinger.comparison import are_conformers
from schrodinger.structure import StructureReader, create_new_structure
from schrodinger.structutils.analyze import (evaluate_asl, evaluate_smarts,
                                             has_valid_lewis_structure,
                                             hydrogens_present)
from schrodinger.structutils.build import remove_alternate_positions
from schrodinger.structutils.measure import get_close_atoms

MAX_VALENCIES = {'H': 4, 'He': 4, 'Li': 8, 'Be': 8, 'B': 8, 'C': 5, 'N': 5, 'O': 5, 'F': 5, 'Ne': 8, 'Na': 8, 'Mg': 8, 'Al': 8, 'Si': 8, 'P': 8, 'S': 8, 'Cl': 8, 'Ar': 8, 'K': 8, 'Ca': 8, 'Sc': 9, 'Ti': 9, 'V': 9, 'Cr': 9, 'Mn': 9, 'Fe': 9, 'Co': 9, 'Ni': 9, 'Cu': 9, 'Zn': 9, 'Ga': 9, 'Ge': 9, 'As': 8, 'Se': 8, 'Br': 8, 'Kr': 8, 'Rb': 8, 'Sr': 8, 'Y': 9, 'Zr': 9, 'Nb': 9, 'Mo': 9, 'Tc': 9, 'Ru': 9, 'Rh': 9, 'Pd': 9, 'Ag': 9, 'Cd': 9, 'In': 9, 'Sn': 9, 'Sb': 9, 'Te': 8, 'I': 8, 'Xe': 8, 'Cs': 8, 'Ba': 8, 'La': 9, 'Ce': 9, 'Pr': 10, 'Nd': 9, 'Pm': 9, 'Sm': 9, 'Eu': 9, 'Gd': 9, 'Tb': 9, 'Dy': 9, 'Ho': 9, 'Er': 9, 'Tm': 9, 'Yb': 9, 'Lu': 9, 'Hf': 9, 'Ta': 9, 'W': 9, 'Re': 9, 'Os': 9, 'Ir': 9, 'Pt': 9, 'Au': 9, 'Hg': 9, 'Tl': 9, 'Pb': 9, 'Bi': 9, 'Po': 9, 'At': 8, 'Rn': 8, 'Fr': 9, 'Ra': 9, 'Ac': 9, 'Th': 9, 'Pa': 9, 'U': 9, 'Np': 9, 'Pu': 9, 'Am': 9, 'Cm': 9, 'Bk': 9, 'Cf': 9, 'Es': 9, 'Fm': 9, 'Md': 9, 'No': 9, 'Lr': 9, 'Rf': 9, 'Db': 9, 'Sg': 9, 'Bh': 9, 'Hs': 9, 'Mt': 9, 'Ds': 9, 'Rg': 9, 'Cn': 9, 'Nh': 9, 'Fl': 9, 'Mc': 9, 'Lv': 9, 'Ts': 1, 'Og': 1, 'DU': 15, 'Lp': 15, '': 15}

def guess_spin_state(st):
    metals = evaluate_asl(st, "metals")
    # We will assume antiferromagnetic coupling for multimetallic systems
    # to ensure we don't put the spin state outside our acceptable range
    total_spin = 0
    local_spins = []
    for metal_idx in metals:
        metal_at = st.atom[metal_idx]
        local_spin = mendeleev.element(metal_at.element).ec.ionize(metal_at.formal_charge).unpaired_electrons()
        # Assume 2nd, 3rd row TMs are low spin, Ln are high spin
        if metal_at.atomic_number > 36 and not metal_at.atomic_number in range(59,70): 
            local_spin = local_spin % 2 
        if local_spin > 0:
            local_spins.append(local_spin)
    for idx, local_spin in enumerate(local_spins):
        total_spin += (-1) ** idx * local_spin
    return abs(total_spin) + 1

def get_cif_location(cod_path, cif_entry):
    dir1 = cif_entry[0]
    dir2 = cif_entry[1:3]
    dir3 = cif_entry[3:5]
    return os.path.join(cod_path, dir1, dir2, dir3, cif_entry + ".cif")

def has_collisions(st):
    def get_expected_length(at1, at2):
        total = 0
        for at in (at1, at2):
            radius = mendeleev.element(at.element).covalent_radius_pyykko
            total += radius
        return total / 100.0
    
    return any(bond.length < 0.55 * get_expected_length(*bond.atom) for bond in st.bond)

def fix_carbonyls(st):
    def revise_carbonyl(st, carb):
        O = st.atom[carb[0]]
        C = st.atom[carb[1]]
        M = st.atom[carb[2]]
        st.getBond(O,C).order = 3
        st.getBond(C,M).order = 1
        O.formal_charge = 1
        C.formal_charge = -1
        M.formal_charge -= 2

    for term_pattern in ('[OX1+0]=[CX2-2]=[!#6,!#7,!#8,!#16,!#1]', '[NX2+0]=[CX2-2]=[!#6,!#7,!#8,!#16,!#1]'):
        terminal_carbonyls = evaluate_smarts(st, term_pattern)
        metals = evaluate_asl(st, "metals")
        for carb in terminal_carbonyls:
            if carb[2] not in metals:
                continue
            revise_carbonyl(st, carb)
    bridging_carbonyls = evaluate_smarts(st, '[OX1+0]=[CX2-2]([!#6,!#7,!#8,!#16,!#1])[!#6,!#7,!#8,!#16,!#1]')
    for carb in bridging_carbonyls:
        if carb[2] not in metals or carb[3] not in metals or st.atom[carb[2]].formal_charge < st.atom[carb[3]].formal_charge:
            continue
        revise_carbonyl(st, carb)

def remove_metal_metal_bonds(st):
    metals = evaluate_asl(st, "metals")
    for at1, at2 in itertools.combinations(metals, 2):
        bond = st.getBond(at1, at2)
        if bond is not None:
            n_mol = st.mol_total
            st.deleteBond(*bond.atom)
            if st.mol_total != n_mol:
                st.addBond(*bond.atom, 1)


def resolve_disorder(st):
    disordered_atoms = [at for at in st.atom if int(at.property.get('s_cif_disorder_group',0)) > 1]
    st.deleteAtoms(disordered_atoms)
    low_occupancy = [at for at in st.atom if at.property.get('r_m_pdb_occupancy', 1) < 0.5]
    st.deleteAtoms(low_occupancy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif_csv", required=True, type=str)
    parser.add_argument("--cod_path", required=True, type=str)
    parser.add_argument("--output_path", default='.', type=str)
    parser.add_argument("--total_chunks", default=1, type=int)
    parser.add_argument("--chunk_idx", default=0, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    return parser.parse_args()

def reduce_to_minimal(st):
    mol_list = [mol.extractStructure() for mol in st.molecule]
    mol_list.sort(key=lambda x: x.atom_total)
    # group those molecules by conformers
    #grouped_mol_list = group_with_comparison(mol_list, are_conformers)
    grouped_mol_list = group_items(mol_list, are_conformers)
    # represent the structure as the counts of each type of molecule
    # and a representative structure
    st_mols = frozenset((len(grp), grp[0]) for grp in grouped_mol_list)
    counts = list(zip(*st_mols))[0]
    divisor = reduce(lambda x, y: math.gcd(x, y), counts, counts[0])
    reduced_st = create_new_structure()
    st_mols = sorted(st_mols, key=lambda x: -len(evaluate_asl(x[1], 'metals')))
    for count, mol_st in st_mols:
        for _ in range(count//divisor):
            reduced_st.extend(mol_st)
    remove_free_oxygen(reduced_st)
    remove_nitrate(reduced_st)
    return reduced_st

def remove_nitrate(st):
    patterns =('[OX1][NX3]([OX1])([OX1])','[OX1][CX3]([OX1])([OX1])', '[OX1][ClX4]([OX1])([OX1])([OX1])', '[OX1][IX3]([OX1])([OX1])','[OX1][BrX3]([OX1])([OX1])','[OX1][SX4]([OX1])([OX1])[OX1]','[OX1][PX4]([OX1])([OX1])[OX1]') 
    charges = (1,2,1,1,1,2,3)
    for smarts, charge in zip(patterns, charges):
        nitrates = {frozenset(i) for i in evaluate_smarts(st,smarts)}
        nitrate_atoms = set()
        for nit in nitrates:
            nitrate_atoms.update(nit)
        st.deleteAtoms(nitrate_atoms)
        st.property['i_m_Molecular_charge'] = st.property.get('i_m_Molecular_charge', 0) + len(nitrates)*charge

def remove_free_oxygen(st):
    free_O = evaluate_smarts(st, '[OX0]')
    st.deleteAtoms([at for at_list in free_O for at in at_list])

def remove_F2_bonds(st):
    f2 = {frozenset(i) for i in evaluate_smarts(st,'[FX2][FX2]')}
    for at1, at2 in f2:
        st.deleteBond(at1, at2)

def pb_correction(st):
    pb_atoms = evaluate_asl(st, "at.ele Pb")
    if not pb_atoms:
        return
    st_copy = st.copy()
    connect_atoms(st_copy, max_valencies=MAX_VALENCIES, cov_factor=1.1)
    for pb in pb_atoms:
        for other_atom in st_copy.atom[pb].bonded_atoms:
            if not st.areBound(pb, other_atom.index):
                st.addBond(pb, other_atom.index, 1)


def equilibrate_metals(st):
    metals = evaluate_asl(st, "metals")
    #charge_dict = defaultdict(list)
    #for metal_idx in metals:
    #    at = st.atom[metal_idx]
    #    charge_dict[at.element].append((at.formal_charge, metal_idx))
    #for charges in charge_dict.values():
    charges = []
    for metal_idx in metals:
        at = st.atom[metal_idx]
        if at.element in {'Li','Na','K','Rb','Cs'} and at.formal_charge == 1:
            continue
        elif at.element in {'Be','Mg','Ca','Sr','Ba','Zn','Cd'} and at.formal_charge == 2:
            continue
        elif at.element in {'Al','La','Lu','Pr','Nd', 'Pm', 'Eu', 'Gd', 'Tb', 'Dy','Ho', 'Er', 'Tm','Sc', 'Y'} and at.formal_charge == 3:
            continue
        charges.append((at.formal_charge, metal_idx, at.element))
    if charges:
        min_chg = min(charges)
        max_chg = max(charges)
        chg_diff = max_chg[0] - min_chg[0]
        while chg_diff > 1: 
            shift = chg_diff // 2
            charges.remove(min_chg)
            charges.remove(max_chg)
            st.atom[min_chg[1]].formal_charge += shift
            st.atom[max_chg[1]].formal_charge -= shift
            charges.append((min_chg[0] + shift, min_chg[1]))
            charges.append((max_chg[0] - shift, max_chg[1]))
            min_chg = min(charges)
            max_chg = max(charges)
            chg_diff = max_chg[0] - min_chg[0]

def is_too_large(st):
    metals = evaluate_asl(st, "metals")
    metal_mols = {st.atom[metal].molecule_number for metal in metals}
    return all(len(st.molecule[mol_idx].atom) > 250 for mol_idx in metal_mols)

def remove_common_solvents(st):
    # DCM, toluene, THF, water, pentane
    for solv in ('[ClX1][CX4]([ClX1])([#1X1])[#1X1]', '[#1X1][CX3]1[CX3]([#1X1])[CX3]([#1X1])[CX3]([CX4]([#1X1])([#1X1])[#1X1])[CX3]([#1X1])[CX3]1[#1X1]','[#1X1][CX4]1([#1X1])[OX2][CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[CX4]1([#1X1])[#1X1]', '[#1X1][OX2][#1X1]', '[#1X1][CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[CX4]([#1X1])([#1X1])[#1X1]'):
        solvs = {frozenset(i) for i in evaluate_smarts(st, solv)}
        solv_atoms = set()
        for ats in solvs:
            solv_atoms.update(ats)
        st.deleteAtoms(solv_atoms)

def extract_molecules(cod_code, cod_path, output_path):
    cif_name = get_cif_location(cod_path, cod_code)
    print(cod_code, 'start')
    try:
        st = StructureReader.read(cif_name)
    except:
        open(cod_code, 'w').close()
        return cod_code
    # If there are no metals, we aren't going to use it anyway
    if not evaluate_asl(st, "metals"):
        open(cod_code, 'w').close()
        return cod_code
    st = remove_alternate_positions(st)
    resolve_disorder(st)
    connect_atoms(st, max_valencies=MAX_VALENCIES, cov_factor=1.2)
    try:
        st = get_cell(st)
    except ValueError:
        open(cod_code, 'w').close()
        return cod_code
    clusterstruct.contract_structure(st)
    connect_atoms(st, max_valencies=MAX_VALENCIES, cov_factor=1.2)
    clusterstruct.contract_structure(st)
    connect_atoms(st, max_valencies=MAX_VALENCIES)
    if is_infinite(st) or get_close_atoms(st, 0.6):
        open(cod_code, 'w').close()
        return cod_code
    try:
        # One last way of setting the sigma skeleton
        st = clean_st(st)
    except (RuntimeError, AssertionError): 
        connect_atoms(st, max_valencies=MAX_VALENCIES)
    for bond in st.bond:
        bond.order = 1
    for at in st.atom:
        at.formal_charge = 0
    remove_common_solvents(st)
    #pb_correction(st)
    remove_F2_bonds(st)
    remove_metal_metal_bonds(st)
    # Skip things that consist solely of very large molecules that
    # we won't be taking anyway
    if is_too_large(st):
        open(cod_code, 'w').close()
        return cod_code
    st = reduce_to_minimal(st)
    try:
        mmjag_update_lewis(st, LewisModes.THOROUGH)
    except AssertionError:
        open(cod_code, 'w').close()
        return cod_code
    except TypeError:
        try:
            mmjag_update_lewis(st)
        except TypeError:
            print(f'Can\'t update lewis on {cod_code}')
            return cod_code
    fix_carbonyls(st)
    equilibrate_metals(st)
    mol_sts = [mol.extractStructure() for mol in st.molecule if len(mol.atom) > 15]
    mol_sts = uniquify_with_comparison(
        mol_sts, are_conformers, use_lewis_structure=False
    )
    for idx, mol_st in enumerate(mol_sts):
        try:
            spin = guess_spin_state(mol_st)
        except ValueError:
            open(cod_code, 'w').close()
            return cod_code
        charge = mol_st.formal_charge
        mol_st.write(os.path.join(output_path, f"{cod_code}_molecule_{idx}_{charge}_{spin}.mae"))
    print(cod_code, 'end')

def main(cif_csv, cod_path, output_path, n_cores, n_chunks, chunk_idx):
    with open(cif_csv, 'r') as fh:
        csv_reader = csv.reader(fh, delimiter='\t')
        cif_list = [line[0] for line in csv_reader]
    cif_list = cif_list[1:] # Remove header line
    done_codes = set()
    done_codes = {os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(output_path,'*.mae'))}
    if os.path.exists('skip_list.txt'):
        with open('skip_list.txt', 'r') as fh:
            done_codes.update([f.strip() for f in fh.readlines()])
    cif_list = [cif for cif in cif_list if cif not in done_codes]
    chunks_to_process = np.array_split(cif_list, n_chunks)
    chunk = chunks_to_process[chunk_idx]
    print(len(chunk))
    fxn = partial(extract_molecules, cod_path=cod_path, output_path=output_path)
    with mp.Pool(n_cores) as pool:
        skip_list = list(tqdm(pool.imap(fxn, chunk), total=len(chunk)))


if __name__ == "__main__":
    args = parse_args()
    main(args.cif_csv, args.cod_path, args.output_path, args.num_workers, args.total_chunks, args.chunk_idx)
