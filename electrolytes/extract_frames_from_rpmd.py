from schrodinger.structure import StructureReader, StructureWriter, Structure
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger.application.jaguar.utils import mmjag_update_lewis
from schrodinger.application.jaguar.utils import get_stoichiometry_string
from collections import Counter
import sys
import os
import pandas as pd
import re

#DIR= '/checkpoint/levineds/rpmd'
#CSV_NAME = 'rpmd_elytes.csv'
DIR= '/checkpoint/levineds/droplet'
CSV_NAME = 'omm-elytes.csv'

def copy_bonding(st1: Structure, st2: Structure):
    """
    Impose the bonding and formal charges of st1 onto st2
    The two structures must have the same
    number of atoms or a ValueError is raised.

    """

    for bond in st1.bond:
        bond2 = st2.getBond(*bond.atom)
        if bond2 is None:
            st2.addBond(bond.atom1.index, bond.atom2.index, bond.order)
        else:
            bond2.order = bond.order

    for iat, jat in zip(st1.atom, st2.atom):
        jat.formal_charge = iat.formal_charge

def assign_charge_to_res(st: Structure, row_idx:int)->None:
    """
    Assign charges to the residues in the Structure using the
    information in the csv file.

    We just dump the formal charge on the first atom of the residue
    because only the total residue's charge matters, not any given atom.

    :param st: the structure to label with charges
    :param row_idx: the row index of the csv being processed
    """
    row = pd.read_csv(CSV_NAME).iloc[row_idx].dropna()
    chg_dct = {}
    for key, val in row.items():
        if key.startswith('cation') and len(key) == 7:
            m = re.match(r'(.*)(\+\d?)', val)
            charge = 1 if m.group(2) == '+' else int(m.group(2))
            chg_dct[m.group(1)] = charge
        elif key.startswith('anion') and len(key) == 6:
            m = re.match(r'(.*)(\-\d?)', val)
            charge = -1 if m.group(2) == '-' else int(m.group(2))
            stoich = m.group(1)
            if stoich == 'OH':
                stoich = 'HO'
            elif stoich == 'C4BO8':
                stoich = 'BC4O8'
            elif stoich == 'C4H6F3O2':
                stoich = 'C4F3H6O2'
            chg_dct[stoich] = charge
    for res in st.residue:
        stoich = get_stoichiometry_string([at.element for at in res.atom])
        res.atom[1].formal_charge = chg_dct[stoich]
    assert st.formal_charge == 0

def main(row_idx, use_pbc, n_struct=1):

    out_name = f'{DIR}/{row_idx}/frames.maegz'
    if os.path.exists(out_name):
        return
    st_list = []
    for j in range(1, n_struct + 1):
        print(f'reading {j}')
        if use_pbc:
            fname = f'{DIR}/{row_idx}/rpmd_0_bead_{j}.pdb'
        else:
            fname = f'{DIR}/{row_idx}/trajectory_0.pdb'
        if not os.path.exists(fname):
            return
        # Drop the first frame
        st_list.extend(StructureReader(fname, index=2))
    print('done reading')
    counter = Counter()
    for res in st_list[0].residue:
        counter[res.pdbres] += 1
    res_to_chain = {}
    for res, count in counter.items():
        res_to_chain[res] = 'A' if count < 20 else 'B'
    if 'A' not in res_to_chain.values():
        res_to_chain = {k:'A' for k in res_to_chain}
    # I am making a hard assumption that all the solute atoms are numbered first
    # and that no chain B things have charge. I think that's right...
    for idx, st in enumerate(st_list):
        for res in st.residue:
            res.chain = res_to_chain[res.pdbres]
        if idx == 0:
            charged_ions = st.chain['A'].extractStructure()
            assign_charge_to_res(charged_ions, row_idx)
            ep_atoms = evaluate_asl(st, 'atom.ptype " OM "')
        if use_pbc:
            st.pbc.applyToStructure(st)
        copy_bonding(charged_ions, st)
        st.deleteAtoms(ep_atoms)
    with StructureWriter(out_name) as writer:
        writer.extend(st_list)

if __name__=='__main__':
    row_idx = int(sys.argv[1])
    # for RPMD:
    #main(row_idx, use_pbc=True, n_struct=32)
    # for droplet:
    main(row_idx, use_pbc=False, n_struct=1)
