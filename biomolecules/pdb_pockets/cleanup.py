
"""
    Cap cleaved cysteines with H
    Find ACE in ligand chain and remove ACE but first find the adjacent N
    at=next((at for at in res.getCarbonylCarbon().bonded_atoms if at.element == 'N'))
    verify that this N has 4 bonds and if so delete the ACE, increase charge to 1 and 
    call build.add_hydrogens(st, atom_list=[at]) on it
    What to do with NME? 
"""
import argparse
import glob
import os
from tqdm import tqdm
from schrodinger.structure import StructureReader, StructureWriter, Structure
from schrodinger.structutils import analyze, build

def rename_coord_chain(st):
    chain_names = [ch.name for ch in st.chain]
    if 'l' not in chain_names and 'c' in chain_names:
        lig = []
        if len(st.chain['c'].getAtomList()) == 1:
            lig = st.chain['c'].getAtomList()
        if not lig:
            lig = analyze.evaluate_asl(st, 'chain c and metals')
        if not lig:
            lig = analyze.evaluate_asl(st, 'chain c and res D8U')
        if not lig:
            lig = analyze.evaluate_asl(st, 'chain c and atom.ato >=18')
        if not lig:
            lig = analyze.evaluate_asl(st, 'chain c and atom.ele F,Cl,Br,I')
        for at in lig:
            st.atom[at].chain = 'l'
        if lig:
            return True
    return False

def fix_disrupted_disulfides(st):
    broken_cysteines = analyze.evaluate_asl(st, 'atom.pt SG and atom.att 1')
    if broken_cysteines:
        build.add_hydrogens(st, atom_list=broken_cysteines)
        return True
    return False

def remove_ligand_ace_cap(st):
    ats_to_delete = []
    N_to_protonate = []
    for res in st.chain['l'].residue:
        if res.pdbres.strip() == 'ACE' and len(st.chain['l'].getAtomList()) != 6:
            capped_N = next((at for at in res.getCarbonylCarbon().bonded_atoms if at.element == 'N'))
            ats_to_delete.extend(res.getAtomList())
            N_to_protonate.append(capped_N)
    for at in N_to_protonate:
        st.atom[at].formal_charge += 1
    if N_to_protonate:
        st.deleteAtoms(ats_to_delete)
        build.add_hydrogens(st, atom_list = N_to_protonate)
        return True
    return False

def remove_ligand_nma_cap(st):
    ats_to_delete = []
    for res in st.chain['l'].residue:
        if res.pdbres.strip() == 'NMA' and len(res.getAtomList()) == 6:
            n_at = res.getBackboneNitrogen()
            if n_at is None:
                raise RuntimeError
            ats_to_delete.extend(res.getAtomList())
            ats_to_delete.remove(n_at.index)
            n_at.element = 'O'
            n_at.formal_charge = -1
            for b_at in n_at.bonded_atoms:
                adj_res = b_at.getResidue()
                if adj_res != res:
                    break
            n_at.resnum = adj_res.resnum
            n_at.inscode = adj_res.inscode
            n_at.pdbres = adj_res.pdbres
            n_at.pdbname = ''
    st.deleteAtoms(ats_to_delete)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--prefix", default='')
    return parser.parse_args()

def main():
    args = parse_args()
    for fname in tqdm(glob.glob(os.path.join(args.output_path, f'{args.prefix}*.pdb'))):
        st = StructureReader.read(fname)
        change_made = rename_coord_chain(st)
        change_made = fix_disrupted_disulfides(st) or change_made
        try:
            change_made = remove_ligand_ace_cap(st) or change_made
        except:
            print(fname)
            raise
        try:
            change_made = remove_ligand_nma_cap(st) or change_made
        except:
            print(fname)
            raise
        if change_made:
            st = build.reorder_protein_atoms_by_sequence(st)
            st.write(fname)

if __name__ == "__main__":
    main()
