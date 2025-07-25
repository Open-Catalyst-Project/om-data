import sys
import io
import random
import numpy as np

from architector.io_molecule import Molecule
from rdkit.Chem import MolFromSmiles, AddHs, RWMol, Mol, BondType
from rdkit.Chem import MolFromMolBlock, MolToXYZBlock

from ase import Atom
from ase.io import read, write
from openbabel import pybel
from tempfile import NamedTemporaryFile

class Chain(Molecule):
    '''
    Chain is an Architector Molecule with additional attributes:
        repeat_units :      list of smiles representation for each repeat unit
        extra_units:        list of smiles representation for each extra molecule
        n_extra:            list of integers corresponding to number of a given extra molecule
        n_repeats :         list of integers corresponding to number of repeat units (same order as repeat_units)
        ends :              list of indices corresponding to chain end indices
        end_to_end:         if a single chain, the end to end distance
        rdkit_mol :         RDKit Mol
        extra_rdkit_mol :   RDkit Mol for each extra molecule
        
        remove_extra() : remove any extra molecules to return a clean Chain
    '''
    def __init__(self, structure, repeat_smiles, extra_smiles=[]):
        if ".pdb" in structure:
            structure = read(structure)
        super().__init__(structure)
        
        self.repeat_units = repeat_smiles
        self.extra_units = extra_smiles

        self.rdkit_mol = get_rdkit_mol(self.ase_atoms)
        self.extra_rdkit_mol = self.get_extra_rdkit_mol()
        self.n_extra = self.get_nextra()
        
        self.ends, self.end_to_end = self.assign_chain_ends()
        self.n_repeats = self.get_nrepeats()

        self.charge = self.ase_atoms.info.get("charge", 0)
    
        self.create_mol_graph()
        
    def assign_chain_ends(self):
        ends = []
        end_to_end = None
        res_labels = self.ase_atoms.arrays["residuenames"] 
        
        # dependent on LLNL implementation
        tu0_indices = np.where(res_labels == "TU0")[0]
        tu1_indices = np.where(res_labels == "TU1")[0]
        h_end_indices = np.concatenate([tu0_indices, tu1_indices]).tolist()
        
        for h_end_idx in h_end_indices:
            atom = self.rdkit_mol.GetAtomWithIdx(h_end_idx)
            end_idx = atom.GetNeighbors()[0].GetIdx()
            ends.append(end_idx)
        
        # define end to end distance if a single chain
        if len(h_end_indices) == 2:
            atoms = self.ase_atoms.get_positions()
            start = atoms[ends[0]]
            end = atoms[ends[-1]]
            end_to_end = np.linalg.norm(end - start)
        
        return ends, end_to_end

    def get_nrepeats(self):
        n_repeats = []
        n_extra_total = 0
        for i, extra in enumerate(self.extra_rdkit_mol):
            n_extra_total += extra.GetNumAtoms() * self.n_extra[i]  
        for repeat in self.repeat_units:
            repeat_mol = AddHs(MolFromSmiles(repeat))
            chain_mol = self.rdkit_mol
            
            natoms_repeat = repeat_mol.GetNumAtoms() - 2 # subtract extra atoms
            natoms_chain = chain_mol.GetNumAtoms() - len(self.ends) - n_extra_total
            n_repeats.append(round(natoms_chain / natoms_repeat)) # round for small defects (+/- H)
        return n_repeats
    
    def remove_extra(self):
        """
        remove the extra molecules (i.e., solvent) to yield a clean chain
        """
        chain_mol = RWMol(self.rdkit_mol)
        
        atoms_to_remove = set()
        for extra in self.extra_rdkit_mol:
            matches = chain_mol.GetSubstructMatches(extra)
            for match in matches:
                atoms_to_remove.update(match)

        atoms_to_remove = sorted(atoms_to_remove, reverse=True)

        # Map old atom indices to new after deletion
        total_atoms = chain_mol.GetNumAtoms()
        keep_mask = np.ones(total_atoms, dtype=bool)
        keep_mask[list(atoms_to_remove)] = False

        for idx in atoms_to_remove:
            chain_mol.RemoveAtom(idx)

        # Create new ASE Atoms object
        xyz_block = MolToXYZBlock(chain_mol)
        new_ase_atoms = read(io.StringIO(xyz_block), format='xyz')

        #  Add back residue names, needed for chain_ends
        if "residuenames" in self.ase_atoms.arrays:
            original_residues = self.ase_atoms.arrays["residuenames"]
            kept_residues = original_residues[keep_mask]
            new_ase_atoms.set_array("residuenames", kept_residues)

        new_chain = Chain(new_ase_atoms, self.repeat_units)

        assert new_chain.n_repeats == self.n_repeats
        assert new_chain.ends == self.ends

        return new_chain
    
    def get_nextra(self):
        """
        list of number of extra molecules in order of self.extra_units
        """
        n_extra = []
        chain_mol = remove_bond_order(self.rdkit_mol)
        for extra in self.extra_rdkit_mol:
            extra = remove_bond_order(extra)
            matches = chain_mol.GetSubstructMatches(extra)
            n_extra.append(len(matches))
        
        return n_extra
    
    def get_extra_rdkit_mol(self):
        rdkit_mols = []
        for extra in self.extra_units:
            rdkit_mols.append(AddHs(MolFromSmiles(extra)))
        return rdkit_mols
    
def get_rdkit_mol(ase_atoms):
    
    with NamedTemporaryFile(suffix=".xyz") as tmp:
        write(tmp.name, ase_atoms, format="xyz")
        obmol = next(pybel.readfile("xyz", tmp.name))
        
    obmol.OBMol.ConnectTheDots()
    obmol.OBMol.PerceiveBondOrders()
    molblock = obmol.write("mol")
    
    mol = MolFromMolBlock(molblock, sanitize=False)
    
    ase_atoms = [atom.number for atom in ase_atoms]
    rdkit_atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    assert ase_atoms == rdkit_atoms, "Atom types do not match between ASE and RDKit"
    return mol

def remove_bond_order(query_mol):
    copy_mol = Mol(query_mol)
    for bond in copy_mol.GetBonds():
        bond.SetBondType(BondType.SINGLE)
        bond.SetIsAromatic(False)
        # Remove query constraints on bond order if any
        bond.SetProp('bondType', '')  
    return copy_mol

# for homolytic and heterolytic dissociation      
def get_bonds_to_break(chain, max_H_bonds=1, max_other_bonds=4, center_radius=5.0, fraction_ring=0.25):
    clean_chain = chain.remove_extra()
    mol = clean_chain.rdkit_mol

    positions = clean_chain.ase_atoms.get_positions()
    center = clean_chain.ase_atoms.get_center_of_mass()
    
    def within_radius(idx):
        return np.linalg.norm(positions[idx] - center) <= center_radius

    def get_local_density_score(a1, a2, radius=center_radius):
        midpoint = 0.5 * (positions[a1] + positions[a2])
        distances = np.linalg.norm(positions - midpoint, axis=1)
        # Exclude the two atoms in the bond
        mask = np.ones(len(positions), dtype=bool)
        mask[[a1, a2]] = False
        return np.count_nonzero((distances < radius) & mask)
        
    h_bonds = []
    other_bonds = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if not (within_radius(a1) or within_radius(a2)):
            continue

        z1, z2 = mol.GetAtomWithIdx(a1).GetAtomicNum(), mol.GetAtomWithIdx(a2).GetAtomicNum()
        bond_tuple = tuple(sorted((a1, a2)))

        skip = False
        for idx, z in [(a1, z1), (a2, z2)]:
            if z == 8:  
                neighbors = mol.GetAtomWithIdx(idx).GetNeighbors()
                other_bonded_neighbors = [n.GetIdx() for n in neighbors if n.GetIdx() not in (a1, a2)]
                if len(other_bonded_neighbors) == 0:
                    skip = True # avoid making triplet oxygen
        if skip:
            continue
        density_score = get_local_density_score(a1, a2)
        bond_data = (density_score, bond_tuple)

        atomic_nums = {z1, z2}
        if 1 in atomic_nums:  # H bond
            h_bonds.append(bond_data)
        else:
            other_bonds.append(bond_data)
        
    h_bonds.sort(reverse=True)
    other_bonds.sort(reverse=True)
    top_h = h_bonds[:10]  
    top_other = other_bonds[:10]

    selected_h = random.sample(top_h, min(max_H_bonds, len(top_h)))
    selected_other = random.sample(top_other, min(max_other_bonds, len(top_other)))

    # Extract just the bond tuples 
    selected_h = [b[1] for b in selected_h]
    ring_bonds = [b for b in other_bonds if mol.GetBondBetweenAtoms(*b[1]).IsInRing()]
    non_ring_bonds = [b for b in other_bonds if not mol.GetBondBetweenAtoms(*b[1]).IsInRing()]

    if len(ring_bonds) > max_other_bonds:
        # Prevent over-representation of ring bonds
        max_ring = int(fraction_ring * max_other_bonds)
        max_non_ring = max_other_bonds - max_ring
        selected_other = non_ring_bonds[:max_non_ring] + ring_bonds[:max_ring]

    selected_other = [b[1] for b in selected_other]
    
    return selected_h + selected_other


