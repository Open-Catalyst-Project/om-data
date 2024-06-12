from typing import Any, Dict, List, Set, Tuple

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.babel import BabelMolAdaptor


def info_from_smiles(
    smiles: Dict[str, str] | List[str] | Set[str]
    ) -> Dict[str, Any]:
    """
    Generate Calculate the number of atoms in a molecule from SMILES.

    Args:
        smiles (Dict[str, str] | List[str] | Set[str]): Collection of SMILES, either as a dict {key: smiles},
            or as a list/set.

    Returns:
        num_atoms (Dict[str, Any]): Map between SMILES and their size in terms of total number of atoms
    """

    data = dict()

    if isinstance(smiles, dict):
        names_smiles = smiles.items()
    else:
        names_smiles = [(s, s) for s in smiles]

    for (name, this_smiles) in names_smiles:
        mol = BabelMolAdaptor.from_str(this_smiles, file_format="smi")
        charge = mol.pybel_mol.charge
        spin = mol.pybel_mol.spin
        pmg_mol = mol.pymatgen_mol
        pmg_mol.set_charge_and_spin(charge, spin)
        
        ase_atoms = AseAtomsAdaptor.get_atoms(pmg_mol)
        ase_atoms.charge = charge
        ase_atoms.uhf = spin - 1

        rdkit_mol = Chem.MolFromSmiles(this_smiles)
        
        num_atoms = len(pmg_mol)
        num_heavy_atoms = len([s for s in pmg_mol.species if str(s) != "H"])

        data[name] = {
            "smiles": this_smiles, "charge": charge, "num_atoms": num_atoms, "num_heavy_atoms": num_heavy_atoms,
            "pmg_mol": pmg_mol, "rdkit_mol": rdkit_mol, "ase_atoms": ase_atoms
        }
    
    return data
