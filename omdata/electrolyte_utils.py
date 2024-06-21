from typing import Any, Dict, List, Set, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import Descriptors

from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor


def info_from_smiles(
    smiles: Dict[str, str] | List[str] | Set[str]
    ) -> Dict[str, Any]:
    """
    Generate information about a molecule from SMILES.

    Args:
        smiles (Dict[str, str] | List[str] | Set[str]): Collection of SMILES, either as a dict {key: smiles},
            or as a list/set.

    Returns:
        data (Dict[str, Any]): Map between SMILES and their properties (charge, spin, number of atoms, etc.)
    """

    data = dict()

    if isinstance(smiles, dict):
        names_smiles = smiles.items()
    else:
        names_smiles = [(s, s) for s in smiles]

    for (name, this_smiles) in names_smiles:
        rdkit_mol = Chem.MolFromSmiles(this_smiles)

        if rdkit_mol is None:
            print("PROBLEM", name, this_smiles)

        Chem.AddHs(rdkit_mol)
        
        charge = Chem.GetFormalCharge(rdkit_mol)
        spin = Descriptors.NumRadicalElectrons(rdkit_mol) + 1

        num_atoms = rdkit_mol.GetNumAtoms()
        num_heavy_atoms = rdkit_mol.GetNumHeavyAtoms()

        data[name] = {
            "smiles": this_smiles, "charge": charge, "spin": spin, "num_atoms": num_atoms,
            "num_heavy_atoms": num_heavy_atoms, "rdkit_mol": rdkit_mol,
        }
    
    return data


def validate_structure(species: List[str], coords: Any, tolerance: float = 0.9) -> bool:
    """
    Check if any atoms in the molecule are too close together

    Args:
        species (List[str]): List of atomic elements with length N, where N is the number of atoms
        Coords (Any): Atomic positions as an Nx3 matrix, where N is the number of atoms. Should be
            of type np.ndarray, but might be of an error type

    Returns:
        bool. Is this molecule valid?
    """

    try:
        pmg_mol = Molecule(species, coords)
    except:
        return False

    return pmg_mol.is_valid(tol=tolerance)