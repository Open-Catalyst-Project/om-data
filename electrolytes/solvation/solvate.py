import copy
import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple

# For molecule representations
from ase import Atoms
from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from sella import Sella

# For solvation shell formation
from architector import (build_complex,
                         view_structures,
                         convert_io_molecule)
import architector.io_arch_dock as io_arch_dock
from architector.io_molecule import Molecule as ArchMol

from omdata.electrolyte_utils import info_from_smiles



metals = [
    "[Li+]", "[Na+]", "[K+]", "[Cs+]", "[Ti+]", "[Cu+]", "[Ag+]", "O=[V+]=O", "[Rb+]", "[Ca+2]", "[Mg+2]", "[Zn+2]",
    "[Cu+2]", "[Ni+2]", "[Pt+2]", "[Co+2]", "[Pd+2]", "[Ag+2]", "[Mn+2]", "[Hg+2]", "[Cd+2]", "[Yb+2]", "[Sn+2]",
    "[Pb+2]", "[Eu+2]", "[Sm+2]", "[Ra+2]", "[Cr+2]", "[Fe+2]", "O=[V+2]", "[V+2]", "[Ba+2]", "[Sr+2]", "[Al+3]",
    "[Cr+3]", "[V+3]", "[Ce+3]", "[Ce+4]", "[Fe+3]", "[In+3]", "[Tl+3]", "[Y+3]", "[La+3]", "[Pr+3]", "[Nd+3]",
    "[Sm+3]", "[Eu+3]", "[Gd+3]", "[Tb+3]", "[Dy+3]", "[Er+3]", "[Tm+3]", "[Lu+3]", "[Hf+4]","[Zr+4]"
]

cations = [
    "[OH3+]", "[NH4+]", "CCCC[N+]1(CCCC1)C", "CCN1C=C[N+](=C1)C", "CCC[N+]1(C)CCCC1", "CCC[N+]1(CCCCC1)C",
    "CC[N+](C)(CC)CCOC", "CCCC[P+](CCCC)(CCCC)CCCC", "CCCC[N+]1(CCCC1)CCC", "COCC[NH2+]CCOC", "CC(=O)[NH2+]C",
    "CC(COC)[NH3+]", "C[N+](C)(C)CCO", "CC1(CCCC(N1[O+])(C)C)C", "[Be+2]", "C[N+]1=CC=C(C=C1)C2=CC=[N+](C=C2)C",
]

anions = [
    "F[Al-](F)(F)F", "[AlH4-]", "[B-]1(OC(=O)C(=O)O1)(F)F", "[B-]12(OC(=O)C(=O)O1)OC(=O)C(=O)O2", "[B-](F)(F)(F)F",
    "[BH4-]", "[CH-]1234[BH]5%12%13[BH]1%10%11[BH]289[BH]367[BH]145[BH]6%14%15[BH]78%16[BH]9%10%17[BH]%11%12%18[BH]1%13%14[BH-]%15%16%17%18",
    "[BH-]1234[BH]5%12%13[BH]1%10%11[BH]289[BH]367[BH]145[BH]6%14%15[BH]78%16[BH]9%10%17[BH]%11%12%18[BH]1%13%14[BH-]%15%16%17%18",
    "C[O-]", "CC[O-]", "CC(C)[O-]", "[O-]CC[O-]", "CCOC([O-])C(F)(F)F", "[Br-]", "C(F)(F)(F)S(=O)(=O)[O-]",
    "C(=O)(O)[O-]", "CC(=O)[O-]", "C(=O)([O-])[O-]", "C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F",
    "C[Si](C)(C)[N-][Si](C)(C)C", "CC1(CCCC(N1[O-])(C)C)C", "[Cl-]", "[O-]Cl(=O)(=O)=O", "[N-](S(=O)(=O)F)S(=O)(=O)F",
    "[O-]P(=O)(F)F", "F[As-](F)(F)(F)(F)F", "F[P-](F)(F)(F)(F)F", "[OH-]", "[F-]", "[I-]", "[N+](=O)([O-])[O-]",
    "[O-]P(=O)([O-])[O-]", "C1=C(C(=O)C=C(C1=O)[O-])[O-]", "[O-]S(=O)(=O)[O-]"
]

neutrals = [
    "C1=CC=C2C(=C1)C(=O)C3=CC=CC=C3C2=O", "C(=O)(N)N", "CC(=O)C", "CC#N", "O=C(N)C", "CCO", "CS(=O)C",
    "C1C(OC(=O)O1)F", "C1COC(=O)O1", "CC(=O)NC", "CC(C)O", "O=S(=O)(OCC)C", "COCCOC", "CC(COC)N", "CCOC(=O)C(F)(F)F",
    "O=C1OCCC1", "CC1COC(=O)O1", "CCCC#N", "C1CCOC1", "O=C(OCC)C", "C(CO)O", "C1CCS(=O)(=O)C1", "C1COS(=O)(=O)O1",
    "COCCOCCOC", "COC(=O)OC", "CCOC(=O)OC", "COCCNCCOC", "COP(=O)(OC)OC", "O=P(OCC)(OCC)OCC", "C1=CC(=O)C=CC1=O",
    "C1=C(C(=O)C=C(C1=O)O)O", "C1=CC=CC=C1", "C1=CC=C(C=C1)[N+](=O)[O-]", "C(C(C(F)F)(F)F)OC(C(F)F)(F)F", "CC(COC)N",
    "O", "CC1(CCCC(N1[O])(C)C)C",
]


def generate_solvated_mol(
    mol: Molecule | Atoms,
    charge: int,
    spin_multiplicity: int,
    species_smiles: List[str],
    architector_params: Dict = dict()
) -> Atoms:
    """
    Generate a solvation shell around a molecule using Architector.

    Args:
        mol (Molecule | Atoms): molecule to be solvated
        charge (int): charge of the core molecule
        spin_multiplicity (int): spin multiplicity of the core molecule
        species_smiles (List[str]): SMILES for each atom to be placed around the core molecule
        architector_params (Dict): parameters for Architector solvation shell generation

    Returns:
        shell (Atoms): molecule with solvation shell
    """

    # Convert to Architector internal molecule representation
    if isinstance(mol, Molecule):
        mol = AseAtomsAdaptor.get_atoms(mol)

    atoms = convert_io_molecule(mol)

    # Set charge and number of unpaired electrons for xTB
    atoms.charge = charge
    atoms.uhf = spin_multiplicity - 1

    architector_params["species_list"] = species_smiles

    binding = io_arch_dock.add_non_covbound_species(atoms, parameters=architector_params)
    shell = binding[0].ase_atoms
    
    return shell


def generate_full_solvation_shell(
    mol: Molecule | Atoms,
    charge: int,
    spin_multiplicity: int,
    solvent: str,
    max_atom_budget: int = 150,
    architector_params: Dict = dict()
) -> Atoms:
    """
    Generate a solvation shell comprised of a single solvent around a molecule

    Args:
        mol (Molecule | Atoms): molecule to be solvated
        charge (int): charge of the core molecule
        spin_multiplicity (int): spin multiplicity of the core molecule
        solvent (str): SMILES for the solvent to surround the central molecule
        architector_params (Dict): parameters for Architector solvation shell generation

    Returns:
        shell (Atoms): molecule with solvation shell
    """

    solvent_info = info_from_smiles([solvent])[solvent]

    if solvent_info["charge"] != 0:
        raise ValueError("generate_full_solvation_shell will only work for neutral solvents! Provided charge for"
                         f"{solvent}: {solvent_info['charge']}.")

    this_max_atoms = round(random.gauss(mu=50 + len(mol), sigma=40))
    if this_max_atoms > max_atom_budget:
        this_max_atoms = max_atom_budget
    if this_max_atoms < len(mol) + 20:
        this_max_atoms = len(mol) + 20
    
    budget = this_max_atoms - len(mol)
    num_solvent_mols = round(budget / solvent_info["num_atoms"])
    if num_solvent_mols < 1:
        num_solvent_mols = 1

    species_smiles = [solvent] * num_solvent_mols

    shell = generate_solvated_mol(mol, charge, spin_multiplicity, species_smiles, architector_params=architector_params)
    
    return shell
    

def generate_random_solvated_mol(
    mol: Molecule | Atoms,
    charge: int,
    spin_multiplicity: int,
    cations: Dict[str, Dict],
    anions: Dict[str, Dict],
    neutrals: Dict[str, Dict],
    max_atom_budget: int = 150,
    max_trials: int = 25,
    architector_params: Dict = dict()
) -> Atoms:
    """
    Generate (quasi)random solvated molecule using Architector.

    Cations, anions, and neutral species (e.g. solvents, additives) are placed around a central molecule.

    Args:
        mol (Molecule | Atoms): molecule to be solvated
        charge (int): charge of the core molecule
        spin_multiplicity (int): spin multiplicity of the core molecule
        cations (Dict[str, int]): Map <SMILES>:<number of atoms> for potential solvating cations
        anions (Dict[str, int]): Map <SMILES>:<number of atoms> for potential solvating anions
        neutrals (Dict[str, int]): Map <SMILES>:<number of atoms> for potential solvating neutral molecules
        max_atom_budget (int): Maximum number of atoms that can be in a complex
        max_trials (int): Maximum number of attempts adding a solvating molecule
        architector_params (Dict): parameters for Architector solvation shell generation

    Returns:
        shell (Atoms): molecule with solvation shell
    
    """

    # Select cap for number of atoms in this solvation shell
    # For now, using a normal (Gaussian distribution) with mean at (50 + len(mol)) atoms and stdev of 40
    # We then turn this continuous selection into an integer and make sure that it's within some reasonable bounds
    this_max_atoms = round(random.gauss(mu=50 + len(mol), sigma=40))
    lower_bound_maxatoms = len(mol) + 15
    if this_max_atoms < lower_bound_maxatoms:
        this_max_atoms = lower_bound_maxatoms
    elif this_max_atoms > max_atom_budget:
        this_max_atoms = max_atom_budget

    combined_data = copy.deepcopy(neutrals)
    combined_data.update(metals)
    combined_data.update(cations)
    combined_data.update(anions)
    
    species_smiles = list()
    total_num_atoms = len(mol)
    total_charge = charge
    for trial in range(max_trials):
        budget = this_max_atoms - total_num_atoms
        if budget < 1:
            break

        # Assign weights based on number of atoms
        # Don't allow cations be added to a cationic complex or anions be added to an anionic complex
        choice_smiles = list()
        choice_weights = list()

        for nsmiles, ndata in neutrals.items():
            choice_smiles.append(nsmiles)
            choice_weights.append(1 / ndata["num_atoms"])

        if total_charge < 0:
            for csmiles, cdata in cations.items():
                choice_smiles.append(csmiles)
                choice_weights.append(1 / cdata["num_atoms"])
        elif total_charge > 0:
            for asmiles, adata in anions.items():
                choice_smiles.append(asmiles)
                choice_weights.append(1 / adata["num_atoms"])
                    
        choice = random.choices(choice_smiles, weights=choice_weights, k=1)[0]
        choice_num_atoms = combined_data[choice]["num_atoms"]
        choice_charge = combined_data[choice]["charge"]
        
        if total_num_atoms + choice_num_atoms <= this_max_atoms:
            species_smiles.append(choice)
            total_num_atoms += choice_num_atoms
            total_charge += choice_charge
            
    shell = generate_solvated_mol(mol, charge, spin_multiplicity, species_smiles, architector_params=architector_params)
    return shell


def generate_random_dimers(
    mol: Molecule | Atoms,
    charge: int,
    spin_multiplicity: int,
    candidates: Dict[str, Dict],
    max_atom_budget: int = 200,
    num_selections: int = 5,
    architector_params: Dict = dict()
) -> Atoms:
    """
    Generate (quasi)random solvated molecule using Architector.

    Cations, anions, and neutral species (e.g. solvents, additives) are placed around a central molecule.

    Args:
        mol (Molecule | Atoms): molecule to be solvated
        charge (int): charge of the core molecule
        spin_multiplicity (int): spin multiplicity of the core molecule
        cations (Dict[str, int]): Map <SMILES>:<number of atoms> for potential solvating cations
        anions (Dict[str, int]): Map <SMILES>:<number of atoms> for potential solvating anions
        neutrals (Dict[str, int]): Map <SMILES>:<number of atoms> for potential solvating neutral molecules
        max_atom_budget (int): Maximum number of atoms that can be in a complex
        max_trials (int): Maximum number of attempts adding a solvating molecule
        architector_params (Dict): parameters for Architector solvation shell generation

    Returns:
        shell (Atoms): molecule with solvation shell
    
    """

    # Since we're only adding one molecule, don't weight.
    # Just make sure that the total size isn't too large

    this_size = len(mol)
    budget = max_atom_budget - this_size

    real_candidates_names = [k for k, v in candidates.items() if v["num_atoms"] <= budget]

    if len(real_candidates_names) == 0:
        return list()
    elif len(real_candidates_names) < num_selections:
        choices = real_candidates_names
    else:
        choices = random.sample(real_candidates_names, k=num_selections)

    complexes = list()
    for candidate in choices:
        complex = generate_solvated_mol(
            mol,
            charge,
            spin_multiplicity,
            [candidates[candidate]["smiles"]],
            architector_params
        )
        complexes.append(complex)

    return complexes