import argparse
import copy
import glob
from math import ceil
import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple

# For molecule representations
from ase import Atoms
from ase.io import write
from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

# from sella import Sella

# For solvation shell formation
from architector import convert_io_molecule
import architector.io_arch_dock as io_arch_dock

from omdata.electrolyte_utils import info_from_smiles, validate_structure



metals = [
    "[Li+]", "[Na+]", "[K+]", "[Cs+]", "[Ti+]", "[Cu+]", "[Ag+]", "O=[V+]=O", "[Ca+2]", "[Mg+2]", "[Zn+2]",
    "[Cu+2]", "[Ni+2]", "[Pt+2]", "[Co+2]", "[Pd+2]", "[Ag+2]", "[Mn+2]", "[Hg+2]", "[Cd+2]", "[Yb+2]", "[Sn+2]",
    "[Pb+2]", "[Eu+2]", "[Sm+2]", "[Ra+2]", "[Cr+2]", "[Fe+2]", "O=[V+2]", "[V+2]", "[Ba+2]", "[Sr+2]", "[Ti+2]", "[Al+3]",
    "[Cr+3]", "[V+3]", "[Ce+3]", "[Fe+3]", "[In+3]", "[Tl+3]", "[Y+3]", "[La+3]", "[Pr+3]", "[Nd+3]",
    "[Sm+3]", "[Eu+3]", "[Gd+3]", "[Tb+3]", "[Dy+3]", "[Er+3]", "[Tm+3]", "[Lu+3]", "[Ti+3]", "[Hf+4]", "[Zr+4]", "[Ce+4]",
]

other_cations = [
    "[OH3+]", "[NH4+]", "CCCC[N+]1(CCCC1)C", "CCN1C=C[N+](=C1)C", "CCC[N+]1(C)CCCC1", "CCC[N+]1(CCCCC1)C",
    "CC[N+](C)(CC)CCOC", "CCCC[P+](CCCC)(CCCC)CCCC", "CCCC[N+]1(CCCC1)CCC", "COCC[NH2+]CCOC", "CC(=O)[NH2+]C",
    "CC(COC)[NH3+]", "C[N+](C)(C)CCO", "CC1(CCCC(N1[O+])(C)C)C", "[Be+2]", "C[N+]1=CC=C(C=C1)C2=CC=[N+](C=C2)C",
]

# NOTE: removed AlH4- and BH4- because hydrogens were flying off in Architector
anions = [
    "F[Al-](F)(F)F", "[B-]1(OC(=O)C(=O)O1)(F)F", "[B-]12(OC(=O)C(=O)O1)OC(=O)C(=O)O2", "[B-](F)(F)(F)F",
    "C[O-]", "CC[O-]", "CC(C)[O-]", "[O-]CC[O-]", "CCOC([O-])C(F)(F)F", "[Br-]", "C(F)(F)(F)S(=O)(=O)[O-]",
    "C(=O)(O)[O-]", "CC(=O)[O-]", "C(=O)([O-])[O-]", "C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F",
    "C[Si](C)(C)[N-][Si](C)(C)C", "CC1(CCCC(N1[O-])(C)C)C", "[Cl-]", "[O-]Cl(=O)(=O)=O", "[N-](S(=O)(=O)F)S(=O)(=O)F",
    "F[P-](F)(F)(F)(F)F", "[OH-]", "[F-]", "[I-]", "[N+](=O)([O-])[O-]",
    "[O-]P(=O)([O-])[O-]", "C1=C(C(=O)C=C(C1=O)[O-])[O-]", "[O-]S(=O)(=O)[O-]"
]

neutrals = [
    "C1=CC=C2C(=C1)C(=O)C3=CC=CC=C3C2=O", "C(=O)(N)N", "CC(=O)C", "CC#N", "CCO", "CS(=O)C",
    "C1C(OC(=O)O1)F", "C1COC(=O)O1", "CC(=O)NC", "CC(C)O", "O=S(=O)(OCC)C", "COCCOC", "CC(COC)N", "CCOC(=O)C(F)(F)F",
    "O=C1OCCC1", "CC1COC(=O)O1", "CCCC#N", "C1CCOC1", "O=C(OCC)C", "C1CCS(=O)(=O)C1", "C1COS(=O)(=O)O1",
    "COCCOCCOC", "COC(=O)OC", "CCOC(=O)OC", "COCCNCCOC", "COP(=O)(OC)OC", "O=P(OCC)(OCC)OCC", "C1=CC(=O)C=CC1=O",
    "C1=C(C(=O)C=C(C1=O)O)O", "C1=CC=CC=C1", "C1=CC=C(C=C1)[N+](=O)[O-]", "C(C(C(F)F)(F)F)OC(C(F)F)(F)F", "CC(COC)N",
    "O", "CC1(CCCC(N1[O])(C)C)C",
]


metals_ood = ["[Rb+]", "[Co+2]", "[Y+3]",]
other_cations_ood = ["COCC[NH2+]CCOC"]
anions_ood = ["F[As-](F)(F)(F)(F)F", "[O-]P(=O)(F)F"]
neutrals_ood = ["O=C(N)C", "C(CO)O"]


def generate_solvated_mol(
    mol: Molecule | Atoms,
    charge: int,
    spin_multiplicity: int,
    species_smiles: List[str],
    architector_params: Dict = dict()
) -> Tuple[Atoms, int, int]:
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
        shell_charge (int): final charge of the solvated molecule
        shell_spin (int): final spin multiplicity of the solvated molecule
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
    shell_charge = int(binding[0].charge)
    shell_spin = int(binding[0].uhf) + 1
    
    return shell, shell_charge, shell_spin


def generate_full_solvation_shell(
    mol: Molecule | Atoms,
    charge: int,
    spin_multiplicity: int,
    solvent: str,
    min_added_atoms: int = 20,
    max_atom_budget: int = 200,
    architector_params: Dict = dict()
) -> Tuple[Atoms, int, int]:
    """
    Generate a solvation shell comprised of a single solvent around a molecule

    Args:
        mol (Molecule | Atoms): molecule to be solvated
        charge (int): charge of the core molecule
        spin_multiplicity (int): spin multiplicity of the core molecule
        solvent (str): SMILES for the solvent to surround the central molecule
        min_added_atoms (int): Minimum number of atoms to be allowed to add to this molecule. Default is 20.
        max_atom_budget (int): Maximum number of atoms allowed in this solvation shell. Default is 200.
        architector_params (Dict): parameters for Architector solvation shell generation

    Returns:
        shell (Atoms): molecule with solvation shell
        shell_charge (int): final charge of the solvated molecule
        shell_spin (int): final spin multiplicity of the solvated molecule
    """

    solvent_info = info_from_smiles([solvent])[solvent]

    if solvent_info["charge"] != 0:
        raise ValueError("generate_full_solvation_shell will only work for neutral solvents! Provided charge for"
                         f"{solvent}: {solvent_info['charge']}.")

    this_max_atoms = round(random.gauss(mu=50 + len(mol), sigma=40))
    this_max_atoms = max(this_max_atoms, len(mol) + min_added_atoms)
    this_max_atoms = min(this_max_atoms, max_atom_budget)
    
    budget = this_max_atoms - len(mol)
    num_solvent_mols = ceil(budget / solvent_info["num_atoms"])

    species_smiles = [solvent] * num_solvent_mols

    shell, shell_charge, shell_spin = generate_solvated_mol(
        mol, charge, spin_multiplicity, species_smiles, architector_params=architector_params
    )
    
    return shell, shell_charge, shell_spin
    

def generate_random_solvated_mol(
    mol: Molecule | Atoms,
    charge: int,
    spin_multiplicity: int,
    solvating_info: Dict[str, Dict],
    ood_solvating_info: Optional[Dict[str, Dict]] = None,
    min_added_atoms: int = 20,
    max_atom_budget: int = 200,
    max_trials: int = 25,
    weight_key: Optional[str] = "num_atoms",
    architector_params: Dict = dict()
) -> Tuple[Atoms, int, int]:
    """
    Generate (quasi)random solvated molecule using Architector.

    Cations, anions, and neutral species (e.g. solvents, additives) are placed around a central molecule.

    Args:
        mol (Molecule | Atoms): molecule to be solvated
        charge (int): charge of the core molecule
        spin_multiplicity (int): spin multiplicity of the core molecule
        solvating_info (Dict[str, int]): Map <SMILES>:info (number of atoms, charge, etc.) for potential solvating
            molecules
        ood_solvating_info (Optional[Dict[str, int]]): Map <SMILES>:info (number of atoms, charge, etc.) for potential
            out-of-distribution (OOD) solvating molecules. Default is None, meaning that the complex will be
            in-distribution and not contain OOD species
        min_added_atoms (int): Minimum number of atoms to be allowed to add to this molecule. Default is 20.
        max_atom_budget (int): Maximum number of atoms that can be in a complex. Default is 200.
        max_trials (int): Maximum number of attempts adding a solvating molecule
        weight_key (Optional[str]): If not None (default is "num_atoms"), possible solvating molecules will be weighted
            using the given key.
        architector_params (Dict): parameters for Architector solvation shell generation

    Returns:
        shell (Atoms): molecule with solvation shell
        shell_charge (int): final charge of the solvated molecule
        shell_spin (int): final spin multiplicity of the solvated molecule
    """

    # Select cap for number of atoms in this solvation shell
    # For now, using a normal (Gaussian distribution) with mean at (50 + len(mol)) atoms and stdev of 40
    # We then turn this continuous selection into an integer and make sure that it's within some reasonable bounds
    this_max_atoms = round(random.gauss(mu=50 + len(mol), sigma=40))
    this_max_atoms = max(this_max_atoms, len(mol) + min_added_atoms)
    this_max_atoms = min(this_max_atoms, max_atom_budget)
    
    if ood_solvating_info is not None:
        all_solvating_info = copy.deepcopy(solvating_info)
        all_solvating_info.update(ood_solvating_info)

    species_smiles = list()
    total_num_atoms = len(mol)
    total_charge = charge
    for i in range(max_trials):
        budget = this_max_atoms - total_num_atoms
        if budget < 1:
            break

        if ood_solvating_info is not None:
            if i == 0:
                possible_solvs = ood_solvating_info
            else:
                possible_solvs = all_solvating_info
        else:
            possible_solvs = solvating_info

        # Assign weights based on number of atoms
        choice_smiles = list()
        choice_weights = list()
        for smiles, data in possible_solvs.items():
            # Check that we don't try to add species of like charge
            if total_charge * data["charge"] > 0:
                continue

            # Is the potential solvating molecule too large?
            if total_num_atoms + data["num_atoms"] <= this_max_atoms:
                choice_smiles.append(smiles)
                if weight_key is not None:
                    choice_weights.append(1 / data[weight_key])

        # No valid molecules
        if len(choice_smiles) == 0:
            break

        if weight_key is not None:
            choice = random.choices(choice_smiles, weights=choice_weights, k=1)[0]
        else:
            choice = random.choices(choice_smiles, k=1)[0]

        choice_num_atoms = possible_solvs[choice]["num_atoms"]
        choice_charge = possible_solvs[choice]["charge"]
        
        species_smiles.append(choice)
        total_num_atoms += choice_num_atoms
        total_charge += choice_charge

    if len(species_smiles) == 0:
        return None

    shell, shell_charge, shell_spin = generate_solvated_mol(
        mol, charge, spin_multiplicity, species_smiles, architector_params=architector_params
    )
    return shell, shell_charge, shell_spin


def generate_random_dimers(
    mol: Molecule | Atoms,
    charge: int,
    spin_multiplicity: int,
    solvating_info: Dict[str, Dict],
    max_atom_budget: int = 200,
    num_selections: int = 5,
    architector_params: Dict = dict()
) -> List[Tuple[Atoms, int, int]]:
    """
    Generate (quasi)random dimers (central molecule + 1 counter ion or ion) using Architector.

    Args:
        mol (Molecule | Atoms): molecule to be solvated
        charge (int): charge of the core molecule
        spin_multiplicity (int): spin multiplicity of the core molecule
        solvating_info (Dict[str, int]): Map <SMILES>:info (number of atoms, charge, etc.) for potential solvating
            molecules
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

    # Make sure solvating molecules aren't too large and aren't of like charge with the central molecule
    real_candidates_names = [
        k for k, v in solvating_info.items()
        if v["num_atoms"] <= budget
        and charge * v["charge"] <= 0
    ]

    if not real_candidates_names:
        return list()
    elif len(real_candidates_names) < num_selections:
        choices = real_candidates_names
    else:
        choices = random.sample(real_candidates_names, k=num_selections)

    complexes = list()
    for candidate in choices:
        this_complex, this_complex_charge, this_complex_spin = generate_solvated_mol(
            mol,
            charge,
            spin_multiplicity,
            [solvating_info[candidate]["smiles"]],
            architector_params
        )
        complexes.append((this_complex, this_complex_charge, this_complex_spin))

    return complexes


def dump_xyzs(
    complexes: List[Tuple[Atoms, int, int]],
    prefix: str,
    path: Path
):
    """
    Create *.xyz files for each complex in a set of solvated complexes

    Args:
        complexes (List[Tuple[Atoms, int, int]]): Collection of solvated molecules. Each entry is a molecular
            structure, its charge, and its spin multiplicity
        prefix (str): Prefix for all *.xyz files
        path (Path): Path in which to dump *.xyz files

    Returns:
        None
    """

    path.mkdir(exist_ok=True)

    for ii, (atoms, charge, spin) in enumerate(complexes):
        write(str(path / f"{prefix}_solv{ii}_{charge}_{spin}.xyz"), atoms, format="xyz")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parameters for OMol24 (quasi)-random solvation")
    parser.add_argument('--xyz_dir', type=str, required=True, help="Directory containing input XYZ files")
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    # TODO
    parser.add_argument('--base_dir', type=str, default="dump", help="Output directory (default: 'dump')")
    
    parser.add_argument(
        '--num_dimers',
        type=int,
        default=3,
        help="Number of dimer structures to generate (default: 3)"
    )
    parser.add_argument(
        '--num_random_shells',
        type=int,
        default=1,
        help="Number of truly random solvation shells to generate (default: 1)"
    )
    parser.add_argument(
        '--max_core_molecule_size',
        type=int,
        default=50,
        help="Maximum size (number of atoms) for a molecule to serve as the core of a solvation shell (default: 50)"
    )
    parser.add_argument(
        '--max_atom_budget',
        type=int,
        default=60,
        help="Maximum size (number of atoms) of a solvation shell (default: 60)"
    )    
    
    args = parser.parse_args()

    # For reproducibility
    random.seed(args.seed)

    # TODO: CHANGE THESE
    # `xyz_dir` should point to a root-level directory with *.xyz files
    # *.xyz's can be nested multiple levels down - this script will recursively search for them
    xyz_dir = Path(args.xyz_dir)
    # `base_dir` should point to root-level directory where generated complexes will be dumped
    base_dir = Path(args.base_dir)
    ood_dir = base_dir / "ood"

    base_dir.mkdir(exist_ok=True)
    ood_dir.mkdir(exist_ok=True)

    # Set-up: get info from predefined set of SMILES
    solvating_info = info_from_smiles(
        metals + other_cations + anions + neutrals
    )

    just_solvent_info = info_from_smiles(
        neutrals
    )

    # Identify all molecules for solvation
    # NOTE: it's important the files have the format "<NAME>_<charge>_<spin>.xyz(.gz)"
    # Because otherwise, we can't identify charge/spin information from the *.xyz format
    xyz_files = [
        f for f in glob.glob(f'{xyz_dir.resolve().as_posix()}/**/*.xyz*', recursive=True)
        if f.endswith('.xyz') or f.endswith('.xyz.gz')
    ]
    print("TOTAL NUMBER OF XYZ FILES:", len(xyz_files))

    # For now (2024/06/13), the plan is to do the following for each molecule:
    # 1. Generate `m` random dimers with neutral molecules or molecules of opposite charge
    # 2. Generate 1 solvation shell with a random pure solvent from a pre-defined list
    # 3. Generate `n` random solvation shells with combinations of random combinations of components (ions,
    #    neutral species, etc.)

    num_dimers = args.num_dimers
    num_random_shells = args.num_random_shells # Note: this is currently unused
    max_core_molecule_size = args.max_core_molecule_size
    max_atom_budget = args.max_atom_budget

    # TODO: play around with these more
    # In initial testing, random placement seems to help better surround central molecule
    # Sella might be helpful, but also really slows things down
    architector_params={"species_location_method": "random"}

    for xyz_file in xyz_files:

        mol = Molecule.from_file(xyz_file)

        # If molecule is too large, don't bother trying to make solvation complexes
        if len(mol) > max_core_molecule_size:
            continue

        name = os.path.splitext(xyz_file)[0]
        subname = name.split("/")[-1]
        contents = name.split("_")
        charge = int(contents[-2])
        spin = int(contents[-1])

        mol.set_charge_and_spin(charge, spin)

        this_dir = base_dir / subname

        # In-distribution (ID) data

        filtered = list()

        # Step 1 - dimers
        complexes = generate_random_dimers(
            mol=mol,
            charge=charge,
            spin_multiplicity=spin,
            solvating_info=solvating_info,
            max_atom_budget=max_atom_budget,
            num_selections=num_dimers,
            architector_params=architector_params
        )

        # Step 2 - pure solvent shell
        # Pick random solvent
        solvent = random.choice(list(just_solvent_info))
        solvent_complex = generate_full_solvation_shell(
            mol=mol,
            charge=charge,
            spin_multiplicity=spin,
            solvent=solvent,
            max_atom_budget=max_atom_budget,
            architector_params=architector_params
        )
        complexes.append(solvent_complex)

        # Step 3 - random solvation shell
        random_complex = generate_random_solvated_mol(
            mol=mol,
            charge=charge,
            spin_multiplicity=spin,
            solvating_info=solvating_info,
            max_atom_budget=max_atom_budget,
            max_trials=10,
            architector_params=architector_params
        )

        # Possible that you'll end up with no complex
        if random_complex is not None:
            complexes.append(random_complex)

        # Make sure that structures are physically sound
        # Possible that the MD produces some wild structures, e.g. with atoms too close
        filtered = list()
        for comp in complexes:
            if validate_structure(comp[0].get_chemical_symbols(), comp[0].get_positions()):
                filtered.append(comp)            

        print(this_dir)
        # Dump new complexes as *.xyz files
        dump_xyzs(
            complexes=filtered,
            prefix=subname,
            path=this_dir,
        )

        # Out-of-distribution (OOD) data
        # We generate an OOD point for 10% of the input structures
        if random.random() > 0.1:
            continue

        ood_solvating_info = info_from_smiles(
            metals_ood + other_cations_ood + anions_ood + neutrals_ood
        )

        ood_just_solvent_info = info_from_smiles(
            neutrals_ood
        )

        # For each molecule, only select one of dimer, solvent shell, or random shell
        # TODO: should we ONLY be doing random shells?
        choice = random.choice([1, 2, 3])

        if choice == 1:
            ood_complex = generate_random_dimers(
                mol=mol,
                charge=charge,
                spin_multiplicity=spin,
                solvating_info=ood_solvating_info,
                max_atom_budget=max_atom_budget,
                num_selections=1,
                architector_params=architector_params
            )[0]
        elif choice == 2:
            ood_solvent = random.choice(list(ood_just_solvent_info.keys()))
            ood_complex = generate_full_solvation_shell(
                mol=mol,
                charge=charge,
                spin_multiplicity=spin,
                solvent=ood_solvent,
                max_atom_budget=max_atom_budget,
                architector_params=architector_params
            )
        else:
            ood_complex = generate_random_solvated_mol(
                mol=mol,
                charge=charge,
                spin_multiplicity=spin,
                solvating_info=solvating_info,
                ood_solvating_info=ood_solvating_info,
                max_atom_budget=max_atom_budget,
                max_trials=10,
                architector_params=architector_params
            )

        if validate_structure(ood_complex[0].get_chemical_symbols(), ood_complex[0].get_positions()):
            # Dump new complexes as *.xyz files
            dump_xyzs(
                complexes=[ood_complex],
                prefix=subname,
                path=(ood_dir / subname),
            )
