from collections import defaultdict
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from monty.serialization import dumpfn

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from openbabel import pybel

import scine_molassembler

from pymatgen.core.structure import Molecule

from omdata.electrolyte_utils import info_from_smiles


"""
This module provides tools to generate reasonable small molecules via functional group substitution.

It requires a collection of SMILES strings of templates and substituents. Templates should be labeled with
numeric sites (e.g. [1*]) where substitutions can occur. Substituents should not be so labeled.

This is also a stand-alone script to construct a collection of small molecules from predefined collections taken
from the literature and from our own design.
"""


ATTACH_PATTERN = re.compile(r"\[\d+\*\]")


templates_solvent_additive = {
    "cyclic_carbonate": "[1*]C([2*])1C([3*])([4*])OC(=O)O1",
    "linear_carbonate": "[1*]OC(=O)O[2*]",
    "glyme": "C([1*])([2*])([3*])OC([4*])([5*])C([6*])([7*])OC([8*])([9*])([10*])",
    "carboxylate": "C([1*])([2*])([3*])OC(=O)[4*]",
    "thf": "C([1*])([2*])1C([3*])([4*])C([5*])([6*])OC([7*])([8*])1",
    "cyclic_sulfate": "C([1*])([2*])1C([3*])([4*])OS(=O)(=O)O1", 
    "sulfone": "[1*]S(=O)(=O)[2*]",
    "sultone": "C([1*])([2*])1C([3*])([4*])C([5*])([6*])S(=O)(=O)OC([7*])([8*])1",
    "sulfite_ester": "[1*]OS(=O)O[2*]",
    "cyclic_sulfite_ester": "C1([1*])([2*])OS(=O)OC([3*])([4*])C([5*])([6*])C1([7*])([8*])",
    "sulfonyl_fluoride": "[1*]S(F)(=O)=O",
    "sulfamoyl_fluoride": "N([1*])([2*])S(=O)(=O)F",
    "dinitrile": "N#CC([1*])([2*])C([3*])([4*])C#N",
    "carbamate": "[1*]OC(=O)N([2*])([3*])",
    "methoxyalkylamine": "C([3*])([4*])([5*])OC([6*])([7*])C([8*])([9*])N([1*])([2*])",
    "phosphine": "P([1*])([2*])([3*])",
    "phosphorane": "P([1*])([2*])([3*])([4*])([5*])",
    "organophosphate": "P(=O)(O[1*])(O[2*])(O[3*])",
    "silane": "[Si]([1*])([2*])([3*])([4*])",
    "siloxane": "O([Si]([1*])([2*])[3*])[Si]([4*])([5*])[6*]",
    "borane": "B([1*])([2*])[3*]",
    "boroxine": "O1B([1*])OB([2*])OB([3*])1",
    "maleic_anhydride": "C1([1*])=C([2*])C(=O)OC1=O",
    "lactone": "C1([3*])([4*])C([1*])([2*])C(=O)OC([5*])([6*])1",
    "lactam": "C1([3*])([4*])C([1*])([2*])C(=O)NC([5*])([6*])1",

}

templates_ions = {
    "sulfonylimide": "[1*]S(=O)(=O)[N-]S(=O)(=O)[2*]",
    "organosulfate": "O=S(=O)([O-])O[1*]",
    "cyclic_borate": "O1C([3*])([4*])C(O[B-]([1*])([2*])1)([5*])([6*])",
    "cyclic_aluminate": "O1C([3*])([4*])C(O[Al-]([1*])([2*])1)([5*])([6*])",
    "cyclic_phosphate": "O1C([5*])([6*])C(O[P-]([1*])([2*])([3*])([4*])1)([7*])([8*])",
}

templates_redox_flow = {
    "anthraquinone": "O=C1c2c([1*])c([2*])c([3*])c([4*])c2C(=O)c3c([5*])c([6*])c([7*])c([8*])c13",
    "naphthoquinone": "O=C1c2c([1*])c([2*])c([3*])c([4*])c2C(=O)c([5*])c([6*])1",
    "benzoquinone": "C1([1*])=C([2*])C(=O)C([3*])=C([4*])C1=O",
    "tempo": "CC1(CC([1*])([2*])CC(N1[O])(C)C)C",
    "phthalimide": "O=C2c1c([2*])c([3*])c([4*])c([5*])c1C(=O)N([1*])2",
    "viologen": "[1*][n+]1ccc(cc1)c2cc[n+](cc2)[2*]",
    "quinoxaline": "c([1*])1c([2*])c([3*])c([4*])c2nc([5*])c([6*])nc12",
    "tetrazine": "C([1*])1=NN=C([2*])N=N1",
    "benzothiadizaole": "C([1*])1=C([2*])C2=NSN=C2C([3*])=C([4*])1",
    "pyridine_ester": "c(C(=O)(O[1*]))1c([2*])c([3*])[n+]([4*])c([5*])c([6*])1",
    "cyclopropenium": "[C+](N([1*])([2*]))2C(N([3*])([4*]))=C(N([5*])([6*]))2",
    "ptio": "[1*]C1(C([N+](=C(N1[O])C2=C([2*])C([3*])=C([4*])C([5*])=C([6*])2)[O-])([7*])[8*])[9*]",
    "phenothiazine": "c([4*])1c([3*])c([2*])c2c(c1([5*]))N([1*])c3c([6*])c([7*])c([8*])c([9*])c3S2",
}

templates_ilesw_cation = {
    "sulphonium": "[S+]([1*])([2*])([3*])",
    "ammonium": "[N+]([1*])([2*])([3*])([4*])",
    "phosphonium": "[P+]([1*])([2*])([3*])([4*])",
    "tetrazolium": "c([3*])1nnn([1*])[n+]([2*])1",
    "triazolium": "c([3*])1n([1*])nc([4*])[n+]([2*])1",
    "guanidinium": "N([1*])([2*])C(=[N+]([3*])([4*]))N([5*])([6*])",
    "imidazolium": "c([1*])1n([2*])c([3*])c([4*])[n+]([5*])1",
    "pyridazinium": "c([1*])1c([2*])c([3*])n[n+]([4*])c([5*])1",
    "122-triazole": "[N+]([1*])([2*])=c1n([3*])n([4*])c([5*])n1",
    "thiazolium": "C([1*])([2*])1C([3*])([4*])SC([5*])=[N+]([6*])1",
    "pyridinium": "c([1*])1c([2*])c([3*])c([4*])[n+]([5*])c([6*])1",
    "pyrroline": "C([1*])([2*])1C([3*])([4*])C([5*])([6*])C([7*])=[N+]([8*])1",
    "oxazolidinium": "C([1*])([2*])1OC([3*])([4*])C([5*])([6*])[N+]([7*])([8*])1",
    "imidazolidine": "[N+]([1*])([2*])=C1N([3*])C([4*])([5*])C([6*])([7*])N([8*])1",
    "pyrrolidinium": "C([1*])([2*])1C([3*])([4*])C([5*])([6*])C([7*])([8*])[N+]([9*])([10*])1",
    "oxadiazine": "[N+]([1*])([2*])=C1N([3*])C([4*])([5*])OC([6*])([7*])N([8*])1",
    "morpholinium": "C([1*])([2*])1C([3*])([4*])OC([5*])([6*])C([7*])([8*])[N+]([9*])([10*])1",
    "piperazinium": "C([1*])([2*])1C([3*])([4*])N([5*])C([6*])([7*])C([8*])([9*])[N+]([10*])([11*])1",
    "pyrimidine": "[N+]([1*])([2*])=C1N([3*])C([4*])([5*])C([6*])([7*])C([8*])([9*])N([10*])1",
    "piperidinium": "C([1*])([2*])1C([3*])([4*])C([5*])([6*])C([7*])([8*])[N+]([9*])([10*])C([11*])([12*])1",
    "iosquinolinium": "c([1*])1c([2*])c([3*])c2c(c([4*])1)c([5*])[n+]([6*])c([7*])c([8*])2"
}

templates_ilesw_anion = {
    "azanide": "[N-]([1*])([2*])",
    "methanide": "[C-]([1*])([2*])([3*])",
    "methanoate": "[O-]C([1*])=O",
    "tetrahydroborate": "[B-]([1*])([2*])([3*])([4*])",
    "tetrahydroaluminate": "[Al-]([1*])([2*])([3*])([4*])",
    "tetrahydrogalldate": "[Ga-]([1*])([2*])([3*])([4*])",
    "tetrahydroindate": "[In-]([1*])([2*])([3*])([4*])",
    "hydrogen_sulfite": "[O-]S([1*])(=O)=O",
    "glycinate": "[O-]C(=O)C([1*])([2*])N([3*])([4*])",
    "hexahydrophosphate": "[P-]([1*])([2*])([3*])([4*])([5*])([6*])",
    "hexahydroarsenate": "[As-]([1*])([2*])([3*])([4*])([5*])([6*])",
    "hexahydroniobate": "[Nb-]([1*])([2*])([3*])([4*])([5*])([6*])",
    "hexahydroantimonate": "[Sb-]([1*])([2*])([3*])([4*])([5*])([6*])",
    "acetate": "[O-]C(=O)[1*]",
    "hexahydrotantalate": "[Ta-]([1*])([2*])([3*])([4*])([5*])([6*])"
}

substituents = [
    'B(O)C',
    'B(O)CC',
    'B(O)N',
    'B(O)N(C)(C)',
    'B(O)O',
    'B(OC)O',
    'B(OCC)O',
    'Br',
    'C',
    'C#C',
    'C(=O)C(C(C(C(=O)O)(F)F)(F)F)(F)F',
    'C(=O)NC(=O)F',
    'C(Br)=C(Br)C',
    'C(Br)C(Br)C',
    'C(Br)CC',
    'C(C)CC',
    'C(C)CCC',
    'C(CC[Si](c1ccccc1)(C)C)(F)F',
    'C(Cl)=C(Cl)C',
    'C(Cl)C(Cl)C',
    'C(Cl)CC',
    'C(F)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)F',
    'C(F)(C(C(C(F)(F)F)(F)F)(F)F)F',
    'C(F)(C(C(F)(F)F)(F)F)F',
    'C(F)(C(F)(F)F)F',
    'C(F)(C(F)F)F',
    'C(F)(F)(F)',
    'C(F)C',
    'C(F)CC',
    'C(I)=C(I)C',
    'C(I)C(I)C',
    'C(I)CC',
    'C([O-])(=O)',
    'C(c1cccc(c1)C(F)(F)F)(F)F',
    'C=C',
    'CB(O)O',
    'CC',
    'CC#C',
    'CC(=N)',
    'CC(=N)C',
    'CC(=O)Br',
    'CC(=O)Cl',
    'CC(=O)F',
    'CC(=O)I',
    'CC(=O)N(C)(C)',
    'CC(=O)O',
    'CC(=O)OC(=O)C',
    'CC(=O)c1ccccc1',
    'CC(Br)C',
    'CC(C(F)(F)F)O',
    'CC(C)C',
    'CC(CN)O',
    'CC(Cl)C',
    'CC(F)C',
    'CC(I)C',
    'CC(N)(N)N',
    'CC(O)C',
    'CC(OCc1ccccc1)C',
    'CC1CO1',
    'CC=C',
    'CC=O',
    'CCC',
    'CCC(=C(F)F)F',
    'CCC(=O)N',
    'CCC(=O)Nc1ccccc1',
    'CCC(=O)O',
    'CCC(C(Br)(F)F)(Br)F',
    'CCC(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F',
    'CCC(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F',
    'CCC(C(C(C(F)(F)F)(F)F)(F)F)(F)F',
    'CCC(C(C(C(S(F)(F)(F)(F)F)(F)F)(F)F)(F)F)(F)F',
    'CCC(C(S(F)(F)(F)(F)F)(F)F)(F)F',
    'CCC(C)C',
    'CCC(CCC=C(C)C)C',
    'CCC(F)(F)F',
    'CCCBr',
    'CCCC',
    'CCCC#C',
    'CCCC(=O)O',
    'CCCC(=O)OCC',
    'CCCC(C(F)(F)F)(F)F',
    'CCCCC',
    'CCCCC(C(S(F)(F)(F)(F)F)(F)F)(F)F',
    'CCCCCC',
    'CCCCCCC',
    'CCCCCCCC',
    'CCCCCCCCC',
    'CCCCCCCCCC',
    'CCCCCCCCCCC',
    'CCCCCCCCCCCC',
    'CCCCCCCCCCCCC',
    'CCCCCCCCCCCCCC',
    'CCCCCCCCCCCCCCC',
    'CCCCCCCCCCCCCCCC',
    'CCCCCCCCCCCCCCCCCC',
    'CCCCCCCCCCCCCCCCCCCC',
    'CCCCCCCCCCCCOC',
    'CCCCCCCCCCCCn1cccc1',
    'CCCCCCCCCCCOC',
    'CCCCCCCCCCOC',
    'CCCCCCCCCOC',
    'CCCCCCCCOC',
    'CCCCCCCCOC(=O)C',
    'CCCCCCCC[Si](c1ccccc1)(C)C',
    'CCCCCCCOC',
    'CCCCCCCSc1nnco1',
    'CCCCCCOC',
    'CCCCCCOC(=O)C',
    'CCCCCCOC(=O)C=C',
    'CCCCCN',
    'CCCCCOC',
    'CCCCCOC(=O)C',
    'CCCCCOCC',
    'CCCCN',
    'CCCCN(C(=O)C)C',
    'CCCCN(C(=S)SCCC)CCCC',
    'CCCCNC(=N)N',
    'CCCCNC(=O)C',
    'CCCCO',
    'CCCCOC',
    'CCCCOC(=O)C',
    'CCCCOC=O',
    'CCCCS(=O)(=O)Cl',
    'CCCCS(=O)(=O)O',
    'CCCCSc1nnco1',
    'CCCCl',
    'CCCCn1cccc1',
    'CCCF',
    'CCCI',
    'CCCN',
    'CCCNCc1ccccc1O',
    'CCCO',
    'CCCOC',
    'CCCOC(=O)C',
    'CCCP(=O)(OCC)OCC',
    'CCCS',
    'CCCS(=O)(=O)O',
    'CCCSC(=S)N(CC)CC',
    'CCCSC(=S)N(CCCCCCCC)CCCCCCCC',
    'CCCSC(=S)N(c1ccccc1)c1ccccc1',
    'CCCSC(=S)N1CCOCC1',
    'CCCSC(=S)n1cncc1',
    'CCCc1ccccc1',
    'CCF',
    'CCN',
    'CCN(C(=O)C)CC',
    'CCO',
    'CCOC(=O)C',
    'CCOC(=O)CC',
    'CCOC=O',
    'CCOCC',
    'CCOCC(C(C(C(F)F)(F)F)(F)F)(F)F',
    'CCOCC(F)(F)F',
    'CCOCCOc1ccc(cc1)C(CC(C)(C)C)(C)C',
    'CCO[Si](C)(C)C',
    'CCP(=O)(OCC)OCC',
    'CCSc1nnco1',
    'CCc1ccccc1',
    'CN',
    'CNC',
    'COC',
    'COC(=O)C',
    'COC(F)(F)F',
    'COCC',
    'COCCC',
    'COCCOCC',
    'COCCOCCOCC',
    'COCOCCO',
    'CON=O',
    'COO',
    'CO[N+](=O)[O-]',
    'COc1cc(C)ccc1C(C)C',
    'COc1ccccc1',
    'CP(=O)(O)O',
    'CS(=O)=O',
    'CSCC',
    'Cc1ccc(cc1)C',
    'Cc1cccc(c1)C',
    'Cc1ccccc1',
    'Cc1ccccc1C',
    'Cl',
    'F',
    'I',
    'N',
    'N(C(=O)C)C(=O)C',
    'N(C(=O)F)C(=O)F',
    'N(C)C(=O)C',
    'N(O)O',
    'N=C(C)(C)',
    'N=C=O',
    'N=C=S',
    'NCC',
    'NCCC',
    'NCCCC',
    'NNN',
    'O',
    'OB(O)(O)',
    'OC#N',
    'OCC',
    'OCC(C(C)C)C',
    'OCC(CC(C)C)C',
    'OCC(O)C',
    'OCCC',
    'OCCCC',
    'OCCNCC(O)C',
    'OCCOCC',
    'ON=O',
    'ONO',
    'O[N+](=O)[O-]',
    'P(=O)(O)O',
    'P(C)C',
    'P(C)CC',
    'P(CC)CC',
    'P(O)Br',
    'P(O)C',
    'P(O)Cl',
    'P(O)I',
    'P(O)N',
    'P(O)O',
    'P=N',
    'P=O',
    'P=S',
    'PP(=O)(O)O',
    'S',
    '[O]C(=O)c1ccccc1',
    '[S](=O)(=O)(C(F)(F)(C(F)(F)(F)))',
    '[S](=O)(=O)C',
    '[S](=O)(=O)C(C(C(C(F)(F)F)(F)F)(F)F)(F)F',
    '[S](=O)(=O)C(C(C(F)(F)F)(F)F)(F)F',
    '[S](=O)(=O)C(F)(F)(F)',
    '[S](F)(=O)=O',
    '[S]CC(=O)OCC(OC(=O)CS)C',
    '[Si](C)(C)(C)',
    'c1ccccc1',
    'c1ccccn1',
    'c1cccnc1',
    'c1ccncc1'
]


def select_replacement_points(template: str) -> List[str]:
    """
    (Randomly) select points on a template to substitute. All non-selected substitution points will be filled by
    hydrogens ([H]).

    Args:
        template (str): SMILES string of the template to be substituted
    
    Returns:
        points_selected (List[str]): List of the labeled sites (e.g. [1*]) to be subject to substitution
    """
    
    attach_points = ATTACH_PATTERN.findall(template)

    num_points = len(attach_points)

    # Should never happen
    if num_points == 0:
        return None
    elif num_points == 1:
        # No choice needs to be made if you have only one possible point of attachment
        return attach_points
    else:
        num_to_choose = random.randint(1, num_points)

        points_selected = random.sample(attach_points, num_to_choose)
        return points_selected


def select_substituent(
    substituent_info: Dict[str, Any],
    sub_key: str,
    weight: bool = True,
    budget: Optional[int] = None) -> Mol:

    """
    (Randomly) select a substituent from a predefined collection.

    Args:
        substituent_info (Dict[str, Any]): Key-value pairs of possible substituents, where the keys are names (or
            SMILES strings) and the values are molecule representations, numbers of atoms, etc.
        sub_key (str): Key in substituent_info used for weighting and budgeting purposes. Could be "num_atoms" or
            "num_heavy_atoms"
        weight (bool): If True (default), use a weight function to prefer smaller substituents. Current weighing
            function is 1/n, where n is the number of atoms or number of heavy atoms
        budget (Optional[int]): Maximum size of substituent to be added, in terms of number of atoms or number of
            heavy atoms. Default is None, meaning that a substituent of any size can be added

    Returns:
        Mol: Chosen substituent

    """

    acceptable_subs = list()
    weights = list()
    for name, info in substituent_info.items():
        if budget is None or info[sub_key] < budget:
            acceptable_subs.append(info["rdkit_mol"])
            if weight:
                weights.append(1 / info[sub_key])

    if len(acceptable_subs) == 0:
        # If there are no substituents small enough, must replace with a hydrogen atom
        return Chem.MolFromSmiles("[H]")
    else:
        # Weight to prefer smaller substituents
        # TODO: is there a more clever way to do this?
        if weight:
            choice = random.choices(acceptable_subs, weights=weights, k=1)[0]
        else:
            choice = random.choices(acceptable_subs, k=1)[0]
    
    return choice


def generate_new_molecule(
    template: str,
    substituent_info: Dict[str, Any],
    sub_key: str,
    weight_subs: bool = True,
    max_atoms: Optional[int] = None) -> str:
    
    """
    Construct a new molecule by performing a functional group substitution.

    Args:
        template (str): SMILES string for a base template
        substituent_info (Dict[str, Any]): Key-value pairs of possible substituents, where the keys are names (or
            SMILES strings) and the values are molecule representations, numbers of atoms, etc.
        sub_key (str): Key in substituent_info used for weighting and budgeting purposes. Could be "num_atoms" or
            "num_heavy_atoms"
        weight_subs (bool): If True (default), use a weight function to prefer smaller substituents. Current weighing
            function is 1/n, where n is the number of atoms or number of heavy atoms
        max_atoms (Optional[int]): Maximum size of substituent to be added, in terms of number of atoms or number of
            heavy atoms. Default is None, meaning that a substituent of any size can be added

    Returns:
        SMILES string for the substituted molecule, with possibly multiple substitutions

    """
    
    points = select_replacement_points(template)

    # No valid points for substitution
    if points is None:
        return None

    mol = Chem.MolFromSmiles(template)

    # Perform substitutions iteratively
    # If the molecule gets too large, only H will be chosen
    for point in points:
        sub = select_substituent(
            substituent_info,
            sub_key,
            weight=weight_subs,
            budget=max_atoms
        )

        mol = Chem.ReplaceSubstructs(
            mol, 
            Chem.MolFromSmiles(point), 
            sub
        )[0]

    # Search for any remaining points that didn't see substitution, and replace them with H
    remaining_points = ATTACH_PATTERN.findall(Chem.MolToSmiles(mol))
    for point in remaining_points:
        mol = Chem.ReplaceSubstructs(
            mol, 
            Chem.MolFromSmiles(point), 
            Chem.MolFromSmiles("[H]")
        )[0]
    
    # Remove explicit hydrogens
    mol = Chem.RemoveHs(mol)

    return Chem.MolToSmiles(mol)


def generate_library(
    templates: Dict[str, str],
    substituents: List[str],
    attempts_per_template: int = 50,
    max_atoms: Optional[int] = None,
    max_heavy_atoms: Optional[int] = 50,
    dump_to: Optional[str | Path] = None,
) -> Dict[str, List[str]]:

    """
    Generate a collection of functionalized molecules from a collection of templates and a collection of substituents

    Args:
        templates (Dict[str, str]): Key-value pairs of templates, where the keys are names and the values are SMILES
            strings
        substituents (List[str]): List of SMILES strings for substituents
        attempts_per_template (int): How many substituted molecules should we try for each template? Default is 50
        max_atoms (Optional[int]): Maximum size of a substituted molecule, in terms of number of atoms. Default is
            None, meaning that there will be no hard upper limit on molecule size in terms of total number of atoms.
        max_heavy_atoms (Optional[int]): Maximum size of a substituted molecule, in terms of number of heavy (non-H)
            atoms. Default is 50, meaning that molecules with more than 50 heavy atoms will not be allowed.
        dump_to (Optional[str, Path]): If this is not None, then the collection of substituted molecules will be dumped
            to the specified path as a JSON of zipped JSON file

    Returns:
        library (Dict[str, List[str]]): Final collection of substituted molecules. Keys are template names, and values
            are lists of (substituted) SMILES strings
    """
    
    # Only total atoms or heavy atoms can be used to decide substitutions
    if max_atoms is not None and max_heavy_atoms is not None:
        raise ValueError("Both max_atoms and max_heavy_atoms were provided! Please provide only up to one criterion.")

    sub_info = info_from_smiles(substituents)

    if max_atoms is not None:
        do_weight = True
        maximum = max_atoms
        sub_key = "num_atoms"
    elif max_heavy_atoms is not None:
        do_weight = True
        maximum = max_heavy_atoms
        sub_key = "num_heavy_atoms"
    else:
        do_weight = False
        maximum = None
        sub_key = "num_atoms"

    library = defaultdict(list)

    for temp_name, temp_smiles in templates.items():
        for _ in range(attempts_per_template):
            library[temp_name].append(
                generate_new_molecule(
                    temp_smiles,
                    sub_info,
                    sub_key,
                    do_weight,
                    maximum
                )
            )

    if dump_to is not None:
        dumpfn(dict(library), dump_to)

    return library


def filter_library(
    library: Dict[str, List[str]]
) -> Dict[str, List[str]]:

    """
    Check for and remove duplicate InChIs in the libary

    Args:
        library (Dict[str, List[str]]): Collection of substituted molecules organized by template name
    
    Returns:
        filtered (Dict[str, List[str]]): Collection of substituted molecules with duplicates removed
    """

    inchis_charges = set()

    filtered = dict()

    for name, smiles in library.items():
        if name not in filtered:
            filtered[name] = list()
        
        for smi in smiles:
            mol = pybel.readstring("smi", smi)
            charge = mol.charge
            inchi = mol.write(format="inchi")
            if (inchi, charge) not in inchis_charges:
                filtered[name].append(smi)
                inchis_charges.add((inchi, charge))

    return filtered


def validate_structure(species: List[str], coords: np.ndarray) -> bool:
    """
    Check if any atoms in the molecule are too close together

    Args:
        species (List[str]): List of atomic elements with length N, where N is the number of atoms
        Coords (np.ndarray): Atomic positions as an Nx3 matrix, where N is the number of atoms

    Returns:
        bool. Is this molecule valid?
    """

    if isinstance(coords, scine_molassembler.dg.Error):
        return False

    pmg_mol = Molecule(species, coords)

    return pmg_mol.is_valid(tol=0.9)


def dump_xyzs(
    library: Dict[str, List[str]],
    base_dir: str | Path,
    validate: bool = True
):
    """
    Create XYZs for each molecule in a library of substituted molecules

    Args:
        library (Dict[str, List[str]]): Collection of substituted molecules organized by template name
        base_dir (str | Path): Path in which to dump XYZ files
        validate (bool): Should molecular structures be validated? Default is True

    Returns:
        None
    """

    # TODO: conformer selection?

    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    # If the directory doesn't exist, make it
    base_dir.mkdir(exist_ok=True)

    for name, smiles in library.items():
        path = base_dir / name
        path.mkdir(exist_ok=True)

        for ii, smi in enumerate(smiles):
            mol = scine_molassembler.io.experimental.from_smiles(smi)
            ensemble = scine_molassembler.dg.generate_random_ensemble(mol, 25)

            pos = None
            for member in ensemble:
                if not isinstance(member, scine_molassembler.dg.Error):
                    pos = member
                    break

            # No member of the ensemble passed Molassembler internal validation
            if pos is None:
                continue

            if validate:
                species = [str(e) for e in mol.graph.elements()]
                if not validate_structure(species, pos):
                    continue

            pbmol = pybel.readstring("smi", smi)
            pbmol.addh()
            charge = pbmol.charge
            spin = pbmol.spin

            scine_molassembler.io.write((path / f"mol{ii}_{charge}_{spin}.xyz").resolve().as_posix(), mol, pos)


if __name__ == "__main__":
    
    # For reproducibility
    random.seed(42)

    # Combine all of the templates
    templates = templates_solvent_additive
    templates.update(templates_ions)
    templates.update(templates_redox_flow)
    templates.update(templates_ilesw_cation)
    templates.update(templates_ilesw_anion)

    print("TOTAL NUMBER OF TEMPLATES:", len(templates))

    # Combine all substituents
    substituents = list(set(substituents))

    print("TOTAL NUMBER OF SUBSTITUENTS:", len(substituents))

    # dump_path = Path("dump")
    dump_path = Path("/home/ewcss/data/omol24/20240521_small_mol_dump")
    # If the directory doesn't exist, make it
    dump_path.mkdir(exist_ok=True)

    # Generate library based on functional group substitution
    library = generate_library(
        templates=templates,
        substituents=substituents,
        attempts_per_template=50,
        max_atoms=None,
        max_heavy_atoms=50,
        dump_to=dump_path / "initial_library_smiles.json",
    )

    # Filter using InChI to remove duplicates
    filtered_library = filter_library(
        library
    )

    dump_xyzs(
        library=filtered_library,
        base_dir=dump_path / "xyzs"
    )