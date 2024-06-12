import copy
import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple, Union

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


cations = [
    "[OH3+]", "[Li+]", "[Na+]", "[K+]", "[Cs+]", "[Ti+]", "[Cu+]", "[Ag+]", "O=[V+]=O", "[Rb+]", "[NH4+]",
    "CCCC[N+]1(CCCC1)C", "CCN1C=C[N+](=C1)C", "CCC[N+]1(C)CCCC1", "CCC[N+]1(CCCCC1)C", "CC[N+](C)(CC)CCOC",
    "CCCC[P+](CCCC)(CCCC)CCCC", "CCCC[N+]1(CCCC1)CCC", "COCC[NH2+]CCOC", "CC(=O)[NH2+]C", "CC(COC)[NH3+]",
    "C[N+](C)(C)CCO", "CC1(CCCC(N1[O+])(C)C)C", "[Ca+2]", "[Mg+2]", "[Zn+2]", "[Be+2]", "[Cu+2]", "[Ni+2]", "[Pt+2]",
    "[Co+2]", "[Pd+2]", "[Ag+2]", "[Mn+2]", "[Hg+2]", "[Cd+2]", "[Yb+2]", "[Sn+2]", "[Pb+2]", "[Eu+2]", "[Sm+2]",
    "[Ra+2]", "[Cr+2]", "[Fe+2]", "O=[V+2]", "[V+2]", "[Ba+2]", "[Sr+2]", "C[N+]1=CC=C(C=C1)C2=CC=[N+](C=C2)C",
    "[Al+3]", "[Cr+3]", "[V+3]", "[Ce+3]", "[Ce+4]", "[Fe+3]", "[In+3]", "[Tl+3]", "[Y+3]", "[La+3]", "[Pr+3]",
    "[Nd+3]", "[Sm+3]", "[Eu+3]", "[Gd+3]", "[Tb+3]", "[Dy+3]", "[Er+3]", "[Tm+3]", "[Lu+3]", "[Hf+4]","[Zr+4]"
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