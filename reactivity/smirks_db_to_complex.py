import argparse
import csv
import math
import os
from typing import List, Tuple

import numpy as np
from more_itertools import collapse
from rdkit import Chem
from rdkit.Chem import AllChem
from schrodinger.adapter import to_structure
from schrodinger.application.jaguar.autots_input import AutoTSInput
from schrodinger.application.jaguar.autots_rmsd import \
    reform_barely_broken_bonds
from schrodinger.application.jaguar.file_logger import FileLogger
from schrodinger.application.jaguar.packages.autots_modules.active_bonds import \
    mark_active_bonds
from schrodinger.application.jaguar.packages.autots_modules.autots_stereochemistry import \
    ChiralityMismatchError
from schrodinger.application.jaguar.packages.autots_modules.complex_formation import (
    _add_atom_transfer_dummies, _remove_atom_transfer_dummies,
    minimize_path_distance, reaction_center, translate_close)
from schrodinger.application.jaguar.packages.autots_modules.renumber import \
    build_reaction_complex
from schrodinger.application.jaguar.packages.reaction_mapping import \
    build_reaction_complex as get_renumbered_complex
from schrodinger.application.jaguar.packages.reaction_mapping import (
    flatten_st_list, get_net_matter)
from schrodinger.infra import fast3d
from schrodinger.structure import Structure, StructureWriter
from schrodinger.structutils import transform

"""
Convert Reaction SMIRKS (from RMechDB/PMechDB) to 3D fully mapped complexes.
"""


class local_rinp(AutoTSInput):
    """
    Need a patched version that I can populated programatically easily
    """

    def __init__(self, reactants, products):
        super().__init__()
        self.reactants = reactants
        self.products = products

    def getReactants(self):
        return self.reactants

    def getProducts(self):
        return self.products


def invert_structures(products):
    for st in products:
        refl_mat = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        transform.transform_structure(st, refl_mat)


def build_complexes(
    reactants: List[Structure], products: List[Structure]
) -> Tuple[Structure, Structure]:
    """
    Build the reaction complexes.

    This tries to use the built-in Schrodinger tools but if you don't
    have full licenses (i.e. the academic version), you might not be
    able to use it. Hence, I've also written a minimal version that will
    at least work if not as well as the Schrodinger one.
    """

    rinp = local_rinp(reactants, products)
    rinp.values.debug = False
    reactants, products = mark_active_bonds(
        reactants, products, max_n_active_bonds=6, water_wire=False
    )

    try:
        with FileLogger("logger", True):
            reactant_complex, product_complex = build_reaction_complex(
                reactants, products, rinp
            )
    except ChiralityMismatchError as e:
        print(
            "Bad stereoguess, inverting everything in hopes that fixes it (i.e. no diasteromers)..."
        )
        invert_structures(products)
        try:
            with FileLogger("logger", True):
                reactant_complex, product_complex = build_reaction_complex(
                    reactants, products, rinp
                )
        except ChiralityMismatchError as e:
            print(e)
            print(
                "There is still a chirality problem so there must be "
                "multiple stereocenters and only some are set wrong. We "
                "could go through and fix it but that's more effort than "
                "it's worth. This will not end well if we are going to "
                "do an interpolation with chirality changes so skip this."
            )
            raise
        except Exception as e:
            print(e)
            print(
                "getting renumbered complexes but can't do assembly. This is probably a license issue"
            )
            reactants = flatten_st_list(reactants)
            products = flatten_st_list(products)
            reactant_complex, product_complex = get_renumbered_complex(
                reactants, products
            )
            reactant_complex, product_complex = minimal_form_reaction_complex(
                reactant_complex, product_complex, rinp
            )
    except Exception as e:
        print(e)
        print(
            "getting renumbered complexes but can't do assembly. This is probably a license issue"
        )
        reactants = flatten_st_list(reactants)
        products = flatten_st_list(products)
        reactant_complex, product_complex = get_renumbered_complex(reactants, products)
        reactant_complex, product_complex = minimal_form_reaction_complex(
            reactant_complex, product_complex, rinp
        )

    return reactant_complex, product_complex


def minimal_form_reaction_complex(
    reactant: Structure, product: Structure, rinp: local_rinp
) -> Tuple[Structure, Structure]:
    reactant_out = reactant.copy()
    product_out = product.copy()

    rxn_center = reaction_center(reactant, product)

    # add dummy atoms to handle single atom transfers
    added_dummies = _add_atom_transfer_dummies(reactant_out, product_out)

    reactant_out, product_out = minimize_path_distance(
        reactant_out,
        product_out,
        indep_only=True,
        check_stab=rinp.values.check_alignment_stability,
    )

    translate_close(
        reactant_out, rxn_center=rxn_center, vdw_scale=rinp.values.vdw_scale
    )
    translate_close(product_out, rxn_center=rxn_center, vdw_scale=rinp.values.vdw_scale)
    # remove dummy atoms before returning
    _remove_atom_transfer_dummies(reactant_out)
    _remove_atom_transfer_dummies(product_out)

    return reactant_out, product_out


def get_rxn_list(smirks_list):
    rxn_list = []
    for rxn_smirks in smirks_list:
        rxn = AllChem.ReactionFromSmarts(rxn_smirks, useSmiles=True)
        try:
            reactants = [to_structure(mol) for mol in rxn.GetReactants()]
        except ValueError:
            print("R", rxn_smirks)
            continue
        try:
            products = [to_structure(mol) for mol in rxn.GetProducts()]
        except ValueError:
            print("P", rxn_smirks)
            continue
        rxn_st = []

        # *MechDBs sometimes include molecules with no mapped atoms which
        # seem to be spectators. We exclude these molecules from the reaction
        # complexes
        for st_list in (reactants, products):
            rxn_st.append(
                [
                    st
                    for st in st_list
                    if any("i_rdkit_molAtomMapNumber" in at.property for at in st.atom)
                ]
            )
        rxn_list.append(rxn_st)
    return rxn_list


# nh3I = Chem.MolFromSmiles('[NH3+]I.[Cl-]')
# nh3 = Chem.MolFromSmiles('N')
# I = Chem.MolFromSmiles('ICl')
# nh3I = rdkit_adapter.from_rdkit(nh3I)
# nh3 = rdkit_adapter.from_rdkit(nh3)
# I = rdkit_adapter.from_rdkit(I)
# rxn_list = {'name':([nh3I],[nh3,I])}
# rxn_smirks = "[Li:11][CH2:10]CCC[CH:20]=[CH:21][C:22](=[O:23])OC(C)(C)C.CCCCI>>[Li+:11].CCCCI.CC(C)(C)O[C:22](=[CH:21][CH:20]1[CH2:10]CCC1)[O-:23] 10"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=".")
    parser.add_argument("--batch_idx", default=0, type=int)
    parser.add_argument("--n_batch", default=1, type=int)
    parser.add_argument("--db_name", type=str)
    return parser.parse_args()


def main(n_batch, batch_idx, output_path, db_name):
    with open(f"{db_name}.csv", "r") as fh:
        csvreader = csv.reader(fh)
        smirks_list = [row[0] for row in csvreader]
    batch_size = math.ceil(len(smirks_list) / n_batch)
    smirks_list = smirks_list[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    fast3d_volumizer = fast3d.Volumizer()

    rxn_list = get_rxn_list(smirks_list)

    for idx, rxn in enumerate(rxn_list, start=batch_idx * batch_size):
        net_matter = get_net_matter(flatten_st_list(rxn[0]), flatten_st_list(rxn[1]))
        if any(f for f in net_matter):
            print("Reaction does not conserve mattter, will not do")
            print(smirks_list[idx - batch_idx * batch_size])
            print(net_matter)
            continue
        output_name = os.path.join(output_path, f"{db_name}_{idx}.sdf")
        if os.path.exists(output_name):
            continue
        for st in collapse(rxn):
            fast3d_volumizer.run(st, False, True)
        try:
            r, p = build_complexes(*rxn)
        except Exception as e:
            print(e)
            print(f"problem with reaction {idx}")
            print(smirks_list[idx - batch_idx * batch_size])
            continue
        else:
            if reform_barely_broken_bonds(r, p):
                # If the reaction does not change the molecular graph
                # (e.g. resonance structures are included in some of
                # these databases) then we don't include them.
                print(f"reaction {idx} is a no-op")
                print(smirks_list[idx - batch_idx * batch_size])
                continue
            # Stick the total charge in the comment line of the .xyz
            r.title = f"charge={r.formal_charge}"
            p.title = f"charge={p.formal_charge}"
            with StructureWriter(output_name) as writer:
                writer.extend([r, p])


if __name__ == "__main__":
    args = parse_args()
    main(args.n_batch, args.batch_idx, args.output_path, args.db_name)
